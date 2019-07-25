import itertools
import logging
import math
import os
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.nn.functional import kl_div, softmax, log_softmax

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser

from common import get_logger
from data import get_dataloaders
from metrics import accuracy, Accumulator
from networks import get_model, num_class

from warmup_scheduler import GradualWarmupScheduler


logger = get_logger('Unsupervised Data Augmentation')
logger.setLevel(logging.INFO)

best_valid_top1 = 0


def run_epoch(model, loader_s, loader_u, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, unsupervised=False, scheduler=None):
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))
    if verbose:
        loader_s = tqdm(loader_s, disable=tqdm_disable)
        loader_s.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))

    iter_u = iter(loader_u)

    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader_s)
    steps = 0
    for data, label in loader_s:
        steps += 1

        if not unsupervised:
            data, label = data.cuda(), label.cuda()
            preds = model(data)
            loss = loss_fn(preds, label)  # loss for supervised learning
        else:
            label = label.cuda()
            try:
                unlabel1, unlabel2 = next(iter_u)
            except StopIteration:
                iter_u = iter(loader_u)
                unlabel1, unlabel2 = next(iter_u)
            data_all = torch.cat([data, unlabel1, unlabel2]).cuda()

            preds_all = model(data_all)
            preds = preds_all[:len(data)]
            loss = loss_fn(preds, label)  # loss for supervised learning

            preds_unsup = preds_all[len(data):]
            preds1, preds2 = torch.chunk(preds_unsup, 2)
            preds1 = softmax(preds1, dim=1).detach()
            preds2 = log_softmax(preds2, dim=1)
            assert len(preds1) == len(preds2) == C.get()['batch_unsup']

            loss_kldiv = kl_div(preds2, preds1, reduction='none')    # loss for unsupervised
            loss_kldiv = torch.sum(loss_kldiv, dim=1)
            assert len(loss_kldiv) == len(unlabel1)
            # loss += (epoch / 200. * C.get()['ratio_unsup']) * torch.mean(loss_kldiv)
            if C.get()['ratio_mode'] == 'constant':
                loss += C.get()['ratio_unsup'] * torch.mean(loss_kldiv)
            elif C.get()['ratio_mode'] == 'gradual':
                loss += (epoch / float(C.get()['epoch'])) * C.get()['ratio_unsup'] * torch.mean(loss_kldiv)
            else:
                raise ValueError

        if optimizer:
            loss.backward()
            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))

            optimizer.step()
            optimizer.zero_grad()

        top1, top5 = accuracy(preds, label, (1, 5))

        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader_s.set_postfix(postfix)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)

        del preds, loss, top1, top5, data, label

    if tqdm_disable:
        logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics


def train_and_eval(tag, dataroot, metric='last', save_path=None, only_eval=False, unsupervised=False):
    max_epoch = C.get()['epoch']
    trainloader, unsuploader, testloader = get_dataloaders(C.get()['dataset'], C.get()['batch'], C.get()['batch_unsup'], dataroot)

    # create a model & an optimizer
    model = get_model(C.get()['model'], num_class(C.get()['dataset']), data_parallel=True)

    criterion = nn.CrossEntropyLoss()
    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer'].get('momentum', 0.9),
            weight_decay=C.get()['optimizer']['decay'],
            nesterov=C.get()['optimizer']['nesterov']
        )
    else:
        raise ValueError('invalid optimizer type=%s' % C.get()['optimizer']['type'])

    lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        t_max = C.get()['epoch']
        if C.get()['lr_schedule'].get('warmup', None):
            t_max -= C.get()['lr_schedule']['warmup']['epoch']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.)
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if C.get()['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    if not tag.strip():
        from metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided, no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter
    writers = [SummaryWriter(logdir='./logs/%s/%s' % (tag, x)) for x in ['train', 'test']]

    result = OrderedDict()
    epoch_start = 1
    if save_path and os.path.exists(save_path):
        data = torch.load(save_path)
        model.load_state_dict(data['model'])
        optimizer.load_state_dict(data['optimizer'])
        epoch_start = data['epoch']

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        rs['test'] = run_epoch(model, testloader, unsuploader, criterion, None, desc_default='*test', epoch=epoch_start, writer=writers[1])
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'test']):
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result

    # train loop
    global best_valid_top1
    best_valid_loss = 10e10
    for epoch in range(epoch_start, max_epoch + 1):
        model.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, unsuploader, criterion, optimizer, desc_default='train', epoch=epoch, writer=writers[0], verbose=True, unsupervised=unsupervised, scheduler=scheduler)
        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        model.eval()
        if epoch % (10 if 'cifar' in C.get()['dataset'] else 30) == 0 or epoch == max_epoch:
            rs['test'] = run_epoch(model, testloader, unsuploader, criterion, None, desc_default='*test', epoch=epoch, writer=writers[1], verbose=True)

            if best_valid_top1 < rs['test']['top1']:
                best_valid_top1 = rs['test']['top1']

            if metric == 'last' or rs[metric]['loss'] < best_valid_loss:    # TODO
                if metric != 'last':
                    best_valid_loss = rs[metric]['loss']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                writers[1].add_scalar('test_top1/best', rs['test']['top1'], epoch)

            # save checkpoint
            if save_path:
                logger.info('save model@%d to %s' % (epoch, save_path))
                torch.save({
                    'epoch': epoch,
                    'log': {
                        'train': rs['train'].get_dict(),
                        'test': rs['test'].get_dict(),
                    },
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict()
                }, save_path)

    del model

    return result


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--decay', type=float, default=-1)
    parser.add_argument('--unsupervised', action='store_true')
    parser.add_argument('--only-eval', action='store_true')
    args = parser.parse_args()

    assert (args.only_eval and not args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    if args.decay > 0:
        logger.info('decay reset=%.8f' % args.decay)
        C.get()['optimizer']['decay'] = args.decay
    if args.save:
        logger.info('checkpoint will be saved at %s', args.save)
    logger.info('unsupervsed=%s', args.unsupervised)

    import time
    t = time.time()
    result = train_and_eval(args.tag, args.dataroot, save_path=args.save, only_eval=args.only_eval, unsupervised=args.unsupervised)
    elapsed = time.time() - t

    logger.info('training done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('augmentation: %s' % C.get()['aug'])
    logger.info(result)
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
    logger.info('best top1 error in testset: %.4f' % (1. - best_valid_top1))
