import torch
from pretrainedmodels import models

from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn

from networks.wideresnet import WideResNet


def get_model(conf, num_class=10, data_parallel=True):
    name = conf['type']

    if name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_2':
        model = WideResNet(28, 2, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_class)

    else:
        raise NameError('no model named, %s' % name)

    if data_parallel:
        model = model.cuda()
        model = DataParallel(model)
    else:
        import horovod.torch as hvd
        device = torch.device('cuda', hvd.local_rank())
        model = model.to(device)
    cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]
