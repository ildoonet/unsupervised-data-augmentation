# UDA : Unsupervised Data Augmentation

Unofficial PyTorch Implementation of [Unsupervised Data Augmentation](https://arxiv.org/abs/1904.12848v1).

- Experiments on Text Dataset need to be done. Any Pull-Requests would be appreciated.
- Augmentation policies for SVHN, Imagenet using AutoAugment are not available publicly. We use policies from [Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment).

Most of codes are from [Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment).

## Introduction

todo.

## Run

```
$ python train.py -c confs/wresnet28x2.yaml --unsupervised
```

## Experiments

### Cifar10 (Reduced, 4k dataset)

#### Reproduce Paper's Result

| WResNet 28x2 | Paper    | Our(Top1 Err) |
|--------------|---------:|---------:|
| Supervised   | 20.26    | 21.30    |
| AutoAugment  | 14.1*    | 18.37    |
| UDA          | 5.27     | 7.28     |

#### Ablation

##### Batch size for unsupervised data

| WResNet 28x2   | Top1 Err |
|----------------|---------:|
| 0 (supervised) | 18.09    |
| batch=32       | 15.70    |
| batch=64       | 13.52    |
| batch=128      | 11.51    |
| batch=256      | 11.24    |
| batch=512      | 10.07    |
| batch=960      | 10.12    |

Settings

- decay : 0.0005
- ratio_unsup : 5.0
- epoch : 200
- lr : 0.03

Seems to be better if we utilize more unsupervised data(large batch size).

##### Loss ratio(between supervised and unsupervised) with small unsupervised batch size(32)

| WResNet 28x2   | Top1 Err |          |
|----------------|---------:|----------|
| ratio=0 (supervised) | 18.09    |
| ratio=1.0      | 16.03    |
| ratio=2.0      | 14.45    |
| ratio=5.0      | 15.70    |
| gradual(->5.0) | 14.70    |
| gradual(->5.0) | 13.30    | batch=64 |

Settings

- decay : 0.0005
- batch_unsup : 32
- epoch : 200
- lr : 0.03

todo.

### SVHN

todo.

### ImageNet

todo.

## References

- Unsupervised Data Augmentation : https://arxiv.org/abs/1904.12848v1
  - Official Tensorflow Implementation : https://github.com/google-research/uda
- Fast AutoAugment : https://github.com/kakaobrain/fast-autoaugment

