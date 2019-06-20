# UDA : Unsupervised Data Augmentation

Unofficial Implementation of [Unsupervised Data Augmentation](https://arxiv.org/abs/1904.12848v1).

- Experiments on Text Dataset need to be done. Any Pull-Requests would be appreciated.
- Augmentation policies for SVHN, Imagenet using AutoAugment are not available publicly. We use policies from [Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment).

Most of codes are from [Fast AutoAugment](https://github.com/kakaobrain/fast-autoaugment).

## Introduction

todo.

## Experiments

### Cifar10 (Reduced, 4k dataset)

| WResNet 28x2 | Paper    | Top1 Err |
|--------------|---------:|---------:|
| Supervised   | 20.26    | 21.30    |
| AutoAugment  | 14.1*    | 18.37    |
| UDA          | 5.27     | 9.47     |

todo.

### SVHN

todo.

### ImageNet

todo.

## References

- Unsupervised Data Augmentation : https://arxiv.org/abs/1904.12848v1
- Fast AutoAugment : https://github.com/kakaobrain/fast-autoaugment

