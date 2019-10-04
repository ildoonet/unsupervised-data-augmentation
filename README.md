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

| WResNet 28x2 | Paper    | Our Converged(Top1 Err) | Our Best(Top1 Err) | 
|--------------|---------:|---------:|---------:|
| Supervised   | 20.26    | 21.30    | 
| AutoAugment  | 14.1*    | 15.4     | 13.4     |
| UDA          | 5.27     | 6.58     | 6.27     |

### SVHN

todo.

### ImageNet

todo.

## References

- Unsupervised Data Augmentation : https://arxiv.org/abs/1904.12848v1
  - Official Tensorflow Implementation : https://github.com/google-research/uda
- Fast AutoAugment : https://github.com/kakaobrain/fast-autoaugment
