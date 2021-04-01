# Canonical convolutional neural networks
This is the official PyTorch implementation of Canonical convolutional neural networks.

## Overview

### Abstract
This paper formulates canonical weight normalization for convolutional neural networks. Canonical networks express their weight tensors as scaled sums of outer vector products. The canonical tensor decomposition inspires our formulation and serves as an initialization tool. We train network weights in the decomposed form. Similar to established weight normalization, we include a global scaling parameter and add scales for each mode. Our formulation allows us to compress our models conveniently by truncating the parameter sums. We find thatour re-parametrization leads to competitive normalization performance on the MNIST, CIFAR10, and SVHN data sets. Once training has convergenced, we find that our formulation simplifies network compression.

### Reparameterization
![CPNorm_Image](Images/cp_norm.png)

## Code Usage
### Requirements

Install the following required packages
1. Numpy - 1.18.5
2. Python - 3.8.5
3. Tensorly - 0.5.1
4. Torch - 1.7.0
5. Torchvision - 0.8.1

Clone the repository
### Training 
Navigate to scripts and AlexNet
``` bash
 $ cd scripts/AlexNet
``` 
For training the model
``` bash
$ python alexnet.py --lr=<lr>
```
