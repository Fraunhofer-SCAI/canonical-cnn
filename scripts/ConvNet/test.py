from __future__ import print_function, absolute_import
import argparse
from typing import ValuesView
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from ptflops import get_model_complexity_info
from torchscan import summary
from cp_compress import apply_compression
from models.ConvNet_model import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('deivce: ', device, flush=True)

test_kwargs = {'batch_size': 1000}
if device == 'cuda':
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': True}
    test_kwargs.update(cuda_kwargs)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
test_set = datasets.MNIST('./data', train=False,
                    transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)
net_model = Net(cpnorm=True).to(device)
compress_rate=75
if compress_rate != 0:
    print('Running compression.....', flush=True)
    net_model = apply_compression(net_model, compress_rate)

net_model.load_state_dict(torch.load("./mnist_cnn_rmsprop_crate-75.pt", map_location=device))
net_model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        start = time.perf_counter()
        output = net_model(data)
        end = time.perf_counter()
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print('Accuracy: ', accuracy, flush=True)
print('Time: ', end-start, flush=True)

macs, params = get_model_complexity_info(net_model, (1, 28, 28), as_strings=True,
                                        print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

print()
print()
print(summary(net_model, (1, 28, 28), max_depth=1))
