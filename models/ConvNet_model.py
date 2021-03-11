from __future__ import print_function
import argparse
from typing import ValuesView
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import weight_norm
from cp_norm import cp_norm, estimate_rank
from torch.utils.tensorboard.writer import SummaryWriter

class Net(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    # Standard parameter total. 1,199,882
    # CPNorm parameter total. 1,100,308
    def __init__(self, cpnorm=False, wnorm=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
# 70% reconstruction
        #if cpnorm is True and wnorm is False:
        #    self.conv1 = cp_norm(self.conv1, rank=7)
        #    self.conv2 = cp_norm(self.conv2, rank=141)
        #    self.fc1 = cp_norm(self.fc1, rank=116)
        #    self.fc2 = cp_norm(self.fc2, rank=9)
# Full rank
        if cpnorm is True and wnorm is False:
            self.conv1 = cp_norm(self.conv1, rank=16)
            self.conv2 = cp_norm(self.conv2, rank=261)
            self.fc1 = cp_norm(self.fc1, rank=136)
            self.fc2 = cp_norm(self.fc2, rank=16)

        elif cpnorm is False and wnorm is True:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)
            self.fc1 = weight_norm(self.fc1)
            self.fc2 = weight_norm(self.fc2)
            
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output