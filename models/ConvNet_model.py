# ====================================================================
# Implemented LeNet like model
# ====================================================================

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from src.cp_norm import cp_norm

class Net(nn.Module):
    """
    LeNet like achitecture & its forward pass

    Methods:
        forward: A forward pass for the network
    
    Args:
        
    """
    def __init__(self, cpnorm=False, wnorm=False):
        super(Net, self).__init__()
        # Layers initialization
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        # CP Normalization with full rank
        if cpnorm is True and wnorm is False:
            self.conv1 = cp_norm(self.conv1, rank=11)
            self.conv2 = cp_norm(self.conv2, rank=270)
            self.fc1 = cp_norm(self.fc1, rank=128)
            self.fc2 = cp_norm(self.fc2, rank=10)
        # Weight normalization
        elif cpnorm is False and wnorm is True:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)
            self.fc1 = weight_norm(self.fc1)
            self.fc2 = weight_norm(self.fc2)
            
    def forward(self, x):
        """
        Forward pass connections for the network
        
        Args:
            x (torch.Tensor): Input tensor of shape 

        Returns:
            torch.Tensor: Output classfification values
        """
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