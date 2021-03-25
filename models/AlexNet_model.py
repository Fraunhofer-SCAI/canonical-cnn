# =====================================================================
# Implemented AlexNet like model
# =====================================================================

import torch
from torch import nn

class AlexNet(nn.Module):
    """
    AlexNet like achitecture & its forward pass
    
    Methods:
        forward: A forward pass for the network

    Args:

    """
    def __init__(self, num_classes: int = 10):
      super(AlexNet, self).__init__()
      # Convolutional layers
      self.features = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),
          nn.Conv2d(64, 192, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),
          nn.Conv2d(192, 384, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(384, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),
      )
      # Fully connected layer for classification
      self.classifier = nn.Sequential(
          nn.Dropout(),
          nn.Linear(256 * 4 * 4, 1024),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(1024, 512),
          nn.ReLU(inplace=True),
          nn.Linear(512, num_classes),
      )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass connections for the network

        Args:
            x (torch.Tensor): Input tensor of shape 

        Returns:
            torch.Tensor: Output classfification values
        """
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
