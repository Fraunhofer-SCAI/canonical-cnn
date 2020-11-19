import torch
from torch import nn


class AlexNet(nn.Module):
    """
    AlexNet model as suggested in gitlab issue #6
    """
    def __init__(self, num_classes: int = 10):
      super(AlexNet, self).__init__()
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
      self.classifier = nn.Sequential(
          nn.Dropout(),
          nn.Linear(256 * 4 * 4, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, 1024),
          nn.ReLU(inplace=True),
          nn.Linear(1024, num_classes),
      )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
