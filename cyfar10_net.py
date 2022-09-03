from time import time
from numpy import pad
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Cyfar10Net(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 10):
        super(Cyfar10Net, self).__init__()
        self.in_planes = 64
        
        self.l1 = nn.Sequential(
             nn.Conv2d(
                input_dim,
                self.in_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.l2 = self._make_layer(BasicBlock, 64, 1)
        self.l3 = self._make_layer(BasicBlock, 64, 1)
        self.l4 = self._make_layer(BasicBlock, 128, 2)
        self.l5 = self._make_layer(BasicBlock, 128, 1)        
        self.l6 = self._make_layer(BasicBlock, 256, 2)
        self.l7 = self._make_layer(BasicBlock, 256, 1)
        self.l8 = self._make_layer(BasicBlock, 512, 2)
        self.l9 = self._make_layer(BasicBlock, 512, 1)
        self.fc1 = nn.Linear(512, output_dim)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out", 
                    nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, stride):
        layer = block(self.in_planes, planes, stride)
        self.in_planes = planes * block.expansion
        return nn.Sequential(layer)

    def forward(self, x: Tensor):
        x = self.l1(x)
        x = self.pool1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.squeeze(dim=2).squeeze(dim=2)
        x = self.fc1(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if in_planes != self.expansion*planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, 
                    self.expansion*planes,
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))        
        out += self.shortcut(x)   
        out = F.relu(out)     
        return out
