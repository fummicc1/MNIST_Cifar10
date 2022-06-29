from time import time
from numpy import pad
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from resnet import ResBlock


class Cyfar10Net(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 10):
        super(Cyfar10Net, self).__init__()
        self.upsampler = nn.Upsample(size=(64, 64))
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.l1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.l2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.l3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.l4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.fc1 = nn.Linear(2 * 2 * 512, output_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.upsampler(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 2 * 2 * 512)
        x = self.fc1(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
