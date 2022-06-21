from copy import deepcopy
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(n_chans)        
        self.conv2 = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        torch.nn.init.constant_(self.bn.weight, 0.5)
        torch.nn.init.zeros_(self.bn.bias)
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        torch.nn.init.constant_(self.bn2.weight, 0.5)
        torch.nn.init.zeros_(self.bn2.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.conv2(out)
        out = self.bn2(out) + x
        out = torch.relu(out)
        return out