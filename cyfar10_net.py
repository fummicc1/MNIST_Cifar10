from time import time
from numpy import pad
import torch.nn as nn

from resnet import ResBlock

class Cyfar10Net(nn.Module):
	def __init__(self, input_dim: int = 3, output_dim:int = 10):
		super(Cyfar10Net, self).__init__()
		self.upsampler = nn.Upsample(size=(64, 64))
		self.l1 = nn.Sequential(
			nn.Conv2d(input_dim, 64, 3),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d((2, 2))
		)
		self.l2 = nn.Sequential(
			*([ResBlock(64) for i in range(2)])			
		)
		self.l3 = nn.Sequential(
			nn.Conv2d(64, 256, 5, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d((2, 2)),
		)
		self.l4 = nn.Sequential(
			*([ResBlock(256) for i in range(2)])
		)		
		self.l5 = nn.Sequential(
			nn.Conv2d(256, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),		
		)
		self.l6 = nn.Sequential(
			*([ResBlock(512) for i in range(1)])
		)
		self.l7 = nn.Sequential(
			nn.Conv2d(512, 512, 3),
			nn.BatchNorm2d(512),			
			nn.ReLU(inplace=True),		
		)
		self.l8 = nn.Sequential(
			nn.AvgPool2d(2, 2)
		)
		self.fc1 = nn.Sequential(			
			nn.Linear(6 * 6 * 512, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
		)
		self.fc2 = nn.Sequential(
			nn.Linear(1024, output_dim)
		)
    
	def forward(self, x):		
		x = self.upsampler(x)		
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		x = self.l5(x)
		x = self.l6(x)
		x = self.l7(x)
		x = self.l8(x)
		x = x.view(-1, 6 * 6 * 512)
		x = self.fc1(x)
		x = self.fc2(x)
		return x