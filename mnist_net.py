import torch.nn as nn

from resnet import ResBlock

class MnistNet(nn.Module):
	def __init__(self, input_dim: int = 1, output_dim:int = 10):
		super(MnistNet, self).__init__()		
		self.l1 = nn.Sequential(
			nn.Conv2d(input_dim, 128, 3),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(128),
		)
		self.l2 = nn.Sequential(
			nn.Conv2d(128, 256, 3),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, 2),
		)
		self.l3 = nn.Sequential(
			nn.Conv2d(256, 256, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.fc1 = nn.Sequential(			
			nn.Linear(12 * 12 * 256, 512),
			nn.BatchNorm1d(512),
			nn.Tanh(),
		)
		self.fc2 = nn.Sequential(
			nn.Linear(512, output_dim)
		)

	def forward(self, x):
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = x.view(-1, 12 * 12 * 256)
		x = self.fc1(x)
		x = self.fc2(x)
		return x