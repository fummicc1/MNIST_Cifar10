import torch.nn as nn

class MnistNet(nn.Module):
	def __init__(self):
		super(MnistNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.pool = nn.MaxPool2d(2, 2)		
		self.fc1 = nn.Sequential(
			nn.BatchNorm1d(12 * 12 * 64),
			nn.Linear(12 * 12 * 64, 128)
		)
		self.fc2 = nn.Sequential(
			nn.BatchNorm1d(128),
			nn.Linear(128, 10)
		)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.pool(x)
		x = x.view(-1, 12 * 12 * 64)
		x = self.fc1(x)
		x = self.fc2(x)
		return x