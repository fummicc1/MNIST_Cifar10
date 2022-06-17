import numpy as np
import torch
from net import MnistNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from downloader import download_minist_dataset
import torch.optim as optim

if __name__ == "__main__":
	to_tensor = transforms.ToTensor()
	train_dataset, test_dataset = download_minist_dataset("data/", to_tensor)
	train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=2)
	test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=2)
	classes = np.linspace(0, 9, 10, dtype=np.uint8)
	net = MnistNet()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)	

	epoch_size = 25

	for epoch in range(epoch_size):
		batch_loss = 0
		for i, (imgs, labels) in enumerate(train_dataloader):
			optimizer.zero_grad()

			out = net(imgs)
			loss = criterion(out, labels)
			loss.backward()
			optimizer.step()

			batch_loss += loss.item()
			if i % 32 == 31:
				print(f"[{epoch+1}, {i+1} loss: {batch_loss / 32}")
				batch_loss = 0
	print("Finished training")
	torch.save(net, "trained_mnist.pt")