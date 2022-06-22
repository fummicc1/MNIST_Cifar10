from genericpath import exists
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from downloader import download_minist_dataset
from mnist_net import MnistNet
from fix_seed import torch_fix_seed

from train import train
from validate import validate
import wandb

if __name__ == "__main__":
	wandb.init(project=f"practice-mnist")
	torch_fix_seed(32)
	my_transforms = transforms.Compose([
		transforms.ToTensor()	
	])
	train_dataset, test_dataset = download_minist_dataset("data/", my_transforms)
	train_dataloader = DataLoader(train_dataset, batch_size=100, num_workers=8)
	test_dataloader = DataLoader(test_dataset, batch_size=100, num_workers=8)

	net = MnistNet()
	if exists("trained_mnist_state.pt"):
		net.load_state_dict(torch.load("trained_mnist_state.pt"))
	else:
		train(net, train_dataloader, batch_size=100)
		print("Finished training")
		torch.save(net.state_dict(), "trained_mnist_state.pt")
	validate(net, train_dataloader, test_dataloader)
	wandb.save(f"mnist.h5")
	print("Finished validating")