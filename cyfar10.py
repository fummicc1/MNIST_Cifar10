import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from WrappedDataLoader import WrappedDataLoader
from cyfar10_net import BasicBlock, Cyfar10Net
from downloader import download_cyfar10_dataset
from fix_seed import torch_fix_seed

from train import train_and_validate
from torchvision.models import resnet18
import wandb


if __name__ == "__main__":
    wandb.init(project=f"practice-cyfar10", name="custom_model")
    torch_fix_seed(32)
    bt_size = 512
    train_dataset, test_dataset = download_cyfar10_dataset("data/")
    train_dataloader = DataLoader(
        train_dataset, batch_size=bt_size, num_workers=8, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=100, num_workers=8, shuffle=False
    )
    net = Cyfar10Net()
    train_and_validate(net, train_dataloader, test_dataloader, lr=0.001, epoch_size=200)
    print("Finished training and inferring")
    torch.save(net.state_dict(), "trained_cyfar10.pt")
    wandb.save(f"cyfar10.h5")
    print("Finished validating")
