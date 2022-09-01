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


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def preprocess(x, y):
    return x.view(-1, 3, 32, 32).to(device), y.to(device)


if __name__ == "__main__":
    wandb.init(project=f"practice-cyfar10", name="custom_model")
    torch_fix_seed(32)
    bt_size = 500
    train_dataset, test_dataset = download_cyfar10_dataset("data/")
    train_dataloader = DataLoader(
        train_dataset, batch_size=bt_size, num_workers=8, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=bt_size, num_workers=8, shuffle=False
    )
    train_dataloader = WrappedDataLoader(train_dataloader, preprocess)
    test_dataloader = WrappedDataLoader(test_dataloader, preprocess)

    net = Cyfar10Net()
    train_and_validate(net, train_dataloader, test_dataloader, lr=0.001)
    print("Finished training and inferring")
    torch.save(net.state_dict(), "trained_cyfar10.pt")
    wandb.save(f"cyfar10.h5")
    print("Finished validating")
