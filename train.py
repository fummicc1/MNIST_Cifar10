from time import time
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import wandb
import torch

from validate import validate

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_and_validate(net: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, epoch_size:int = 25, lr: float = 0.001, momentum: float = 0.9):
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=0.001
    )
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    param_cnt = 0
    for param in net.parameters():
        if param.requires_grad:
            param_cnt += param.numel()
    wandb.log({
        "model": net,
        "model_parameter_counts": param_cnt,
    })
    print("model_param_cnt", param_cnt)

    for epoch in range(epoch_size):
        epoch_loss = 0
        duration_list = []
        start = time()
        for i, (imgs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            out = net(imgs)
            loss = criterion(out, labels)            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[{epoch+1}, {i+1}] loss: {epoch_loss / len(train_dataloader)}")
        duration = time() - start
        print(f"epoch {epoch+1}, {i+1}, duration:", duration)
        duration_list.append(duration)
        scheduler.step()
        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss / len(train_dataloader),
            f"total duration per epoch_{epoch}": sum(duration_list),
        })
        print(f"epoch {epoch+1}, Inference")
        validate(net, train_dataloader, test_dataloader, epoch)


def train(net: nn.Module, train_dataloader: DataLoader, epoch_size:int = 25, lr: float = 0.001, momentum: float = 0.9):
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=0.0001
    )
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    param_cnt = 0
    for param in net.parameters():
        if param.requires_grad:
            param_cnt += param.numel()
    wandb.log({
        "model": net,
        "model_parameter_counts": param_cnt,
    })
    print("model_param_cnt", param_cnt)

    for epoch in range(epoch_size):
        epoch_loss = 0
        duration_list = []
        start = time()
        for i, (imgs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            imgs = imgs
            out = net(imgs)
            loss = criterion(out, labels)            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[{epoch+1}, {i+1}] loss: {epoch_loss / len(train_dataloader)}")
        duration = time() - start
        print(f"epoch {epoch+1}, {i+1}, duration:", duration)
        duration_list.append(duration)
        scheduler.step()
        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss / len(train_dataloader),
            f"total duration per epoch_{epoch}": sum(duration_list),
        })
