import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb


def validate(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, epoch: int = None):
    model.eval()
    for name, loader in [("train", train_loader), ("test", test_loader)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                out = model(imgs)
                _, prediction = torch.max(out, dim=1)
                total += labels.shape[0]
                correct += int((prediction == labels).sum())
                print("prediction", prediction)
                print("labels", labels)
        if epoch is None:
            print(f"[Accuracy] {name}: {correct/total}")
            wandb.log({
                f"accuracy_{name}": correct / total
            })
        else:
            print(f"[Accuracy for epoch {epoch}] {name}: {correct/total}")
            wandb.log({
                f"accuracy_{name}, epoch {epoch}": correct / total
            })
    param_cnt = 0
    for param in model.parameters():
        if param.requires_grad:
            param_cnt += param.numel()
    wandb.log({
        "model": model,
        "model_parameter_counts": param_cnt,
    })
