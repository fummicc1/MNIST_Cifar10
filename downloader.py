from random import shuffle
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb


def download_minist_dataset(outpath: str, tf: transforms):
    mnist = datasets.MNIST(outpath, train=True, download=True, transform=tf)
    # 標準化・正規化の計算
    mnist_imgs = torch.stack([img for img, _ in mnist], dim=1)
    m = mnist_imgs.view(1, -1).mean(dim=1)
    s = mnist_imgs.view(1, -1).std(dim=1)
    norm = transforms.Normalize(
        mean=m[0],
        std=s[0]
    )
    tf = transforms.Compose([
        tf,
        norm,
    ])
    mnist = datasets.MNIST(outpath, train=True, download=True, transform=tf)
    mnist_test = datasets.MNIST(
        outpath, train=False, download=True, transform=tf)
    return (mnist, mnist_test)


def download_cyfar10_dataset(outpath: str):                
    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),        
        transforms.ToTensor(),
    ])
    tf = transforms.Compose([
        tf,
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cyfar10_test = datasets.CIFAR10(
        outpath,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cyfar10 = datasets.CIFAR10(
        outpath, train=True, download=True, transform=tf)
    cyfar10_test = datasets.CIFAR10(
        outpath, train=False, download=True, transform=test_tf)    
    return (cyfar10, cyfar10_test)


if __name__ == "__main__":
    to_tensor = transforms.ToTensor()
    train_dataset, val_dataset = download_minist_dataset("data/")
    head_img = train_dataset[0][0]
    # to_tensorは特徴量の値を[0, 1]に変換してくれる
    head_img = to_tensor(head_img)
    # shapeは(C, H, W), plt.imshowを使用する際にはshapeは(H, W, C)に変換（permute）する
    print(head_img.shape, head_img.dtype)
