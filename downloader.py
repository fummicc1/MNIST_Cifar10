from torchvision import datasets, transforms

def download_minist_dataset(outpath: str, tf: transforms):
    mnist = datasets.MNIST(outpath, train=True, download=True, transform=tf)
    mnist_val = datasets.MNIST(outpath, train=False, download=True, transform=tf)
    return (mnist, mnist_val)

if __name__ == "__main__":
    to_tensor = transforms.ToTensor()
    train_dataset, val_dataset = download_minist_dataset("data/")
    head_img = train_dataset[0][0]
    # to_tensorは特徴量の値を[0, 1]に変換してくれる
    head_img = to_tensor(head_img)
    # shapeは(C, H, W), plt.imshowを使用する際にはshapeは(H, W, C)に変換（permute）する
    print(head_img.shape, head_img.dtype)
