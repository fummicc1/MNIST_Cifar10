import torch
import torch.nn as nn
from torchvision import transforms

class UpSample:
    def __init__(self):
        pass

    def __call__(self, img):
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)
        upsample = nn.Upsample(scale_factor=(2, 2))                
        img = upsample(img)
        img = img.squeeze(0)
        return img
        