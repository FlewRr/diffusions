import os
from PIL import Image
import torch
import torch.nn as nn
import torch.functional as F
from torchvision.datasets import CIFAR10


class CIFAR10ImagesOnly(CIFAR10):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image


class DDPMDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, transform):
        super().__init__()

        self.image_paths = [os.path.join(image_dir, fname)
                            for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem(self, idx):
        image_path = self.self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
