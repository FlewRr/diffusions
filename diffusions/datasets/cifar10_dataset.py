from torchvision.datasets import CIFAR10

class CIFAR10ImagesOnly(CIFAR10):
    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return image
