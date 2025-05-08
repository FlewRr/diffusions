import os
from PIL import Image
import torch

class DDPMDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, transform):
        super().__init__()

        self.image_paths = [os.path.join(image_dir, fname)
                            for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem(self, idx: int):
        image_path = self.self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
