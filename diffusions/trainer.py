from diffusions.datasets import DDPMDataset, CIFAR10ImagesOnly
from diffusions.model import BaseDiffusionModel
import os
import torch
import torch.nn as nn
from torchvision import transforms as T
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, model: BaseDiffusionModel, config):
        self.config = config
        self.model = model
        self.transform = self._setup_transform()
        self.dataloader = self._setup_dataloader()
        self.optimizer = self._setup_optimizer()

        self.epochs = self.config.epochs
        self.device = self.config.device

        self.use_wandb = self.config.use_wandb
        self.eval_num_samples = self.config.eval_num_samples
        self.eval_sampling_epochs = self.config.eval_sampling_epochs
        self.save_checkpoints = self.config.save_checkpoints
        self.save_checkpoints_epochs = self.config.save_checkpoints_epochs
        self.save_checkpoints_path = self.config.save_checkpoints_path

        self.use_amp = self.config.use_amp

        self.scaler = torch.amp.GradScaler(self.config.device, enabled=self.use_amp)

    def _setup_dataloader(self):
        if self.config.dataset.use_cifar10:
            dataset = CIFAR10ImagesOnly(root=self.config.dataset.data_path, train=True,
                                        download=True, transform=self.transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True,
                                                     num_workers=self.config.num_workers, pin_memory=True)

            return dataloader
        else:
            dataset = DDPMDataset(self.config.dataset.data_path, self.transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True,
                                                     num_workers=self.config.num_workers, pin_memory=True)

            return dataloader

    def _setup_transform(self):
        # TODO: maybe make configurable from config
        transform = T.Compose([
            T.Resize(size=self.config.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # to [-1, 1]
        ])

        return transform

    def _setup_optimizer(self):
        optimizer = torch.optim.Adam(params=self.model.get_model().parameters(), lr=self.config.lr,
                                     betas=self.config.betas)

        return optimizer

    def _images_for_wandb(self, tensor: torch.Tensor, n: int = 1, captions=None):
        tensor = tensor.detach().cpu()
        tensor = (tensor + 1) * 0.5  # Rescale from [-1, 1] to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)

        images = []
        for i in range(min(n, tensor.size(0))):
            img = tensor[i]
            caption = captions[i] if captions and i < len(captions) else f"Image {i}"
            images.append(wandb.Image(img, caption=caption))

        return images

    def _log_wandb_images(self, sampled_images, epoch: int):
        images = self._images_for_wandb(sampled_images, n=len(sampled_images))

        wandb.log({"generated_samples": images, "epoch": epoch})

    def _log_wandb(self, total_loss: float, epoch: int):

        wandb.log({"loss": total_loss})

        if epoch % self.config.eval_sampling_epochs == 0:
            model = self.model.get_model()

            sampled_images = self.model.sample(self.eval_num_samples)

    def _save_checkpoint(self, epoch: int):
        checkpoint_path = self.save_checkpoints_path
        if self.save_checkpoints_path[-1] != "/":
            checkpoint_path += "/"

        if not os.path.exists(checkpoint_path):
            return  # TODO: add exception

        checkpoint_path += f"model_{epoch}.pt"
        self.model.save_checkpoint(checkpoint_path)

        if self.use_wandb:
            wandb.save(checkpoint_path)

    def train(self):
        self.model.train()
        for epoch in range(1, self.epochs+1):
            total_loss = 0.

            epoch_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.epochs}", leave=False)
            for batch in epoch_bar:
                loss = self.model.training_step(batch)
                self.optimizer.zero_grad()

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                epoch_bar.set_postfix(loss=loss.item())

                self.model.update_ema()

            total_loss /= len(self.dataloader)

            self.model.eval()
            if self.use_wandb:
                self._log_wandb(total_loss, epoch)

            if self.save_checkpoints and epoch % self.save_checkpoints_epochs == 0:
                self._save_checkpoint(epoch)


        return self.model