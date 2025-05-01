from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.config import CosineSchedulerConfig, LinearSchedulerConfig
from ddpm.unet import UnetWithAttention
from schedulers.cosine_scheduler import CosineScheduler
from schedulers.linear_scheduler import LinearScheduler
from schedulers.scheduler import Scheduler

class DDPM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = UnetWithAttention(in_channels=config.input_channels, hid_channels=config.hidden_channels,
                                       out_channels=config.output_channels, time_emb_dim=config.time_embedding_dim).to(self.config.device)

        self.scheduler = self._setup_scheduler()
        self.alphas, self.alphas_hat = self.scheduler.get_alphas()
        self.betas, self.betas_hat = self.scheduler.get_betas()

        self.T = self.config.scheduler.timesteps
        self.image_size = self.config.image_size
        self.image_channels = self.config.image_channels
        self.device = self.config.device

    def _setup_scheduler(self):
        match self.config.scheduler:
            case LinearSchedulerConfig():
                return LinearScheduler(self.config.scheduler.timesteps, self.config.scheduler.beta_min, self.config.scheduler.beta_max)
            case CosineSchedulerConfig():
                return CosineScheduler(self.config.scheduler.timesteps, self.config.scheduler.s)

    def forward(self, x, t):
        return self.model(x, t)

    def get_model(self):
        return self.model

    def save_checkpoint(self, checkpoint_path:str):
        torch.save(self.model.state_dict(), checkpoint_path)

    def sample_xt(self, x0: torch.Tensor, t: torch.Tensor, alphas_hat: torch.Tensor, noise: torch.Tensor=None):
        if noise is None:
            noise = torch.randn_like(x0)

        t = t.cpu()
        sqrt_alpha_hat = alphas_hat[t].sqrt().view(-1, 1, 1, 1).to(self.device)
        minus_sqrt_alpha_hat = (1. - alphas_hat[t]).sqrt().view(-1, 1, 1, 1).to(self.device)

        return x0 * sqrt_alpha_hat + minus_sqrt_alpha_hat * noise

    def training_step(self, batch) -> float:
        batch = batch.to(self.device)

        t = torch.randint(0, self.T, (batch.shape[0],), device=self.device)
        noise = torch.randn_like(batch)

        noisy_batch = self.sample_xt(x0=batch, t=t, alphas_hat=self.alphas_hat, noise=noise)
        noisy_pred = self.model(noisy_batch, t)

        loss = F.mse_loss(noisy_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, num_samples: int=1):
        x_t = torch.randn(num_samples, self.image_channels, self.image_size[0], self.image_size[1], device=self.device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

            noise_pred = self.model(x_t, t_batch) # TODO: EMA

            alpha = self.alphas[t]
            alpha_sqrt = alpha.sqrt()
            alpha_hat = self.alphas_hat[t]

            mean = (1. / alpha_sqrt) * (x_t - ((1. - alpha) / (1. - alpha_hat).sqrt()) * noise_pred)
            std = self.betas_hat[t].sqrt()
            noise = torch.randn_like(x_t) if t > 0 else 0

            x_t = mean + std * noise

        return x_t
