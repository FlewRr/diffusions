import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusions.config import LinearSchedulerConfig, CosineSchedulerConfig
from diffusions.model import BaseDiffusionModel
from diffusions.schedulers.cosine_scheduler import CosineScheduler
from diffusions.schedulers.linear_scheduler import LinearScheduler
from diffusions.unet import UnetWithAttention, DeeperUnet


class DDPM(BaseDiffusionModel):
    def __init__(self, config):
        super().__init__(config)

    def _setup_model(self):
        ## TODO: change to readable from config
        if self.config.image_size == (64, 64):
            model = DeeperUnet(
                in_channels=self.config.input_channels,
                hid_channels=self.config.hidden_channels,
                out_channels=self.config.output_channels,
                time_emb_dim=self.config.time_embedding_dim
            )
        else:
            model = UnetWithAttention(in_channels=self.config.input_channels,
                                    hid_channels=self.config.hidden_channels,
                                    out_channels=self.config.output_channels,
                                    time_emb_dim=self.config.time_embedding_dim)

        return model

    def _setup_scheduler(self):
        match self.config.scheduler:
            case LinearSchedulerConfig():
                return LinearScheduler(self.config.scheduler.timesteps, self.config.scheduler.beta_min, self.config.scheduler.beta_max)
            case CosineSchedulerConfig():
                return CosineScheduler(self.config.scheduler.timesteps, self.config.scheduler.s)

    def sample_xt(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor=None):
        if noise is None:
            noise = torch.randn_like(x0)


        sqrt_alpha_hat = self.alphas_hat[t].sqrt().view(-1, 1, 1, 1)
        minus_sqrt_alpha_hat = (1. - self.alphas_hat[t]).sqrt().view(-1, 1, 1, 1)

        return x0 * sqrt_alpha_hat + minus_sqrt_alpha_hat * noise

    def training_step(self, batch) -> float:
        batch = batch.to(self.device)

        t = torch.randint(0, self.timesteps, (batch.shape[0],), device=self.device)
        noise = torch.randn_like(batch)

        noisy_batch = self.sample_xt(x0=batch, t=t, noise=noise)

        with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
            noisy_pred = self.model(noisy_batch, t)
            loss = F.mse_loss(noisy_pred, noise)

        return loss

    @torch.no_grad()
    def sample(self, num_samples: int=1):
        self.ema.apply_shadow(self.model)

        x_t = torch.randn(num_samples, self.image_channels, self.image_size[0], self.image_size[1], device=self.device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

            noise_pred = self.model(x_t, t_batch)

            alpha = self.alphas[t]
            alpha_sqrt = alpha.sqrt()
            alpha_hat = self.alphas_hat[t]

            mean = (1. / alpha_sqrt) * (x_t - ((1. - alpha) / (1. - alpha_hat).sqrt()) * noise_pred)
            std = self.betas_hat[t].sqrt()
            noise = torch.randn_like(x_t) if t > 0 else 0

            x_t = mean + std * noise

        self.ema.restore(self.model)

        return x_t

    def _sample_for_gif(self, num_samples: int, step: int):
        self.ema.apply_shadow(self.model)

        x_overtime = []
        x_t = torch.randn(num_samples, self.image_channels, self.image_size[0], self.image_size[1], device=self.device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

            noise_pred = self.model(x_t, t_batch)

            alpha = self.alphas[t]
            alpha_sqrt = alpha.sqrt()
            alpha_hat = self.alphas_hat[t]

            mean = (1. / alpha_sqrt) * (x_t - ((1. - alpha) / (1. - alpha_hat).sqrt()) * noise_pred)
            std = self.betas_hat[t].sqrt()
            noise = torch.randn_like(x_t) if t > 0 else 0

            x_t = mean + std * noise

            if t % step == 0:
                x_overtime.append(x_t.cpu())

        self.ema.restore(self.model)

        return x_overtime
