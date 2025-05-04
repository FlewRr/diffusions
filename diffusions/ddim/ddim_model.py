import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusions.config import CosineSchedulerConfig, LinearSchedulerConfig
from diffusions.model import BaseDiffusionModel
from diffusions.schedulers.cosine_scheduler import CosineScheduler
from diffusions.schedulers.linear_scheduler import LinearScheduler
from diffusions.unet import UnetWithAttention

class DDIM(BaseDiffusionModel):
    def __init__(self, config):
        super().__init__(config)

        self.eta = self.config.eta
        self.ddim_timesteps = self.config.ddim_timesteps

        ddim_t, ddim_alphas, ddim_alphas_prev, sigmas = self._get_ddim_schedule()

        self.register_buffer("ddim_t", ddim_t)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("sigmas", sigmas)

    def _setup_model(self):
        model = UnetWithAttention(in_channels=self.config.input_channels,
                                  hid_channels=self.config.hidden_channels,
                                  out_channels=self.config.output_channels,
                                  time_emb_dim=self.config.time_embedding_dim)

        return model

    def _setup_scheduler(self):
        match self.config.scheduler:
            case LinearSchedulerConfig():
                return LinearScheduler(self.config.scheduler.timesteps, self.config.scheduler.beta_min,
                                       self.config.scheduler.beta_max)
            case CosineSchedulerConfig():
                return CosineScheduler(self.config.scheduler.timesteps, self.config.scheduler.s)

    def _get_ddim_schedule(self):
        ddim_t = torch.linspace(0, self.timesteps-1, self.ddim_timesteps, device=self.device, dtype=torch.long)

        ddim_alphas = self.alphas[ddim_t]
        ddim_alphas_prev = torch.cat((ddim_alphas[:1], ddim_alphas[:-1]))

        sigmas = self.eta * torch.sqrt(
            (1. - ddim_alphas_prev) / (1. - ddim_alphas) *
            (1. - ddim_alphas / ddim_alphas_prev)
        )

        return ddim_t, ddim_alphas, ddim_alphas_prev, sigmas

    def sample_xt(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)

        t = t
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
        x_t = torch.randn(num_samples, self.images_channels, self.image_size[0], self.image_size[1], device=self.device)

        for i in reversed(range(self.ddim_timesteps)):
            t = self.ddim_t[i]

            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

            noise_pred = self.model(x_t, t_batch)

            alpha = self.ddim_alphas[t]
            alpha_prev = self.ddim_alphas_prev[t]
            sigma = self.sigmas[t]

            predicted_x0 = (x_t - (1. - alpha).sqrt() * noise_pred) / alpha.sqrt()

            direction = (1 - alpha_prev - torch.square(sigma)).sqrt() * noise_pred

            epsilon = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

            # TODO: check dims and fix if they are wrong (view(-1, 1, 1, 1))
            print(alpha_prev.sqrt().shape, predicted_x0.shape, direction.shape, sigma.shape, epsilon.shape)
            x_t = alpha_prev.sqrt() * predicted_x0 + direction + sigma * epsilon

        return x_t