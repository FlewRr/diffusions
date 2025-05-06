from abc import abstractmethod, ABC
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
import torchvision.utils as vutils
import imageio
import numpy as np

class BaseDiffusionModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self._setup_model()
        self.scheduler = self._setup_scheduler()

        self.timesteps = self.config.scheduler.timesteps
        self.image_size = self.config.image_size
        self.image_channels = self.config.image_channels
        self.device = self.config.device
        self.use_amp = self.config.use_amp

        alphas, alphas_hat = self.scheduler.get_alphas()
        betas, betas_hat = self.scheduler.get_betas()

        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_hat", alphas_hat)
        self.register_buffer("betas", betas)
        self.register_buffer("betas_hat", betas_hat)

    @abstractmethod
    def _setup_model(self):
        pass

    @abstractmethod
    def _setup_scheduler(self):
        pass

    def forward(self, x, t):
        return self.model(x, t)

    def get_model(self):
        return self.model

    def save_checkpoint(self, checkpoint_path: str):
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path: str):
        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=torch.device(self.device)))

    def _sample_for_gif(self, num_samples: int, step: int = 25):
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
                x_overtime.append(x_t)

        return x_overtime

    def sample_images(self, num_samples:int):
        self.model.eval()
        sampled_images = self.sample(num_samples)

        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) * 0.5  # Rescale from [-1, 1] to [0, 1]
        sampled_images = torch.clamp(sampled_images, 0, 1)

        images = []
        for i in range(sampled_images.size(0)):
            img = sampled_images[i]
            images.append(to_pil_image(img))

        return images

    def sample_images_for_gif(self, num_samples: int, gif_path: str, duration: float = 0.2):
        self.model.eval()
        sampled_images = self._sample_for_gif(num_samples)

        frames = []
        for sampled_image in sampled_images:
            print(1)
            grid = vutils.make_grid(sampled_image, nrow=num_samples, normalize=True, scale_each=True)
            np_img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames.append(np_img)

        gif_path += "gif.gif"
        print(len(frames), gif_path)
        imageio.mimsave(gif_path, frames, duration=duration)

    @abstractmethod
    def sample_xt(self, x0, t, noise):
        pass

    @abstractmethod
    def training_step(self, batch):
        pass

    @abstractmethod
    def sample(self, num_samples: int):
        pass