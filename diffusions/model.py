from abc import abstractmethod, ABC
from typing import OrderedDict

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
import torchvision.utils as vutils
import imageio
import numpy as np

from diffusions.utils.ema import EMA

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
        self.use_ema = self.config.use_ema
        self.ema_decay = self.config.ema_decay
        self.use_amp = self.config.use_amp

        alphas, alphas_hat = self.scheduler.get_alphas()
        betas, betas_hat = self.scheduler.get_betas()

        if self.use_ema:
            self.ema = EMA(self.model, self.ema_decay)

        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_hat", alphas_hat)
        self.register_buffer("betas", betas)
        self.register_buffer("betas_hat", betas_hat)

        self.to(self.device)

    def forward(self, x, t):
        return self.model(x, t)

    def get_model(self):
        return self.model

    def state_dict(self):
        return self.model.state_dict()

    def ema_state_dict(self):
        return self.ema.state_dict()

    def load_from_checkpoint(self, state: OrderedDict):
        self.model.load_state_dict(state["model"])

        if "ema" in state.keys():
            self.ema.load_state_dict(state["ema"])
        else:
            self.ema.load(self.model)

    def load_state_dict(self, state_dict_path: str):
        state = torch.load(state_dict_path, weights_only=True, map_location=torch.device(self.device))

        self.model.load_state_dict(state)
        self.ema.load(self.model)

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
        sampled_images = self._sample_for_gif(num_samples, step=10)

        frames = []
        for sampled_image in sampled_images:
            grid = vutils.make_grid(sampled_image, nrow=num_samples, normalize=True, scale_each=True)
            np_img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames.append(np_img)

        gif_path += "gif.gif"

        imageio.mimsave(gif_path, frames, duration=duration)

    def update_ema(self):
        self.ema.update(self.model)

    @abstractmethod
    def _setup_model(self):
        pass

    @abstractmethod
    def _setup_scheduler(self):
        pass

    @abstractmethod
    def sample_xt(self, x0, t, noise):
        pass

    @abstractmethod
    def training_step(self, batch):
        pass

    @abstractmethod
    def sample(self, num_samples: int):
        pass

    @abstractmethod
    def _sample_for_gif(self, num_samples: int, step: int):
        pass