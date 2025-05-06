from abc import abstractmethod, ABC
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

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
        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=torch.device(self.device)))

    def sample_images(self, num_samples:int):
        sampled_images = self.sample(num_samples)
        self.model.eval()

        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) * 0.5  # Rescale from [-1, 1] to [0, 1]
        sampled_images = torch.clamp(sampled_images, 0, 1)

        images = []
        for i in range(sampled_images.size(0)):
            img = sampled_images[i]
            images.append(to_pil_image(img))

        return images
    @abstractmethod
    def sample_xt(self, x0, t, noise):
        pass

    @abstractmethod
    def training_step(self, batch):
        pass

    @abstractmethod
    def sample(self, num_samples: int):
        pass