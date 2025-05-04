from abc import abstractmethod, ABC
import torch
import torch.nn as nn

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

    @abstractmethod
    def sample_xt(self, x0, t, noise):
        pass

    @abstractmethod
    def training_step(self, batch):
        pass

    @abstractmethod
    def sample(self, num_samples: int):
        pass