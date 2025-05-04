from diffusions.schedulers.scheduler import Scheduler
import torch


class LinearScheduler(Scheduler):
    def __init__(self, timesteps: int, beta_0: float = 1e-4, beta_t: float = 0.02):
        self.timesteps = timesteps
        self.beta_0 = beta_0
        self.beta_t = beta_t

        self.betas = self.calculate_betas()
        self.alphas, self.alphas_hat = self.calculate_alphas()

        self.betas_hat = self.calculate_betas_hat()

    def calculate_betas(self):
        betas = torch.linspace(self.beta_0, self.beta_t, self.timesteps)

        return betas

    def calculate_alphas(self):
        alphas = 1. - self.betas
        alphas_hat = torch.cumprod(alphas, dim=0)

        return alphas, alphas_hat

    def calculate_betas_hat(self):
        betas_hat = self.betas[:-1] * (1. - self.alphas_hat[:-1]) / (1. - self.alphas_hat[1:])
        betas_hat = torch.cat([betas_hat, self.betas[-1:]])

        return betas_hat

    def get_betas(self):
        return self.betas, self.betas_hat

    def get_alphas(self):
        return self.alphas, self.alphas_hat