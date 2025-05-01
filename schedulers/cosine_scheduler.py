from schedulers.scheduler import Scheduler
import torch


class CosineScheduler(Scheduler):
    def __init__(self, timesteps: int, s: float = 0.008):
        self.timesteps = timesteps
        self.s = s

        self.t = torch.arange(timesteps)

        self.alphas_hat, self.alphas_hat_minus_1 = self.calculate_alphas_hat()

        self.betas, self.betas_hat = self.calculate_betas()
        self.alphas = 1.0 - self.betas

    def calculate_alphas_hat(self):
        alphas_hat = torch.pow(torch.cos((self.t / self.timesteps + self.s) / (1 + self.s) * torch.pi / 2.0), 2)
        alphas_hat_minus_1 = torch.roll(alphas_hat, shifts=1, dims=0)
        alphas_hat_minus_1[0] = alphas_hat_minus_1[1]

        return alphas_hat, alphas_hat_minus_1

    def calculate_betas(self):
        betas = 1.0 - self.alphas_hat / self.alphas_hat_minus_1
        betas = torch.minimum(betas, torch.tensor(0.999))
        betas_hat = (1 - self.alphas_hat_minus_1) / (1 - self.alphas_hat) * betas
        betas_hat[torch.isnan(betas_hat)] = 0.0

        return betas, betas_hat

    def get_betas(self):
        return self.betas, self.betas_hat

    def get_alphas(self):
        return self.alphas, self.alphas_hat