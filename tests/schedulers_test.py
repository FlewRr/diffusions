from schedulers.cosine_scheduler import CosineScheduler
from schedulers.linear_scheduler import LinearScheduler
import torch


if __name__ == "__main__":
    linear_scheduler = LinearScheduler(100, 1e-4, 0.02)

    alphas = linear_scheduler.get_alphas()
    betas = linear_scheduler.get_betas()

    cosine_scheduler = CosineScheduler(timesteps=100, s=0.008)

    cosine_alphas = cosine_scheduler.get_alphas()
    cosine_betas = cosine_scheduler.get_betas()
