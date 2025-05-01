from schedulers.cosine_scheduler import CosineScheduler
from schedulers.linear_scheduler import LinearScheduler
from ddpm.unet import UnetWithAttention
from ddpm import DDPM
import torch


if __name__ == "__main__":
    timesteps = 100

    unet = UnetWithAttention(in_channels=3, hid_channels=64, out_channels=3, time_emb_dim=256)
    linear_scheduler = LinearScheduler(timesteps, 1e-4, 0.02)
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

    batch =  torch.randn(1, 3, 32, 32)
    t = torch.randint(0, timesteps, (batch.shape[0],))

    ddpm = DDPM(unet, linear_scheduler, optimizer, timesteps, (32, 32), 3, "cpu")

    loss = ddpm.training_step(batch)
    sample = ddpm.sample(1)

    print(loss, sample.shape)