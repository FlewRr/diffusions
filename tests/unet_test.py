from diffusions.unet.unet import UnetWithAttention
import torch


if __name__ == "__main__":
    unet = UnetWithAttention(in_channels=3, hid_channels=64, out_channels=3, time_emb_dim=256)

    timesteps = 100

    batch =  torch.randn(1, 3, 32, 32)
    t = torch.randint(0, timesteps, (batch.shape[0],))

    output = unet(batch, t)

    print(output.shape)