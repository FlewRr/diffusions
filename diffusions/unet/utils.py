import math
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, time_emb_dim=256):
        super(ConvBlock, self).__init__()
        self.time_mlp = nn.Linear(time_emb_dim, hid_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        # self.relu = nn.ReLU()
        self.relu = nn.SiLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.conv1(x)
        x += t[:, :, None, None]
        x = self.relu(x)
        x = self.conv2(x)
        x = self.dropout(x)

        return x


class ConvBlockWithNorm(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, time_emb_dim=256):
        super(ConvBlockWithNorm, self).__init__()
        self.time_mlp = nn.Linear(time_emb_dim, hid_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        # self.relu = nn.ReLU()
        self.relu = nn.SiLU()

        # num groups for lower channels shouldn't be calculated as channels // 4
        # basically for lower channels group norm acts like layer norm
        num_groups1 = hid_channels // 4 if hid_channels >= 32 else hid_channels
        num_groups2 = out_channels // 4 if out_channels >= 32 else out_channels

        self.gn1 = nn.GroupNorm(num_groups=num_groups1, num_channels=hid_channels)
        self.gn2 = nn.GroupNorm(num_groups=num_groups2, num_channels=out_channels)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.gn1(self.conv1(x))
        x += t[:, :, None, None]
        x = self.relu(x)
        x = self.gn2(self.conv2(x))
        x = self.relu(x)

        return x


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=1)


class SelfAttention2D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()

        self.ln = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_input = x

        # [B, H*W, C]
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.ln(x)

        attn_out, _ = self.attn(x, x, x)

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        return x_input + x