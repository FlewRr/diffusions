import torch.nn as nn
from ddpm.unet.utils import ConvBlock, SelfAttention2D

class EncoderWithAttention(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, time_emb_dim=256):
        super(EncoderWithAttention, self).__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=hid_channels, hid_channels=hid_channels,
                               time_emb_dim=time_emb_dim)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = ConvBlock(in_channels=hid_channels, out_channels=hid_channels * 2, hid_channels=hid_channels * 2,
                               time_emb_dim=time_emb_dim)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.attention1 = SelfAttention2D(hid_channels * 2, num_heads=4)

        self.conv3 = ConvBlock(in_channels=hid_channels * 2, out_channels=hid_channels * 4,
                               hid_channels=hid_channels * 4, time_emb_dim=time_emb_dim)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.attention2 = SelfAttention2D(hid_channels * 4, num_heads=4)

        self.conv4 = ConvBlock(in_channels=hid_channels * 4, out_channels=hid_channels * 8,
                               hid_channels=hid_channels * 8, time_emb_dim=time_emb_dim)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, t_emb):
        x1 = self.conv1(x, t_emb)  # 32x32
        x = self.pool1(x1)

        x2 = self.conv2(x, t_emb)  # 16x16
        x = self.pool2(x2)

        x = self.attention1(x)

        x3 = self.conv3(x, t_emb)  # 8x8
        x = self.pool3(x3)

        x = self.attention2(x)

        x4 = self.conv4(x, t_emb)  # 4x4
        x = self.pool4(x4)

        return x, [x1, x2, x3, x4]