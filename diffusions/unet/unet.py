import torch.nn as nn
from diffusions.unet.utils import ConvBlock, SinusoidalTimeEmbedding
from diffusions.unet.decoder import DecoderWithAttention
from diffusions.unet.encoder import EncoderWithAttention


class UnetWithAttention(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=3, time_emb_dim=256):
        super(UnetWithAttention, self).__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.encoder = EncoderWithAttention(in_channels=in_channels, hid_channels=hid_channels,
                                            time_emb_dim=time_emb_dim)

        self.bottleneck = ConvBlock(in_channels=hid_channels * 8, hid_channels=hid_channels * 16,
                                    out_channels=hid_channels * 16, time_emb_dim=time_emb_dim)

        self.decoder = DecoderWithAttention(in_channels=hid_channels * 16, hid_channels=hid_channels * 8,
                                            time_emb_dim=time_emb_dim)

        self.final_conv = nn.Conv2d(in_channels=hid_channels * 2, out_channels=out_channels, kernel_size=3, stride=1,
                                    padding=1)

    def forward(self, x, t):
        t = self.time_emb(t)
        x, encoder_outputs = self.encoder(x, t)
        x = self.bottleneck(x, t)
        x = self.decoder(x, encoder_outputs, t)
        x = self.final_conv(x)

        return x