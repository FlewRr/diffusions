from typing import List

import torch
import torch.nn as nn
from diffusions.unet.utils import ConvBlock, ConvBlockWithNorm, SelfAttention2D

class DecoderWithAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hid_channels: int,
            time_emb_dim: int):
        super(DecoderWithAttention, self).__init__()

        self.up_conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=hid_channels, kernel_size=2, stride=2,
                                           padding=0)
        self.up_conv1_block = ConvBlock(in_channels=in_channels, hid_channels=hid_channels, out_channels=hid_channels,
                                        time_emb_dim=time_emb_dim)

        self.attention1 = SelfAttention2D(hid_channels, num_heads=4)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=hid_channels, out_channels=hid_channels // 2, kernel_size=2,
                                           stride=2, padding=0)
        self.up_conv2_block = ConvBlock(in_channels=hid_channels, hid_channels=hid_channels // 2,
                                        out_channels=hid_channels // 2, time_emb_dim=time_emb_dim)

        self.attention2 = SelfAttention2D(hid_channels // 2, num_heads=4)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=hid_channels // 2, out_channels=hid_channels // 4, kernel_size=2,
                                           stride=2, padding=0)
        self.up_conv3_block = ConvBlock(in_channels=hid_channels // 2, hid_channels=hid_channels // 4,
                                        out_channels=hid_channels // 4, time_emb_dim=time_emb_dim)

        self.up_conv4 = nn.ConvTranspose2d(in_channels=hid_channels // 4, out_channels=hid_channels // 8, kernel_size=2,
                                           stride=2, padding=0)
        self.up_conv4_block = ConvBlock(in_channels=hid_channels // 4, hid_channels=hid_channels // 8,
                                        out_channels=hid_channels // 4, time_emb_dim=time_emb_dim)

    def forward(self, x: torch.Tensor, encoder_outputs: List[torch.Tensor], t_emb: torch.Tensor):
        x1, x2, x3, x4 = encoder_outputs

        x = self.up_conv1(x)  # 4x4
        x = torch.cat([x4, x], dim=1)
        x = self.up_conv1_block(x, t_emb)

        x = self.attention1(x)

        x = self.up_conv2(x)  # 8x8
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv2_block(x, t_emb)

        x = self.attention2(x)

        x = self.up_conv3(x)  # 16x16
        x = torch.cat([x2, x], dim=1)
        x = self.up_conv3_block(x, t_emb)

        x = self.up_conv4(x)  # 32x32
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv4_block(x, t_emb)

        return x


class DeeperDecoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hid_channels: int,
            time_emb_dim: int):
        super(DeeperDecoder, self).__init__()

        self.up_conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=hid_channels, kernel_size=2, stride=2,
                                           padding=0)
        self.up_conv1_block = ConvBlockWithNorm(in_channels=in_channels, hid_channels=hid_channels, out_channels=hid_channels,
                                        time_emb_dim=time_emb_dim)


        self.up_conv2 = nn.ConvTranspose2d(in_channels=hid_channels, out_channels=hid_channels // 2, kernel_size=2,
                                           stride=2, padding=0)
        self.up_conv2_block = ConvBlockWithNorm(in_channels=hid_channels, hid_channels=hid_channels // 2,
                                        out_channels=hid_channels // 2, time_emb_dim=time_emb_dim)

        self.attention1 = SelfAttention2D(hid_channels // 2, num_heads=4)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=hid_channels // 2, out_channels=hid_channels // 4, kernel_size=2,
                                           stride=2, padding=0)
        self.up_conv3_block = ConvBlockWithNorm(in_channels=hid_channels // 2, hid_channels=hid_channels // 4,
                                        out_channels=hid_channels // 4, time_emb_dim=time_emb_dim)

        self.attention2 = SelfAttention2D(hid_channels // 4, num_heads=4)

        self.up_conv4 = nn.ConvTranspose2d(in_channels=hid_channels // 4, out_channels=hid_channels // 8, kernel_size=2,
                                           stride=2, padding=0)
        self.up_conv4_block = ConvBlockWithNorm(in_channels=hid_channels // 4, hid_channels=hid_channels // 8,
                                        out_channels=hid_channels // 8, time_emb_dim=time_emb_dim)

        self.attention3 = SelfAttention2D(hid_channels // 8, num_heads=4)

        self.up_conv5 = nn.ConvTranspose2d(in_channels=hid_channels // 8, out_channels=hid_channels // 16, kernel_size=2,
                                           stride=2, padding=0)
        self.up_conv5_block = ConvBlockWithNorm(in_channels=hid_channels // 8, hid_channels=hid_channels // 16,
                                        out_channels=hid_channels // 8, time_emb_dim=time_emb_dim)

    def forward(self, x: torch.Tensor, encoder_outputs: List[torch.Tensor], t_emb: torch.Tensor):
        x1, x2, x3, x4, x5 = encoder_outputs

        x = self.up_conv1(x)
        x = torch.cat([x5, x], dim=1)
        x = self.up_conv1_block(x, t_emb) # 4x4

                            
        x = self.up_conv2(x)  
        x = torch.cat([x4, x], dim=1)
        x = self.up_conv2_block(x, t_emb)

        x = self.attention1(x) # 8x8

        x = self.up_conv3(x)  
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv3_block(x, t_emb)
        
        x = self.attention2(x) # 16x16

        x = self.up_conv4(x)  
        x = torch.cat([x2, x], dim=1)
        x = self.up_conv4_block(x, t_emb)

        x = self.attention3(x) # 32x32

        x = self.up_conv5(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv5_block(x, t_emb) # 64x64

        return x