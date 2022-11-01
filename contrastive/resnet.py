
from typing import List

import torch
from torch import nn


class Activation(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x*torch.sigmoid(x)


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            Activation()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            Activation()
        )

        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x):
        aux = self.block1(x)
        aux = self.block2(aux)
        return aux + self.res_conv(x)


class ResNet(nn.Module):

    def __init__(
            self,
            latent_dimensionality: int,
            base_channels: int,
            channel_mults: List[int],
    ):
        super().__init__()

        channels = [1] + [base_channels*i for i in channel_mults]
        channel_pairs = list(zip(channels[:-1], channels[1:]))

        self.down_layers = nn.ModuleList([
            nn.ModuleList([
                ResnetBlock(ch_i, ch_j),
                ResnetBlock(ch_j, ch_j),
                nn.Conv1d(ch_j, ch_j, 3, 2, 1)
            ])
            for (ch_i, ch_j) in channel_pairs
        ])
        
        self.fc = nn.Linear(channels[-1], latent_dimensionality)

    def forward(self, x):
        x = x[:, None, :]
        
        for (block1, block2, downsample) in self.down_layers:
            x = block1(x)
            x = block2(x)
            x = downsample(x)
        
        x = x.mean(dim=(2))
        x = self.fc(x)

        return x