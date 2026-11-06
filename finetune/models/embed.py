"""Put the CSI Cut into three dimensions patch and embed. """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class TokenEmbedding(nn.Module):

    def __init__(self, c_in, d_model, t_patch_size, patch_size):
        super(TokenEmbedding, self).__init__()
        kernel_size = [t_patch_size, patch_size, patch_size]
        self.tokenConv = nn.Conv3d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=kernel_size
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):

        x = self.tokenConv(x)

        x = x.flatten(3)

        x = torch.einsum("ncts->ntsc", x)

        x = x.reshape(x.shape[0], -1, x.shape[-1])

        return x

class DataEmbedding(nn.Module):

    def __init__(self, c_in, d_model, dropout=0.1, t_patch_size=4, patch_size=4):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(
            c_in=c_in,
            d_model=d_model,
            t_patch_size=t_patch_size,
            patch_size=patch_size
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        TokenEmb = self.value_embedding(x)
        return TokenEmb

class SpatialPatchEmb(nn.Module):

    def __init__(self, c_in, d_model, patch_size):
        super(SpatialPatchEmb, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):

        B, C, H, W = x.shape
        x = self.conv(x)
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        return x
