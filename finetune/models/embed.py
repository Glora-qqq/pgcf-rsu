"""把 CSI 切成三维 patch 并嵌入。"""

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

if __name__ == '__main__':
    print("Testing embed.py...")

    print("\n1. Testing TokenEmbedding...")
    token_emb = TokenEmbedding(
        c_in=2,
        d_model=512,
        t_patch_size=4,
        patch_size=4
    )

    x = torch.randn(2, 2, 60, 128, 32)
    tokens = token_emb(x)

    T_patches = 60 // 4
    H_patches = 128 // 4
    W_patches = 32 // 4
    expected_L = T_patches * H_patches * W_patches

    print(f"   Input: {x.shape}")
    print(f"   Output: {tokens.shape}")
    print(f"   Expected L: {expected_L}, Got L: {tokens.shape[1]}")
    assert tokens.shape == (2, expected_L, 512)

    print("\n2. Testing DataEmbedding...")
    data_emb = DataEmbedding(
        c_in=2,
        d_model=512,
        dropout=0.1,
        t_patch_size=4,
        patch_size=4
    )

    x = torch.randn(2, 2, 60, 128, 32)
    emb = data_emb(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {emb.shape}")
    assert emb.shape == (2, expected_L, 512)

    print("\n3. Testing different window sizes...")
    for window_size in [30, 60, 90, 120]:
        x = torch.randn(2, 2, window_size, 128, 32)
        emb = data_emb(x)
        expected_L = (window_size // 4) * 32 * 8
        print(f"   Window={window_size}: {x.shape} -> {emb.shape} (L={expected_L})")
        assert emb.shape[1] == expected_L

    print("\n4. Testing SpatialPatchEmb...")
    spatial_emb = SpatialPatchEmb(c_in=2, d_model=512, patch_size=4)
    x = torch.randn(2, 2, 128, 32)
    out = spatial_emb(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (2, 256, 512)

    print("\n✅ All tests passed!")
