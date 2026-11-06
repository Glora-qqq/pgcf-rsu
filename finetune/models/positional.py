"""CSI patch 的三维正余弦位置编码。"""

import torch
import torch.nn as nn
import numpy as np

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid_with_resolution(embed_dim, pos, res):

    pos = pos * res
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size1, grid_size2, cls_token=False):

    grid_h = np.arange(grid_size1, dtype=np.float32)
    grid_w = np.arange(grid_size2, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size1, grid_size2])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):

    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size_t, grid_size_h, grid_size_w, scale=[1, 1, 1]):

    dim_t = embed_dim // 3
    dim_h = embed_dim // 3
    dim_w = embed_dim - dim_t - dim_h

    t = np.arange(grid_size_t, dtype=np.float32)
    h = np.arange(grid_size_h, dtype=np.float32)
    w = np.arange(grid_size_w, dtype=np.float32)

    tt, hh, ww = np.meshgrid(t, h, w, indexing='ij')

    tt_flat = tt.flatten()
    hh_flat = hh.flatten()
    ww_flat = ww.flatten()

    emb_t = get_1d_sincos_pos_embed_from_grid_with_resolution(dim_t, tt_flat, scale[0])
    emb_h = get_1d_sincos_pos_embed_from_grid_with_resolution(dim_h, hh_flat, scale[1])
    emb_w = get_1d_sincos_pos_embed_from_grid_with_resolution(dim_w, ww_flat, scale[2])

    pos_embed = np.concatenate([emb_t, emb_h, emb_w], axis=1)

    return pos_embed

def get_3d_sincos_pos_embed_from_indices(embed_dim, indices, input_size, scale=[1, 1, 1]):

    T, H, W = input_size

    pos_embed_full = get_3d_sincos_pos_embed(embed_dim, T, H, W, scale)
    pos_embed_full = torch.from_numpy(pos_embed_full).float()

    B, L_keep = indices.shape
    pos_embed_full = pos_embed_full.unsqueeze(0).expand(B, -1, -1)

    pos_embed_selected = torch.gather(
        pos_embed_full,
        dim=1,
        index=indices.unsqueeze(-1).expand(-1, -1, embed_dim)
    )

    return pos_embed_selected

if __name__ == '__main__':
    print("Testing positional.py...")

    print("\n1. Testing 1D SinCos position encoding...")
    pos = np.arange(10)
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim=512, pos=pos)
    print(f"   Input positions: {pos.shape}")
    print(f"   Output encoding: {emb.shape}")
    assert emb.shape == (10, 512)

    print("\n2. Testing 2D SinCos position encoding...")
    emb_2d = get_2d_sincos_pos_embed(embed_dim=512, grid_size1=32, grid_size2=8)
    print(f"   Grid size: (32, 8)")
    print(f"   Output encoding: {emb_2d.shape}")
    assert emb_2d.shape == (256, 512)

    print("\n3. Testing 3D SinCos position encoding (60-frame window)...")
    T, H, W = 15, 32, 8
    emb_3d = get_3d_sincos_pos_embed(
        embed_dim=512,
        grid_size_t=T,
        grid_size_h=H,
        grid_size_w=W,
        scale=[1, 1, 1]
    )
    expected_L = T * H * W
    print(f"   Grid size: T={T}, H={H}, W={W}")
    print(f"   Output encoding: {emb_3d.shape}")
    print(f"   Expected L: {expected_L}")
    assert emb_3d.shape == (expected_L, 512)

    print("\n4. Testing different window sizes...")
    for window_size in [30, 60, 90, 120]:
        T_patches = window_size // 4
        emb = get_3d_sincos_pos_embed(512, T_patches, 32, 8, scale=[1, 1, 1])
        L = T_patches * 32 * 8
        print(f"   Window={window_size} (T={T_patches}): L={L}, emb.shape={emb.shape}")
        assert emb.shape == (L, 512)

    print("\n5. Testing position encoding with indices (masking)...")
    T, H, W = 15, 32, 8
    L = T * H * W
    B = 2
    L_keep = int(L * 0.2)

    indices = torch.randint(0, L, (B, L_keep))
    pos_embed = get_3d_sincos_pos_embed_from_indices(
        embed_dim=512,
        indices=indices,
        input_size=(T, H, W),
        scale=[1, 1, 1]
    )
    print(f"   Batch size: {B}, L_keep: {L_keep}")
    print(f"   Indices: {indices.shape}")
    print(f"   Output encoding: {pos_embed.shape}")
    assert pos_embed.shape == (B, L_keep, 512)

    print("\n6. Testing resolution scaling...")
    pos = np.arange(10)
    emb_no_scale = get_1d_sincos_pos_embed_from_grid_with_resolution(
        512, pos, res=1.0
    )
    emb_scale_2x = get_1d_sincos_pos_embed_from_grid_with_resolution(
        512, pos, res=2.0
    )
    print(f"   No scaling: {emb_no_scale.shape}")
    print(f"   2x scaling: {emb_scale_2x.shape}")
    print(f"   Difference (should be non-zero): {np.abs(emb_no_scale - emb_scale_2x).mean():.6f}")

    print("\n✅ All tests passed!")
