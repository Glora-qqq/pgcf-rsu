"""CSI patch 3D cosine position encoding. """

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
