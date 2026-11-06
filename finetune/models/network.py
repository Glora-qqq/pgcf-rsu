"""Progressive gating: Put the 1024 Dimension Visual Feature Injection WiFo Decoder. """

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from einops import rearrange

from .layers import Block, build_transformer_blocks
from .embed import DataEmbedding
from .positional import get_3d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid_with_resolution
from .masking import random_masking, random_restore, patchify_manual_mask
from .losses import MaskedMSELoss, patchify_csi

class GatedWiFo(nn.Module):

    def __init__(
        self,
        embed_dim=512,
        decoder_embed_dim=512,
        depth=6,
        decoder_depth=4,
        num_heads=8,
        decoder_num_heads=8,
        mlp_ratio=2.0,
        t_patch_size=4,
        patch_size=4,
        pos_emb='SinCos_3D',
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.t_patch_size = t_patch_size
        self.patch_size = patch_size
        self.pos_emb = pos_emb

        self.embedding = DataEmbedding(
            c_in=2,
            d_model=embed_dim,
            dropout=0.1,
            t_patch_size=t_patch_size,
            patch_size=patch_size
        )

        self.blocks = build_transformer_blocks(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
        )

        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=qkv_bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = build_transformer_blocks(
            depth=decoder_depth,
            embed_dim=decoder_embed_dim,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            t_patch_size * patch_size ** 2 * 2,
            bias=True,
        )

        self.dinov2_proj = nn.Sequential(
            nn.Linear(1024, decoder_embed_dim, bias=True),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True),
        )

        self.rgb_gate = nn.Parameter(torch.tensor(0.01))

        self.initialize_weights()

        self._init_rgb_proj_zero()

        print(f"✅ GatedWiFo model initialized!")
        print(f"   Encoder: {embed_dim}d, {depth} layers")
        print(f"   Decoder: {decoder_embed_dim}d, {decoder_depth} layers")
        print(f"   Patch size: T={t_patch_size}, H=W={patch_size}")
        print(f"   Position encoding: {pos_emb}")

    def initialize_weights(self):

        torch.nn.init.normal_(self.mask_token, std=0.02)

        w = self.embedding.value_embedding.tokenConv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_rgb_proj_zero(self):

        last_layer = self.dinov2_proj[2]
        nn.init.constant_(last_layer.weight, 0)
        if last_layer.bias is not None:
            nn.init.constant_(last_layer.bias, 0)
        print("✅ RGBThe last layer of the projection layer has zero initialization (Progressive Convergence) ")

    def patchify(self, imgs):

        N, _, T, H, W = imgs.shape
        p = self.patch_size
        u = self.t_patch_size

        assert H % p == 0 and W % p == 0 and T % u == 0

        h = H // p
        w = W // p
        t = T // u

        x = imgs.reshape(shape=(N, 2, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 2))

        self.patch_info = (N, T, H, W, p, u, t, h, w)

        x = x[:, :, :u * p**2] + 1j * x[:, :, u * p**2:]

        return x

    def unpatchify(self, patches):

        N, T, H, W, p, u, t, h, w = self.patch_info

        patches = torch.cat([patches.real, patches.imag], dim=-1)

        imgs = rearrange(
            patches,
            'b (t h w) (u p q o) -> b (t u) (h p) (w q) o',
            t=t, h=h, w=w, u=u, p=p, q=p, o=2
        )

        imgs = torch.complex(imgs[..., 0], imgs[..., 1])

        return imgs

    def pos_embed_enc_3d(self, ids_keep, batch, input_size, scale=[1, 1, 1]):

        T, H, W = input_size
        t = torch.arange(T)
        h = torch.arange(H)
        w = torch.arange(W)

        tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')

        ED = self.embed_dim
        if ED == 512:
            ED1, ED2, ED3 = 170, 170, 172
        elif ED == 256:
            ED1, ED2, ED3 = 86, 86, 84
        elif ED == 128:
            ED1, ED2, ED3 = 42, 42, 44
        elif ED == 64:
            ED1, ED2, ED3 = 22, 22, 20
        else:
            ED1 = ED // 3
            ED2 = ED // 3
            ED3 = ED - ED1 - ED2

        emb_t = get_1d_sincos_pos_embed_from_grid_with_resolution(ED1, tt.flatten(), scale[0])
        emb_h = get_1d_sincos_pos_embed_from_grid_with_resolution(ED2, hh.flatten(), scale[1])
        emb_w = get_1d_sincos_pos_embed_from_grid_with_resolution(ED3, ww.flatten(), scale[2])

        pos_embed = np.concatenate([emb_t, emb_h, emb_w], axis=1)
        pos_embed = torch.from_numpy(pos_embed).float()
        pos_embed.requires_grad = False

        pos_embed = pos_embed.to(ids_keep.device)
        pos_embed = pos_embed.unsqueeze(0).expand(batch, -1, -1)

        pos_embed_sort = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
        )

        return pos_embed_sort

    def pos_embed_dec_3d(self, batch, input_size, scale=[1, 1, 1]):

        T, H, W = input_size
        t = torch.arange(T)
        h = torch.arange(H)
        w = torch.arange(W)

        tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')

        ED = self.decoder_embed_dim
        if ED == 512:
            ED1, ED2, ED3 = 170, 170, 172
        elif ED == 256:
            ED1, ED2, ED3 = 86, 86, 84
        elif ED == 128:
            ED1, ED2, ED3 = 42, 42, 44
        elif ED == 64:
            ED1, ED2, ED3 = 22, 22, 20
        else:
            ED1 = ED // 3
            ED2 = ED // 3
            ED3 = ED - ED1 - ED2

        emb_t = get_1d_sincos_pos_embed_from_grid_with_resolution(ED1, tt.flatten(), scale[0])
        emb_h = get_1d_sincos_pos_embed_from_grid_with_resolution(ED2, hh.flatten(), scale[1])
        emb_w = get_1d_sincos_pos_embed_from_grid_with_resolution(ED3, ww.flatten(), scale[2])

        pos_embed = np.concatenate([emb_t, emb_h, emb_w], axis=1)
        decoder_pos_embed = torch.from_numpy(pos_embed).float()
        decoder_pos_embed.requires_grad = False

        decoder_pos_embed = decoder_pos_embed.to(self.mask_token.device)
        decoder_pos_embed = decoder_pos_embed.unsqueeze(0).expand(batch, -1, -1)

        return decoder_pos_embed

    def forward_encoder(self, x, mask_ratio=0.8, manual_mask=None, scale=[1, 1, 1]):

        N, _, T, H, W = x.shape
        x = self.embedding(x)
        _, L, C = x.shape

        T_patch = T // self.t_patch_size
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        input_size = (T_patch, H_patch, W_patch)

        manual_mask_patch = None
        if manual_mask is not None:
            manual_mask_patch = patchify_manual_mask(
                manual_mask,
                self.t_patch_size,
                self.patch_size,
                self.patch_size
            )

        x_masked, mask, ids_restore, ids_keep = random_masking(
            x, mask_ratio, manual_mask_patch=manual_mask_patch
        )

        if self.pos_emb == 'SinCos_3D':
            pos_embed = self.pos_embed_enc_3d(ids_keep, N, input_size, scale=scale)
            assert x_masked.shape == pos_embed.shape, f"Shape mismatch: {x_masked.shape} vs {pos_embed.shape}"
            x_attn = x_masked + pos_embed
        else:
            x_attn = x_masked

        for blk in self.blocks:
            x_attn = blk(x_attn)

        x_attn = self.norm(x_attn)

        return x_attn, mask, ids_restore, input_size

    def forward_decoder(self, latent, ids_restore, input_size, rgb_feature=None, scale=[1, 1, 1], return_features=False):

        N = latent.shape[0]
        T, H, W = input_size

        x = self.decoder_embed(latent)
        C = x.shape[-1]

        x = random_restore(x, ids_restore, N, T, H, W, C, self.mask_token)

        if self.pos_emb == 'SinCos_3D':
            decoder_pos_embed = self.pos_embed_dec_3d(N, input_size, scale=scale)
            assert x.shape == decoder_pos_embed.shape, f"Shape mismatch: {x.shape} vs {decoder_pos_embed.shape}"
            x_attn = x + decoder_pos_embed
        else:
            x_attn = x

        has_rgb = rgb_feature is not None
        if has_rgb:
            rgb_proj = self.dinov2_proj(rgb_feature)
            rgb_token = (self.rgb_gate * rgb_proj).unsqueeze(1)
            x_attn = torch.cat([rgb_token, x_attn], dim=1)

        for blk in self.decoder_blocks:
            x_attn = blk(x_attn)

        x_attn = self.decoder_norm(x_attn)

        if return_features:
            return x_attn
        else:
            if has_rgb:
                x_attn = x_attn[:, 1:, ...]
            return x_attn

    def forward(
        self,
        csi,
        rgb_global=None,
        manual_mask=None,
        mask_ratio=0.8,
        add_noise=True,
        snr_db=20,
    ):

        if add_noise:
            noise_power = torch.mean(csi ** 2) * 10 ** (-snr_db / 10)
            noise = torch.randn_like(csi) * torch.sqrt(noise_power)
            csi_n = csi + noise
        else:
            csi_n = csi

        scale = [1, 1, 1]
        latent, mask, ids_restore, input_size = self.forward_encoder(
            csi_n, mask_ratio=mask_ratio, manual_mask=manual_mask, scale=scale
        )

        pred = self.forward_decoder(
            latent, ids_restore, input_size,
            rgb_feature=rgb_global, scale=scale
        )

        pred = self.decoder_pred(pred)

        Len = self.t_patch_size * self.patch_size ** 2
        pred_complex = pred[:, :, :Len] + 1j * pred[:, :, Len:]

        target = self.patchify(csi_n)

        manual_mask_patch = None
        if manual_mask is not None:
            manual_mask_patch = patchify_manual_mask(
                manual_mask,
                self.t_patch_size,
                self.patch_size,
                self.patch_size
            )
            B, T_patch = manual_mask_patch.shape
            L = mask.shape[1]
            L_spatial = L // T_patch
            manual_mask_patch = manual_mask_patch.unsqueeze(2).expand(
                B, T_patch, L_spatial
            ).reshape(B, L)

        criterion = MaskedMSELoss()
        loss_masked, loss_visible, loss_dict = criterion(
            pred_complex, target, mask, manual_mask_patch
        )

        pred_csi = self.unpatchify(pred_complex)
        target_csi = self.unpatchify(target)

        return loss_masked, loss_visible, pred_csi, target_csi, mask, loss_dict

def create_network(model_size='base', **kwargs):

    configs = {
        'base': {
            'embed_dim': 512,
            'decoder_embed_dim': 512,
            'depth': 6,
            'decoder_depth': 4,
            'num_heads': 8,
            'decoder_num_heads': 8,
        },
        'small': {
            'embed_dim': 256,
            'decoder_embed_dim': 256,
            'depth': 6,
            'decoder_depth': 4,
            'num_heads': 8,
            'decoder_num_heads': 8,
        },
        'little': {
            'embed_dim': 128,
            'decoder_embed_dim': 128,
            'depth': 6,
            'decoder_depth': 4,
            'num_heads': 8,
            'decoder_num_heads': 8,
        },
        'tiny': {
            'embed_dim': 64,
            'decoder_embed_dim': 64,
            'depth': 6,
            'decoder_depth': 4,
            'num_heads': 8,
            'decoder_num_heads': 8,
        },
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    config = configs[model_size]
    config.update(kwargs)

    model = GatedWiFo(**config)

    print(f"✅ Created GatedWiFo-{model_size} model")

    return model
