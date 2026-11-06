"""按相机朝向做交叉注意力，融合左、中、右三个视角。"""

import torch
import torch.nn as nn
import math

class SpatialAwareCrossViewFusion(nn.Module):

    def __init__(self, d_model=1024, num_heads=8, num_layers=2, dropout=0.1):

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.geometry = SpatialViewEncoder(d_model)

        self.cross_view_layers = nn.ModuleList([
            CrossViewAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.fusion_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, view_features, view_angles, return_attention=False):

        view_names = ['left', 'middle', 'right']
        encoded_views = []

        for i, (name, feat) in enumerate(zip(view_names, view_features)):
            angle = view_angles[name]
            encoded = self.geometry(feat, angle)
            encoded_views.append(encoded)

        attn_maps = []
        for layer in self.cross_view_layers:
            encoded_views, attn = layer(encoded_views)
            if return_attention:
                attn_maps.append(attn)

        concat_views = torch.cat(encoded_views, dim=-1)
        fused = self.fusion_head(concat_views)
        fused = self.norm(fused)

        if return_attention:
            return fused, attn_maps
        else:
            return fused

class SpatialViewEncoder(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.angle_proj = nn.Linear(64, d_model)

    def forward(self, features, angle):

        B, T, D = features.shape

        freqs = torch.arange(32, device=features.device).float()
        angle_tensor = torch.tensor([angle], device=features.device)

        sin_enc = torch.sin(freqs * angle_tensor)
        cos_enc = torch.cos(freqs * angle_tensor)
        angle_encoding = torch.cat([sin_enc, cos_enc], dim=0)

        angle_feat = self.angle_proj(angle_encoding)

        angle_feat = angle_feat.unsqueeze(0).unsqueeze(0)
        encoded = features + angle_feat

        return encoded

class CrossViewAttentionLayer(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.left_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.middle_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.right_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(3)
        ])

        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, views):

        left, middle, right = views
        B, T, D = left.shape

        kv_for_left = torch.cat([middle, right], dim=1)
        left_out, left_attn = self.left_attn(left, kv_for_left, kv_for_left)
        left = self.norm1[0](left + self.dropout(left_out))
        left = self.norm2[0](left + self.ffn[0](left))

        kv_for_middle = torch.cat([left, right], dim=1)
        middle_out, middle_attn = self.middle_attn(middle, kv_for_middle, kv_for_middle)
        middle = self.norm1[1](middle + self.dropout(middle_out))
        middle = self.norm2[1](middle + self.ffn[1](middle))

        kv_for_right = torch.cat([left, middle], dim=1)
        right_out, right_attn = self.right_attn(right, kv_for_right, kv_for_right)
        right = self.norm1[2](right + self.dropout(right_out))
        right = self.norm2[2](right + self.ffn[2](right))

        updated_views = [left, middle, right]

        attn_weights = {
            'left': left_attn,
            'middle': middle_attn,
            'right': right_attn
        }

        return updated_views, attn_weights

def create_fusion_model(config: dict) -> SpatialAwareCrossViewFusion:

    model = SpatialAwareCrossViewFusion(
        d_model=config.get('d_model', 1024),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.1)
    )

    return model
