"""Stitching projection fusion for ablation, Not used as default path. """

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

class ConcatProjCSIFusion(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.wifo_dim = int(getattr(config, "wifo_dim", 512))
        self.n_rsu_max = int(getattr(config, "n_rsu_max", 4))
        in_dim = self.n_rsu_max * self.wifo_dim
        hidden = int(getattr(config, "concat_proj_hidden", self.wifo_dim))
        dropout = float(getattr(config, "fusion_dropout", 0.1))

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.wifo_dim),
        )
        print(
            f"[INFO] ConcatProjCSIFusion: pad N_max={self.n_rsu_max}, "
            f"{in_dim}→{hidden}→{self.wifo_dim} (No Space Quality Score) "
        )

    def forward(
        self,
        rsf_data_dict: Dict[Any, Dict[str, torch.Tensor]],
        scene_names: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:

        keys = sorted(rsf_data_dict.keys())
        if not keys:
            raise ValueError("rsf_data_dict Is Empty")

        stacked = torch.stack(
            [rsf_data_dict[k]["csi_features"] for k in keys], dim=2
        )
        bsz, seq_len, n_rsu, dim = stacked.shape
        if dim != self.wifo_dim:
            raise ValueError(f"CSI Dimensions {dim} != wifo_dim {self.wifo_dim}")
        if n_rsu > self.n_rsu_max:
            raise ValueError(
                f"RSU Number {n_rsu} > n_rsu_max={self.n_rsu_max}, Please increase the configuration n_rsu_max"
            )

        if n_rsu < self.n_rsu_max:
            pad = stacked.new_zeros(bsz, seq_len, self.n_rsu_max - n_rsu, dim)
            padded = torch.cat([stacked, pad], dim=2)
        else:
            padded = stacked

        flat = padded.reshape(bsz, seq_len, self.n_rsu_max * self.wifo_dim)
        fused = self.proj(flat)

        rsf_weights = stacked.new_full((bsz, seq_len, n_rsu), 1.0 / float(n_rsu))

        scene_name_single = None
        if isinstance(scene_names, list) and scene_names:
            scene_name_single = scene_names[0]
        elif isinstance(scene_names, str):
            scene_name_single = scene_names

        interpretable_info: Dict[str, Any] = {
            "attention_weights": rsf_weights,
            "quality_scores": None,
            "fusion_strategy": "concat_proj",
            "scene_name": scene_name_single,
            "scene_names": scene_names,
            "n_rsf": n_rsu,
            "n_rsu_max": self.n_rsu_max,
        }
        return fused, rsf_weights, interpretable_info
