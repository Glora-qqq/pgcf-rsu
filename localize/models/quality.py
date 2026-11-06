"""由几何特征估计每个路侧单元的观测质量并加权融合。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class SpatialQualityEstimator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        spatial_dim = 12

        self.quality_estimator = nn.Sequential(
            nn.Linear(spatial_dim, config.spatial_quality_estimator_hidden),
            nn.ReLU(),
            nn.Dropout(config.quality_estimation_dropout),
            nn.Linear(config.spatial_quality_estimator_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        if config.use_interpretable_attention:
            self.interpretable_weights = self._create_interpretable_weights_car()
        else:
            self.interpretable_weights = None

    def _create_interpretable_weights_car(self):

        try:
            from config import SCENE_ADAPTIVE_SCALES
            scales = SCENE_ADAPTIVE_SCALES['Scene1']
            feature_weights = torch.tensor([
                scales['center_distance_2d'],
                scales['height_diff'],
                scales['rsf_viewing_angle'],
                scales['heading_alignment'],
                scales['longitudinal_offset'],
                scales['lateral_offset'],
                scales['approaching_trend_2d'],
                scales['same_road_confidence'],
                scales['x_offset'],
                scales['y_offset'],
                scales['relative_angle'],
                scales['elevation_angle'],
            ])
        except (ImportError, KeyError):
            feature_weights = torch.tensor([4.0, 3.0, 4.0, 3.5, 2.5, 4.0, 2.0, 4.0, 2.0, 2.0, 1.5, 3.5])

        feature_weights = torch.clamp(feature_weights, min=1.0, max=4.0)
        return nn.Parameter(feature_weights, requires_grad=True)

    def forward(self, spatial_features, scene_names=None):

        B, T, N_rsf, D = spatial_features.shape
        assert D == 12, f"v4.0-final要求统一12维特征，得到{D}维"

        flat_features = spatial_features.view(-1, 12)

        mask_dims = getattr(self.config, 'spatial_feature_mask', None)
        if mask_dims:
            flat_features = flat_features.clone()
            for dim in mask_dims:
                flat_features[:, int(dim)] = 0.0

        interpretable_info = {}
        if self.interpretable_weights is not None:
            weights = self.interpretable_weights
            weights = torch.clamp(weights, min=1.0, max=4.0)
            weighted_features = flat_features * weights.unsqueeze(0)

            interpretable_info = {
                'feature_weights': weights.detach(),
                'weighted_features': weighted_features.view(B, T, N_rsf, 12),
                'raw_features': spatial_features
            }
        else:
            weighted_features = flat_features

        quality_scores_flat = self.quality_estimator(weighted_features)
        quality_scores = quality_scores_flat.view(B, T, N_rsf, 1)

        return quality_scores, interpretable_info

class MultiRSFSpatialQualityFusion(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fusion_strategy = getattr(config, 'csi_fusion_strategy', 'dynamic').lower()
        if self.fusion_strategy not in {'dynamic', 'average'}:
            self.fusion_strategy = 'dynamic'

        self.quality_estimator = SpatialQualityEstimator(config)

    def forward(self, rsf_data_dict, scene_names=None):

        N_rsf = len(rsf_data_dict)

        if scene_names is not None:
            scene_for_validation = scene_names[0] if isinstance(scene_names, list) else scene_names
            if scene_for_validation == 'Scene1':
                assert 1 <= N_rsf <= 3, f"Scene1 RSF数应在1–3，实际:{N_rsf}"
            else:
                assert 1 <= N_rsf <= 4, f"Scene2/3 RSF数应在1–4，实际:{N_rsf}"

        first_key = list(rsf_data_dict.keys())[0]
        B, T = rsf_data_dict[first_key]['csi_features'].shape[:2]

        all_csi_features = []
        all_spatial_features = []

        for rsf_idx in sorted(rsf_data_dict.keys()):
            all_csi_features.append(rsf_data_dict[rsf_idx]['csi_features'])
            all_spatial_features.append(rsf_data_dict[rsf_idx]['spatial_features'])

        all_csi_features = torch.stack(all_csi_features, dim=2)
        all_spatial_features = torch.stack(all_spatial_features, dim=2)

        quality_scores, interpretable_info = self.quality_estimator(
            all_spatial_features, scene_names
        )

        if self.fusion_strategy == 'average':
            rsf_weights = torch.full(
                (B, T, N_rsf),
                1.0 / float(N_rsf),
                dtype=all_csi_features.dtype,
                device=all_csi_features.device
            )
        else:
            rsf_weights = F.softmax(quality_scores.squeeze(-1), dim=-1)

        fused_csi_features = torch.sum(
            all_csi_features * rsf_weights.unsqueeze(-1), dim=2
        )

        scene_name_single = scene_names[0] if isinstance(scene_names, list) and len(scene_names) > 0 else scene_names
        interpretable_info.update({
            'attention_weights': rsf_weights,
            'quality_scores': quality_scores.squeeze(-1),
            'fusion_strategy': self.fusion_strategy,
            'scene_name': scene_name_single,
            'scene_names': scene_names,
            'n_rsf': N_rsf
        })

        return fused_csi_features, rsf_weights, interpretable_info

class SpatialQualityFusion(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.multi_rsf_fusion = MultiRSFSpatialQualityFusion(config)

    def forward(self, rsf_data_dict, scene_names=None):

        return self.multi_rsf_fusion(rsf_data_dict, scene_names)

def create_quality(config):

    return SpatialQualityFusion(config)

class SpatialQualityAnalyzer:

    @staticmethod
    def analyze_rsf_contributions(interpretable_info, rsf_names=None):

        attention_weights = interpretable_info['attention_weights']
        quality_scores = interpretable_info['quality_scores']

        B, T, N_rsf = attention_weights.shape

        if rsf_names is None:
            rsf_names = [f"RSF{i+1}" for i in range(N_rsf)]

        analysis = {
            'rsf_avg_weights': {},
            'rsf_avg_quality': {},
            'temporal_variance': {}
        }

        for rsf_idx in range(N_rsf):
            rsf_name = rsf_names[rsf_idx]

            avg_weight = attention_weights[:, :, rsf_idx].mean().item()
            analysis['rsf_avg_weights'][rsf_name] = avg_weight

            avg_quality = quality_scores[:, :, rsf_idx].mean().item()
            analysis['rsf_avg_quality'][rsf_name] = avg_quality

            temporal_var = attention_weights[:, :, rsf_idx].var(dim=1).mean().item()
            analysis['temporal_variance'][rsf_name] = temporal_var

        return analysis

    @staticmethod
    def analyze_spatial_feature_importance(interpretable_info):

        if 'feature_weights' not in interpretable_info:
            return None

        feature_weights = interpretable_info['feature_weights'].cpu().numpy()

        feature_names = [
            'center_distance_2d', 'height_diff', 'rsf_viewing_angle',
            'heading_alignment', 'longitudinal_offset', 'lateral_offset',
            'approaching_trend_2d', 'same_road_confidence', 'x_offset',
            'y_offset', 'relative_angle', 'elevation_angle'
        ]

        importance_analysis = {}
        for i, (name, weight) in enumerate(zip(feature_names, feature_weights)):
            importance_analysis[name] = {
                'weight': float(weight),
                'rank': i + 1,
                'category': 'high' if weight >= 2.5 else 'medium' if weight >= 1.5 else 'low'
            }

        return importance_analysis
