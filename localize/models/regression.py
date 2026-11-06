"""以 GPS 为先验，预测六维修正量和逐维系数。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

try:
    from utils.kalman import apply_kalman_smoothing
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    print("[WARNING] Kalman filter module not available. Install filterpy: pip install filterpy")

try:
    from utils.adaptive_filter import VehicleAdaptiveOutputFilter
    ADAPTIVE_FILTER_AVAILABLE = True
except ImportError:
    ADAPTIVE_FILTER_AVAILABLE = False
    print("[WARNING] Vehicle adaptive filter module not available")

class PoseRegressor(nn.Module):

    def __init__(self, input_dim=512, hidden_dim=256, config=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.config = config or {}

        self.pose_6d_regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6)
        )

        self.confidence_6d_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 6),
            nn.Sigmoid()
        )

        self.max_position_correction = getattr(config, 'max_position_correction', 10.0)
        self.max_orientation_correction = getattr(config, 'max_orientation_correction', np.pi/4)

        self._kalman_first_call = True

        self.adaptive_filter = None
        if ADAPTIVE_FILTER_AVAILABLE and getattr(config, 'use_output_adaptive_filter', False):
            self.adaptive_filter = VehicleAdaptiveOutputFilter(config)
            print("[PoseRegressor] 输出端滤波已打开")

    def forward(self, temporal_features, gps_poses_6d, vehicle_names=None):

        B, T, _ = temporal_features.shape

        raw_correction_deltas = self.pose_6d_regressor(temporal_features)

        correction_deltas = self._apply_correction_limits(raw_correction_deltas)

        confidences = self.confidence_6d_head(temporal_features)

        corrected_poses_6d = gps_poses_6d + correction_deltas * confidences

        corrected_positions = corrected_poses_6d[:, :, :3]
        corrected_orientations = corrected_poses_6d[:, :, 3:]

        vehicle_name = getattr(self.config, 'current_vehicle', None)
        corrected_positions = self._force_car_z_coordinate(
            corrected_positions=corrected_positions,
            gps_positions=gps_poses_6d[:, :, :3],
            vehicle_name=vehicle_name
        )

        corrected_orientations = self._normalize_orientations(corrected_orientations)

        corrected_poses_6d = torch.cat([corrected_positions, corrected_orientations], dim=-1)

        final_poses_6d = corrected_poses_6d

        if self.adaptive_filter is not None and vehicle_names is not None:
            try:
                final_poses_6d = self.adaptive_filter.apply(corrected_poses_6d, vehicle_names)
            except Exception as e:
                print(f"[WARNING] 车辆自适应滤波失败: {e}")
                final_poses_6d = corrected_poses_6d

        final_positions = final_poses_6d[:, :, :3]
        final_orientations = final_poses_6d[:, :, 3:]

        result = {
            'corrected_poses_6d': final_poses_6d,
            'correction_deltas': correction_deltas,
            'confidences': confidences,
            'pose_dict': {
                'positions': final_positions,
                'orientations': final_orientations,
                'gps_positions': gps_poses_6d[:, :, :3],
                'gps_orientations': gps_poses_6d[:, :, 3:]
            }
        }

        return result

    def _apply_correction_limits(self, raw_corrections):

        position_corrections = torch.tanh(raw_corrections[:, :, :3]) * self.max_position_correction

        orientation_corrections = torch.tanh(raw_corrections[:, :, 3:]) * self.max_orientation_correction

        return torch.cat([position_corrections, orientation_corrections], dim=-1)

    def _normalize_orientations(self, orientations):

        roll_pitch = torch.clamp(orientations[:, :, :2], -np.pi/2, np.pi/2)

        yaw = torch.fmod(orientations[:, :, 2:3] + np.pi, 2*np.pi) - np.pi

        return torch.cat([roll_pitch, yaw], dim=-1)

    def _force_car_z_coordinate(self, corrected_positions, gps_positions, vehicle_name=None):

        if vehicle_name is None or not vehicle_name.startswith('Car'):
            return corrected_positions

        positions = torch.cat([
            corrected_positions[:, :, :2],
            gps_positions[:, :, 2:3]
        ], dim=-1)

        return positions

class PoseHead(nn.Module):

    def __init__(self, input_dim=512, hidden_dim=256, config=None):
        super().__init__()
        self.config = config or {}

        self.pose_regressor = PoseRegressor(input_dim, hidden_dim, config)

        if getattr(config, 'multi_rsf_post_fusion', False):
            self.post_fusion_layer = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim)
            )
        else:
            self.post_fusion_layer = None

    def forward(self, fused_temporal_features, gps_poses_6d, scene_name=None, vehicle_names=None):

        if self.post_fusion_layer is not None:
            enhanced_features = self.post_fusion_layer(fused_temporal_features)
            enhanced_features = enhanced_features + fused_temporal_features
        else:
            enhanced_features = fused_temporal_features

        results = self.pose_regressor(
            enhanced_features,
            gps_poses_6d,
            vehicle_names=vehicle_names
        )

        results['scene_name'] = scene_name
        results['csi_fusion_strategy'] = getattr(self.config, 'csi_fusion_strategy', 'dynamic')

        return results

def create_regression(config):

    if getattr(config, 'multi_rsf_mode', True):
        return PoseHead(
            input_dim=getattr(config, 'temporal_hidden_dim', 512),
            hidden_dim=getattr(config, 'position_hidden_dim', 256),
            config=config
        )
    else:
        return PoseRegressor(
            input_dim=getattr(config, 'temporal_hidden_dim', 512),
            hidden_dim=getattr(config, 'position_hidden_dim', 256),
            config=config
        )

class PoseEvaluator:

    @staticmethod
    def compute_pose_errors(pred_poses_6d, gt_poses_6d):

        position_errors = torch.norm(
            pred_poses_6d[:, :, :3] - gt_poses_6d[:, :, :3],
            dim=-1
        )

        orientation_errors = torch.abs(
            pred_poses_6d[:, :, 3:] - gt_poses_6d[:, :, 3:]
        )

        yaw_errors = orientation_errors[:, :, 2]
        yaw_errors = torch.min(yaw_errors, 2*np.pi - yaw_errors)
        orientation_errors[:, :, 2] = yaw_errors

        return {
            'position_mae': position_errors.mean().item(),
            'position_rmse': torch.sqrt(torch.mean(position_errors**2)).item(),
            'position_max': position_errors.max().item(),

            'roll_mae': orientation_errors[:, :, 0].mean().item(),
            'pitch_mae': orientation_errors[:, :, 1].mean().item(),
            'yaw_mae': orientation_errors[:, :, 2].mean().item(),

            'orientation_mae': orientation_errors.mean().item(),
            'orientation_rmse': torch.sqrt(torch.mean(orientation_errors**2)).item()
        }

    @staticmethod
    def compute_gps_improvement(pred_poses_6d, gt_poses_6d, gps_poses_6d):

        gps_position_errors = torch.norm(
            gps_poses_6d[:, :, :3] - gt_poses_6d[:, :, :3],
            dim=-1
        )

        pred_position_errors = torch.norm(
            pred_poses_6d[:, :, :3] - gt_poses_6d[:, :, :3],
            dim=-1
        )

        improvement = gps_position_errors - pred_position_errors
        improvement_ratio = improvement / (gps_position_errors + 1e-8)

        return {
            'gps_mae': gps_position_errors.mean().item(),
            'pred_mae': pred_position_errors.mean().item(),
            'improvement_mae': improvement.mean().item(),
            'improvement_ratio': improvement_ratio.mean().item(),
            'improved_frames_ratio': (improvement > 0).float().mean().item()
        }
