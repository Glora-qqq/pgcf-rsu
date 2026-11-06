"""Put the patch Feature aggregation vs. GPS Aligned Frame-Level Features. """

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

def patch_to_frame_features(
    patch_features: torch.Tensor,
    T: int,
    H: int,
    W: int,
    t_patch_size: int = 4,
    patch_size: int = 4
) -> torch.Tensor:

    B, L, D = patch_features.shape

    t = T // t_patch_size
    h = H // patch_size
    w = W // patch_size

    expected_L = t * h * w
    if L != expected_L:
        raise ValueError(f"PatchQuantity mismatch: Expectations{expected_L}, Actual{L}")

    patch_features = patch_features.reshape(B, t, h * w, D)

    frame_features = patch_features.mean(dim=2)

    if t != T:
        frame_features = F.interpolate(
            frame_features.permute(0, 2, 1),
            size=T,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

    return frame_features

def aggregate_window_features_to_frames(
    window_features_list: List[np.ndarray],
    window_start_indices: List[int],
    window_size: int,
    total_frames: int,
    feature_dim: int
) -> np.ndarray:

    features_sum = np.zeros((total_frames, feature_dim), dtype=np.float32)
    features_count = np.zeros(total_frames, dtype=np.int32)

    for window_feat, start_idx in zip(window_features_list, window_start_indices):
        for i in range(window_size):
            frame_idx = start_idx + i
            if frame_idx < total_frames:
                features_sum[frame_idx] += window_feat[i]
                features_count[frame_idx] += 1

    frame_features = np.zeros((total_frames, feature_dim), dtype=np.float32)
    valid_mask = features_count > 0

    frame_features[valid_mask] = features_sum[valid_mask] / features_count[valid_mask, None]

    if not valid_mask.all():
        invalid_indices = np.where(~valid_mask)[0]
        print(f"[INFO] {len(invalid_indices)} boundary frames using nearest neighbor interpolation")

        for idx in invalid_indices:
            if idx > 0:
                prev_valid = idx - 1
                while prev_valid >= 0 and not valid_mask[prev_valid]:
                    prev_valid -= 1
                if prev_valid >= 0:
                    frame_features[idx] = frame_features[prev_valid]
                    continue

            if idx < total_frames - 1:
                next_valid = idx + 1
                while next_valid < total_frames and not valid_mask[next_valid]:
                    next_valid += 1
                if next_valid < total_frames:
                    frame_features[idx] = frame_features[next_valid]

    return frame_features

def crop_features_by_gps(
    features_full: np.ndarray,
    gps_info: Dict,
    verify: bool = True
) -> np.ndarray:

    start_idx = gps_info['start_index']
    total = gps_info['total_frames']

    features_vehicle = features_full[start_idx:start_idx+total]

    if verify:
        if features_vehicle.shape[0] != total:
            raise ValueError(
                f"Feature intercept length mismatch: "
                f"Expectations{total}Frame, Actual{features_vehicle.shape[0]}Frame"
            )

    return features_vehicle
