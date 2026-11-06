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

if __name__ == '__main__':
    print("=" * 70)
    print("Feature Aggregation Tool Test")
    print("=" * 70)

    print("\n[Test1: Patch → Frame characteristics] ")
    B, T, H, W = 2, 60, 128, 32
    t, h, w = T // 4, H // 4, W // 4
    L = t * h * w
    D = 512

    patch_features = torch.randn(B, L, D)
    print(f"InputPatchFeatures: {patch_features.shape}")

    frame_features = patch_to_frame_features(patch_features, T, H, W)
    print(f"Output frame characteristics: {frame_features.shape}")
    print(f"✓ Conversion successful: [B, L, D] → [B, T, D]")

    print("\n[Test2: Sliding Window Aggregation] ")
    window_size = 60
    stride = 5
    total_frames = 1500
    feature_dim = 512

    num_windows = (total_frames - window_size) // stride + 1
    window_features_list = []
    window_start_indices = []

    for i in range(num_windows):
        start_idx = i * stride
        window_feat = np.random.randn(window_size, feature_dim).astype(np.float32)
        window_features_list.append(window_feat)
        window_start_indices.append(start_idx)

    print(f"Number of windows: {len(window_features_list)}")
    print(f"Each window: [{window_size}, {feature_dim}]")

    frame_features = aggregate_window_features_to_frames(
        window_features_list,
        window_start_indices,
        window_size,
        total_frames,
        feature_dim
    )

    print(f"Post-aggregation features: {frame_features.shape}")
    print(f"✓ Aggregation successful: {num_windows}windows → {total_frames}Frame")

    print("\n[Test3: GPSIntercept] ")
    features_full = np.random.randn(1500, 512).astype(np.float32)

    gps_info = {
        'start_frame': 200,
        'end_frame': 400,
        'total_frames': 201,
        'start_index': 199,
        'end_index': 399,
    }

    features_vehicle = crop_features_by_gps(features_full, gps_info)

    print(f"Complete features: {features_full.shape}")
    print(f"GPSScope: Frame{gps_info['start_frame']}-{gps_info['end_frame']} (of{gps_info['total_frames']}Frame)")
    print(f"Intercept index: [{gps_info['start_index']}:{gps_info['start_index']+gps_info['total_frames']}]")
    print(f"Vehicle characteristics: {features_vehicle.shape}")
    print(f"✓ Successfully intercepted: Length Verification {features_vehicle.shape[0]} == {gps_info['total_frames']}")

    print("\n" + "=" * 70)
    print("All tests passed! ")
    print("=" * 70)
