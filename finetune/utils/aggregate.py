"""把 patch 特征聚合成与 GPS 对齐的帧级特征。"""

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
        raise ValueError(f"Patch数量不匹配: 期望{expected_L}，实际{L}")

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
        print(f"[INFO] {len(invalid_indices)} 个边界帧使用最近邻插值")

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
                f"特征截取长度不匹配: "
                f"期望{total}帧，实际{features_vehicle.shape[0]}帧"
            )

    return features_vehicle

if __name__ == '__main__':
    print("=" * 70)
    print("特征聚合工具测试")
    print("=" * 70)

    print("\n【测试1：Patch → 帧特征】")
    B, T, H, W = 2, 60, 128, 32
    t, h, w = T // 4, H // 4, W // 4
    L = t * h * w
    D = 512

    patch_features = torch.randn(B, L, D)
    print(f"输入Patch特征: {patch_features.shape}")

    frame_features = patch_to_frame_features(patch_features, T, H, W)
    print(f"输出帧特征: {frame_features.shape}")
    print(f"✓ 转换成功: [B, L, D] → [B, T, D]")

    print("\n【测试2：滑窗聚合】")
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

    print(f"窗口数: {len(window_features_list)}")
    print(f"每个窗口: [{window_size}, {feature_dim}]")

    frame_features = aggregate_window_features_to_frames(
        window_features_list,
        window_start_indices,
        window_size,
        total_frames,
        feature_dim
    )

    print(f"聚合后特征: {frame_features.shape}")
    print(f"✓ 聚合成功: {num_windows}个窗口 → {total_frames}帧")

    print("\n【测试3：GPS截取】")
    features_full = np.random.randn(1500, 512).astype(np.float32)

    gps_info = {
        'start_frame': 200,
        'end_frame': 400,
        'total_frames': 201,
        'start_index': 199,
        'end_index': 399,
    }

    features_vehicle = crop_features_by_gps(features_full, gps_info)

    print(f"完整特征: {features_full.shape}")
    print(f"GPS范围: 帧{gps_info['start_frame']}-{gps_info['end_frame']} (共{gps_info['total_frames']}帧)")
    print(f"截取索引: [{gps_info['start_index']}:{gps_info['start_index']+gps_info['total_frames']}]")
    print(f"车辆特征: {features_vehicle.shape}")
    print(f"✓ 截取成功: 长度验证 {features_vehicle.shape[0]} == {gps_info['total_frames']}")

    print("\n" + "=" * 70)
    print("所有测试通过！")
    print("=" * 70)
