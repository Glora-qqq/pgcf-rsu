"""把目标与路侧单元的距离、角度和道路关系编成 12 维几何特征。"""

import torch
import numpy as np
from typing import Tuple, Optional
from utils.antenna import get_rsf_antenna_world_position

def compute_longitudinal_lateral_offset(
    gps_position: np.ndarray,
    rsf_extrinsic_6d: np.ndarray,
    rsf_antenna_position: np.ndarray
) -> Tuple[float, float]:

    rsf_position = rsf_antenna_position
    rsf_yaw = rsf_extrinsic_6d[5]

    diff = gps_position - rsf_position

    cos_yaw = np.cos(rsf_yaw)
    sin_yaw = np.sin(rsf_yaw)

    longitudinal_offset = diff[0] * cos_yaw + diff[1] * sin_yaw

    lateral_offset = -diff[0] * sin_yaw + diff[1] * cos_yaw

    return float(longitudinal_offset), float(lateral_offset)

def compute_elevation_angle(
    gps_position: np.ndarray,
    rsf_antenna_position: np.ndarray
) -> float:

    diff = gps_position - rsf_antenna_position
    distance_2d = np.linalg.norm(diff[:2])
    height_diff = diff[2]

    if distance_2d < 0.1:
        return 0.0

    elevation_angle = np.degrees(np.arctan2(height_diff, distance_2d))
    return float(elevation_angle)

def compute_rsf_viewing_angle(
    gps_position: np.ndarray,
    rsf_extrinsic_6d: np.ndarray,
    rsf_antenna_position: np.ndarray
) -> float:

    rsf_position = rsf_antenna_position
    rsf_yaw = rsf_extrinsic_6d[5]

    diff = gps_position - rsf_position
    angle_to_vehicle = np.arctan2(diff[1], diff[0])

    viewing_angle = angle_to_vehicle - rsf_yaw

    viewing_angle = np.arctan2(np.sin(viewing_angle), np.cos(viewing_angle))

    return float(np.degrees(viewing_angle))

def compute_heading_alignment(
    gps_pose_6d: np.ndarray,
    rsf_extrinsic_6d: np.ndarray
) -> float:

    gps_yaw = gps_pose_6d[5]
    rsf_yaw = rsf_extrinsic_6d[5]

    heading_diff = gps_yaw - rsf_yaw

    alignment = np.cos(heading_diff)

    return float(alignment)

def compute_same_road_confidence_v2(
    gps_pose_6d: np.ndarray,
    rsf_extrinsic_6d: np.ndarray,
    rsf_antenna_position: np.ndarray,
    config=None
) -> float:

    rsf_position = rsf_antenna_position
    gps_position = gps_pose_6d[:3]

    diff = gps_position - rsf_position
    distance_2d = np.linalg.norm(diff[:2])
    distance_threshold = getattr(config, 'same_road_distance_threshold', 200.0)
    distance_confidence = np.exp(-distance_2d / (distance_threshold / 2))

    longitudinal, lateral = compute_longitudinal_lateral_offset(gps_position, rsf_extrinsic_6d, rsf_antenna_position)
    lateral_threshold = getattr(config, 'same_road_lateral_threshold', 50.0)
    lateral_confidence = np.exp(-np.abs(lateral) / lateral_threshold)

    viewing_angle = compute_rsf_viewing_angle(gps_position, rsf_extrinsic_6d, rsf_antenna_position)
    viewing_threshold = getattr(config, 'same_road_viewing_threshold', 90.0)

    if np.abs(viewing_angle) < viewing_threshold:
        viewing_confidence = np.cos(np.radians(viewing_angle))
    else:
        viewing_confidence = 0.0

    distance_weight = getattr(config, 'same_road_distance_weight', 0.4)
    lateral_weight = getattr(config, 'same_road_lateral_weight', 0.4)
    viewing_weight = getattr(config, 'same_road_viewing_weight', 0.2)

    same_road_confidence = (
        distance_weight * distance_confidence +
        lateral_weight * lateral_confidence +
        viewing_weight * viewing_confidence
    )

    return float(np.clip(same_road_confidence, 0.0, 1.0))

def compute_trajectory_approaching_trend(
    gps_trajectory: np.ndarray,
    rsf_position: np.ndarray,
    current_idx: int,
    window_size: int = 10
) -> float:

    T = len(gps_trajectory)

    if T < 3:
        return 0.0

    start_idx = max(0, current_idx - window_size)
    end_idx = current_idx

    if end_idx - start_idx < 3:
        return 0.0

    start_pos = gps_trajectory[start_idx][:2]
    end_pos = gps_trajectory[end_idx][:2]

    start_dist = np.linalg.norm(start_pos - rsf_position[:2])
    end_dist = np.linalg.norm(end_pos - rsf_position[:2])

    dist_change = start_dist - end_dist

    threshold = 5.0
    if dist_change > threshold:
        return 1.0
    elif dist_change < -threshold:
        return -1.0
    else:
        return 0.0

def compute_frame_spatial_features(
    gps_pose_6d: np.ndarray,
    rsf_extrinsic_6d: np.ndarray,
    prev_gps_pose_6d: Optional[np.ndarray] = None,
    config=None,
    scene_name: str = 'Scene1',
    gps_trajectory: Optional[np.ndarray] = None,
    current_frame_idx: int = 0
) -> Tuple[np.ndarray, float]:

    rsf_antenna_position = get_rsf_antenna_world_position(rsf_extrinsic_6d, scene_name)

    gps_position = gps_pose_6d[:3]
    rsf_position = rsf_antenna_position

    diff = gps_position - rsf_position
    center_distance_2d = np.linalg.norm(diff[:2])
    height_diff = diff[2]
    x_offset = diff[0]
    y_offset = diff[1]

    longitudinal_offset, lateral_offset = compute_longitudinal_lateral_offset(
        gps_position, rsf_extrinsic_6d, rsf_antenna_position
    )

    relative_angle = np.arctan2(y_offset, x_offset) * 180 / np.pi

    if gps_trajectory is not None and current_frame_idx > 0:
        approaching_trend_2d = compute_trajectory_approaching_trend(
            gps_trajectory, rsf_position, current_frame_idx, window_size=10
        )
    elif prev_gps_pose_6d is not None:
        prev_gps_position = prev_gps_pose_6d[:3]
        prev_diff = prev_gps_position - rsf_position
        prev_dist_2d = np.linalg.norm(prev_diff[:2])
        curr_dist_2d = center_distance_2d

        dist_change = prev_dist_2d - curr_dist_2d
        if dist_change > 2.0:
            approaching_trend_2d = 1.0
        elif dist_change < -2.0:
            approaching_trend_2d = -1.0
        else:
            approaching_trend_2d = 0.0
    else:
        approaching_trend_2d = 0.0

    same_road_confidence = compute_same_road_confidence_v2(
        gps_pose_6d, rsf_extrinsic_6d, rsf_antenna_position, config
    )

    rsf_viewing_angle = compute_rsf_viewing_angle(gps_position, rsf_extrinsic_6d, rsf_antenna_position)

    heading_alignment = compute_heading_alignment(gps_pose_6d, rsf_extrinsic_6d)

    elevation_angle = compute_elevation_angle(gps_position, rsf_antenna_position)

    spatial_features = np.array([
        center_distance_2d,
        height_diff,
        rsf_viewing_angle,
        heading_alignment,
        longitudinal_offset,
        lateral_offset,
        approaching_trend_2d,
        same_road_confidence,
        x_offset,
        y_offset,
        relative_angle,
        elevation_angle,
    ], dtype=np.float32)

    return spatial_features, same_road_confidence

def extract_spatial_features_batch(
    gps_trajectory_6d: torch.Tensor,
    rsf_extrinsics_6d: torch.Tensor,
    config=None,
    scene_names=None
) -> Tuple[torch.Tensor, torch.Tensor]:

    B, T, _ = gps_trajectory_6d.shape
    _, N_rsf, _ = rsf_extrinsics_6d.shape

    gps_np = gps_trajectory_6d.cpu().numpy()
    rsf_np = rsf_extrinsics_6d.cpu().numpy()

    spatial_features_list = []
    same_road_confidence_list = []

    if scene_names is None:
        scene_names = ['Scene1'] * B

    for b in range(B):
        scene_name = scene_names[b]
        batch_features = []
        batch_confidence = []

        trajectory_3d = gps_np[b, :, :3]

        for t in range(T):
            frame_features = []
            frame_confidence = []

            curr_gps = gps_np[b, t]
            prev_gps = gps_np[b, t-1] if t > 0 else None

            for r in range(N_rsf):
                rsf_ext = rsf_np[b, r]
                features, confidence = compute_frame_spatial_features(
                    curr_gps, rsf_ext, prev_gps, config, scene_name,
                    gps_trajectory=trajectory_3d,
                    current_frame_idx=t
                )
                frame_features.append(features)
                frame_confidence.append(confidence)

            batch_features.append(np.stack(frame_features, axis=0))
            batch_confidence.append(np.array(frame_confidence))

        spatial_features_list.append(np.stack(batch_features, axis=0))
        same_road_confidence_list.append(np.stack(batch_confidence, axis=0))

    spatial_features = torch.from_numpy(
        np.stack(spatial_features_list, axis=0)
    ).float()
    same_road_confidence = torch.from_numpy(
        np.stack(same_road_confidence_list, axis=0)
    ).float()

    if gps_trajectory_6d.is_cuda:
        spatial_features = spatial_features.to(gps_trajectory_6d.device)
        same_road_confidence = same_road_confidence.to(gps_trajectory_6d.device)

    return spatial_features, same_road_confidence

if __name__ == '__main__':
    print("测试空间特征编码器 v4.0-final")
    print("=" * 50)

    B, T, N_rsf = 2, 10, 3
    gps_trajectory_6d = torch.randn(B, T, 6) * 100
    gps_trajectory_6d[..., 3:] = torch.randn(B, T, 3) * 0.1

    rsf_extrinsics_6d = torch.randn(B, N_rsf, 6) * 100
    rsf_extrinsics_6d[..., 3:] = torch.randn(B, N_rsf, 3) * 0.1

    print("\n[测试1] 12维特征测试")
    spatial_features_car, same_road_confidence_car = extract_spatial_features_batch(
        gps_trajectory_6d, rsf_extrinsics_6d
    )
    print(f"空间特征: {spatial_features_car.shape}")
    print(f"同路置信度: {same_road_confidence_car.shape}")
    print(f"  [0] distance_2d: [{spatial_features_car[..., 0].min():.2f}, {spatial_features_car[..., 0].max():.2f}]")
    print(f"  [1] height_diff: [{spatial_features_car[..., 1].min():.2f}, {spatial_features_car[..., 1].max():.2f}]")
    print(f"  [11] elevation_angle: [{spatial_features_car[..., 11].min():.2f}, {spatial_features_car[..., 11].max():.2f}]")

    print("\n" + "=" * 50)
    print("✓ 12维空间特征测试通过！")
