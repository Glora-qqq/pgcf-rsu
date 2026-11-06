"""读取车辆位姿真值。"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import (
    apply_height_correction,
    convert_angles_deg_to_rad,
    is_aerial_vehicle
)

def load_trajectory_txt(
    txt_file_path: Path,
    scene_name: str,
    vehicle_name: str,
    apply_correction: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict]:

    try:
        data = np.loadtxt(str(txt_file_path))

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 6:
            raise ValueError(f"数据列数不足，期望至少6列，实际{data.shape[1]}列")

        positions = data[:, :3]
        orientations_raw = data[:, 3:6]

        frame_ids = None
        if data.shape[1] >= 7:
            frame_ids = data[:, 6].astype(int)

        orientations = np.zeros_like(orientations_raw)
        for i in range(len(orientations_raw)):
            roll, pitch, yaw = orientations_raw[i]
            roll_rad, pitch_rad, yaw_rad = convert_angles_deg_to_rad(
                roll, pitch, yaw, vehicle_name
            )
            orientations[i] = [roll_rad, pitch_rad, yaw_rad]

        if apply_correction:
            positions_corrected = np.zeros_like(positions)
            for i in range(len(positions)):
                roll_rad, pitch_rad, yaw_rad = orientations[i]
                positions_corrected[i] = apply_height_correction(
                    positions[i],
                    roll_rad, pitch_rad, yaw_rad,
                    scene_name, vehicle_name
                )
            positions = positions_corrected

        metadata = {
            'file_path': str(txt_file_path),
            'vehicle_name': vehicle_name,
            'num_frames': len(positions),
            'is_aerial': is_aerial_vehicle(vehicle_name),
            'height_correction_applied': apply_correction,
        }

        if frame_ids is not None:
            start_frame_id = int(frame_ids[0])
            end_frame_id = int(frame_ids[-1])

            start_index = start_frame_id - 1
            end_index = end_frame_id - 1

            metadata['frame_range'] = (start_index, end_index)
            metadata['frame_ids_1based'] = (start_frame_id, end_frame_id)
            metadata['has_frame_ids'] = True
        else:
            metadata['frame_range'] = None
            metadata['frame_ids_1based'] = None
            metadata['has_frame_ids'] = False

        return positions, orientations, frame_ids, metadata

    except Exception as e:
        print(f"[ERROR] 加载轨迹文件失败: {txt_file_path}")
        print(f"        错误: {e}")
        return None, None, None, {'error': str(e), 'file_path': str(txt_file_path)}

def load_vehicle_gt(
    data_root: Path,
    scene_name: str,
    vehicle_name: str,
    apply_correction: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict]:

    vehicle_dir = data_root / scene_name / vehicle_name
    gt_file = vehicle_dir / f"{vehicle_name.lower()}_train_cat_gt.txt"

    if not gt_file.exists():
        gt_file = vehicle_dir / f"{vehicle_name}_train_cat_gt.txt"

    if not gt_file.exists():
        print(f"[WARNING] 找不到GT文件:")
        print(f"           场景={scene_name}, 车辆={vehicle_name}")
        print(f"           路径={gt_file}")
        return None, None, None, {'error': 'file_not_found'}

    return load_trajectory_txt(gt_file, scene_name, vehicle_name, apply_correction)

if __name__ == '__main__':
    data_root = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[3] / 'data')) / 'train'

    print("="*80)
    print("测试普通车辆 (Car10, Scene1)")
    print("="*80)
    positions, orientations, frame_ids, metadata = load_vehicle_gt(
        data_root,
        'Scene1',
        'Car10'
    )

    if positions is not None:
        print(f"加载成功:")
        print(f"  帧数: {metadata['num_frames']}")
        print(f"  是否空中车辆: {metadata['is_aerial']}")
        print(f"  高度修正: {metadata['height_correction_applied']}")
        print(f"\n前3帧位置:")
        print(positions[:3])
        print(f"\n前3帧姿态 (弧度):")
        print(orientations[:3])

    print("\n" + "="*80)
    print("测试空中出租车 (AT1, Scene3)")
    print("="*80)
    positions, orientations, frame_ids, metadata = load_vehicle_gt(
        data_root,
        'Scene3',
        'AT1'
    )

    if positions is not None:
        print(f"加载成功:")
        print(f"  帧数: {metadata['num_frames']}")
        print(f"  是否空中车辆: {metadata['is_aerial']}")
        print(f"  高度修正: {metadata['height_correction_applied']}")
        print(f"\n前3帧位置:")
        print(positions[:3])
        print(f"\n前3帧姿态 (弧度):")
        print(orientations[:3])
