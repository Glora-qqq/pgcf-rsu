"""Read Vehicle Position True Value. """

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
            raise ValueError(f"Insufficient data columns, Expect at least6Columns, Actual{data.shape[1]}Columns")

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
        print(f"[ERROR] Failed to load track file: {txt_file_path}")
        print(f"        Error: {e}")
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
        print(f"[WARNING] Not foundGTFile:")
        print(f"           Scenario={scene_name}, Vehicle={vehicle_name}")
        print(f"           Path={gt_file}")
        return None, None, None, {'error': 'file_not_found'}

    return load_trajectory_txt(gt_file, scene_name, vehicle_name, apply_correction)
