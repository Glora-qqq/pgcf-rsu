"""From GPS Text Read Vehicle Appears Frame Range. """

import os
import os
import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

def find_gps_file(scene: str, vehicle: str, data_root: Path, is_test: bool = False) -> Optional[Path]:

    if data_root is None:
        raise ValueError("data_rootParameter cannot beNone")

    vehicle_dir = data_root / scene / vehicle

    if not vehicle_dir.exists():
        print(f"[WARNING] Vehicle directory does not exist: {vehicle_dir}")
        return None

    vehicle_num = vehicle.replace('Car', '').replace('car', '').replace('AT', '').replace('UAV', '')

    if is_test:
        possible_patterns = [
            f"{vehicle}_test_cat_gt.txt",
            f"{vehicle}_test_cat_noise.txt",
            f"{vehicle.lower()}_test_cat_gt.txt",
            f"{vehicle.lower()}_test_cat_noise.txt",
            f"car{vehicle_num}_test_cat_gt.txt",
            f"car{vehicle_num}_test_cat_noise.txt",
            f"Car{vehicle_num}_test_cat_gt.txt",
            f"Car{vehicle_num}_test_cat_noise.txt",
        ]
    else:
        possible_patterns = [
            f"{vehicle}_train_cat_gt.txt",
            f"{vehicle.lower()}_train_cat_gt.txt",
            f"car{vehicle_num}_train_cat_gt.txt",
            f"{vehicle}_train_cat_gt_noise.txt",
            f"{vehicle.lower()}_train_cat_gt_noise.txt",
            f"car{vehicle_num}_train_cat_gt_noise.txt",
        ]

    for pattern in possible_patterns:
        gps_file = vehicle_dir / pattern
        if gps_file.exists():
            return gps_file

    print(f"[WARNING] Not foundGPSFile: {vehicle_dir}/[{', '.join(possible_patterns[:3])}...]")
    return None

def extract_frame_range_from_gps(gps_file_path: Path) -> Tuple[int, int, int]:

    try:
        data = np.loadtxt(gps_file_path)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        frame_ids = data[:, 6].astype(int)

        start_frame = int(frame_ids[0])
        end_frame = int(frame_ids[-1])
        total_frames = len(frame_ids)

        expected_frames = end_frame - start_frame + 1
        if total_frames != expected_frames:
            print(f"[WARNING] GPSFrame discontinuity: Expected{expected_frames}Frame, Actual{total_frames}Frame")

        return start_frame, end_frame, total_frames

    except Exception as e:
        error_msg = f"[ERROR] ReadGPSFile failed: {gps_file_path}\nError: {e}"
        raise RuntimeError(error_msg)

def extract_gps_coordinates(gps_file_path: Path) -> np.ndarray:

    try:
        data = np.loadtxt(gps_file_path)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        coordinates = data[:, :3]

        return coordinates

    except Exception as e:
        error_msg = f"[ERROR] ReadGPSCoordinates failed: {gps_file_path}\nError: {e}"
        raise RuntimeError(error_msg)

def get_vehicle_frame_info(scene: str, vehicle: str, data_root: Path, is_test: bool = False) -> Dict:

    gps_file = find_gps_file(scene, vehicle, data_root, is_test)

    if gps_file is None:
        data_type = "Test set" if is_test else "Training Sets"
        error_msg = f"[ERROR] Not foundGPSFile: {data_type} {scene}/{vehicle}\n" \
                   f"   Check Path: {data_root}/{scene}/{vehicle}/"
        raise FileNotFoundError(error_msg)

    start_frame, end_frame, total_frames = extract_frame_range_from_gps(gps_file)

    start_index = start_frame - 1
    end_index = end_frame - 1

    expected_length = end_index - start_index + 1
    if expected_length != total_frames:
        print(f"[WARNING] Inconsistent index calculation: {expected_length} vs {total_frames}")

    gps_coordinates = extract_gps_coordinates(gps_file)

    frame_ids = np.arange(start_frame, end_frame + 1)

    return {
        'gps_file': str(gps_file),
        'start_frame': start_frame,
        'end_frame': end_frame,
        'total_frames': total_frames,
        'start_index': start_index,
        'end_index': end_index,
        'gps_coordinates': gps_coordinates,
        'frame_ids': frame_ids,
    }

def get_scene_vehicles_frame_info(
    scene: str,
    vehicles: List[str],
    data_root: Path,
    is_test: bool = False
) -> Dict[str, Dict]:

    vehicles_info = {}

    for vehicle in vehicles:
        try:
            frame_info = get_vehicle_frame_info(scene, vehicle, data_root, is_test)
            vehicles_info[vehicle] = frame_info
        except FileNotFoundError as e:
            print(f"[WARNING] Skip Vehicle {vehicle}: {e}")
            continue

    return vehicles_info
