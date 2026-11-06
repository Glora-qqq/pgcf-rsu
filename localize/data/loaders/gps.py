"""Read noisy GPS Six-dimensional posture. """

import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import (
    apply_height_correction,
    convert_angles_deg_to_rad,
    is_aerial_vehicle,
    DEG_TO_RAD
)

def add_gps_orientation_noise(
    orientations: np.ndarray,
    vehicle_name: str,
    config=None
) -> np.ndarray:

    orientations = np.array(orientations, dtype=float)
    N = len(orientations)

    if config is None:
        return orientations.copy()

    if is_aerial_vehicle(vehicle_name):
        at_noise_std_deg = getattr(config, 'at_orientation_noise_std_deg', 0.0)
        if at_noise_std_deg == 0:
            return orientations.copy()

        noise_std_rad = at_noise_std_deg * DEG_TO_RAD
        noise = np.random.normal(0, noise_std_rad, size=(N, 3))
        noisy_orientations = orientations + noise
    else:
        noisy_orientations = orientations.copy()

        noisy_orientations[:, 0] = orientations[:, 0]
        noisy_orientations[:, 1] = orientations[:, 1]

        car_yaw_noise_std_deg = getattr(config, 'car_yaw_straight_noise_std_deg', 0.0)
        car_yaw_turning_ratio = getattr(config, 'car_yaw_turning_ratio', 0.0)

        if car_yaw_noise_std_deg > 0:
            if car_yaw_turning_ratio > 0:
                turning_mask = np.random.rand(N) < car_yaw_turning_ratio
                yaw_noise = np.zeros(N)
                if np.sum(turning_mask) > 0:
                    yaw_noise[turning_mask] = np.random.normal(
                        0, 3 * car_yaw_noise_std_deg * DEG_TO_RAD, np.sum(turning_mask)
                    )
                if np.sum(~turning_mask) > 0:
                    yaw_noise[~turning_mask] = np.random.normal(
                        0, car_yaw_noise_std_deg * DEG_TO_RAD, np.sum(~turning_mask)
                    )
                noisy_orientations[:, 2] = orientations[:, 2] + yaw_noise
            else:
                yaw_noise = np.random.normal(0, car_yaw_noise_std_deg * DEG_TO_RAD, size=N)
                noisy_orientations[:, 2] = orientations[:, 2] + yaw_noise

    noisy_orientations[:, 2] = np.arctan2(
        np.sin(noisy_orientations[:, 2]),
        np.cos(noisy_orientations[:, 2])
    )

    return noisy_orientations

def load_gps_txt(
    txt_file_path: Path,
    scene_name: str,
    vehicle_name: str,
    apply_correction: bool = False,
    add_orientation_noise: bool = True,
    config=None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:

    try:
        data = np.loadtxt(str(txt_file_path))

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] < 6:
            raise ValueError(f"Insufficient data columns, Expect at least6Columns, Actual{data.shape[1]}Columns")

        positions = data[:, :3]
        orientations_raw = data[:, 3:6]

        if not is_aerial_vehicle(vehicle_name):
            original_z_min = positions[:, 2].min()
            original_z_max = positions[:, 2].max()
            positions[:, 2] = 0.0
            print(f"  [GPS] CarVehicle {vehicle_name}: zCoordinates set to0 (OriginalzScope: {original_z_min:.3f} To {original_z_max:.3f}) ")

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
            from utils.coordinates import apply_height_correction
            positions_corrected = np.zeros_like(positions)
            for i in range(len(positions)):
                roll_rad, pitch_rad, yaw_rad = orientations[i]
                positions_corrected[i] = apply_height_correction(
                    positions[i],
                    roll_rad, pitch_rad, yaw_rad,
                    scene_name, vehicle_name
                )
            positions = positions_corrected
            if not is_aerial_vehicle(vehicle_name):
                print(f"  [Height] CarVehicle {vehicle_name}: Height correction complete")

        if config is not None and add_orientation_noise:
            if is_aerial_vehicle(vehicle_name):
                noise_std_deg = getattr(config, 'at_orientation_noise_std_deg', 0.0)
            else:
                noise_std_deg = getattr(config, 'car_yaw_straight_noise_std_deg', 0.0)
        else:
            noise_std_deg = 0.0

        metadata = {
            'file_path': str(txt_file_path),
            'vehicle_name': vehicle_name,
            'num_frames': len(positions),
            'is_aerial': is_aerial_vehicle(vehicle_name),
            'height_correction_applied': apply_correction,
            'data_type': 'gps_noise',
            'orientation_noise_added': add_orientation_noise,
            'orientation_noise_std_deg': noise_std_deg,
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

        return positions, orientations, metadata

    except Exception as e:
        print(f"[ERROR] LoadGPSFile failed: {txt_file_path}")
        print(f"        Error: {e}")
        return None, None, {'error': str(e), 'file_path': str(txt_file_path)}

def load_vehicle_gps(
    data_root: Path,
    scene_name: str,
    vehicle_name: str,
    config=None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:

    vehicle_dir = data_root / scene_name / vehicle_name
    gps_file = vehicle_dir / f"{vehicle_name.lower()}_train_cat_noise.txt"

    if not gps_file.exists():
        gps_file = vehicle_dir / f"{vehicle_name}_train_cat_noise.txt"

    if not gps_file.exists():
        print(f"[WARNING] Not foundGPSNoise File:")
        print(f"           Scenario={scene_name}, Vehicle={vehicle_name}")
        print(f"           Path={gps_file}")
        return None, None, {'error': 'file_not_found'}

    if config is not None:
        add_orientation_noise = getattr(config, 'use_orientation_noise', False)
    else:
        add_orientation_noise = False

    positions, orientations, metadata = load_gps_txt(
        gps_file,
        scene_name,
        vehicle_name,
        apply_correction=True,
        add_orientation_noise=add_orientation_noise,
        config=config
    )

    return positions, orientations, metadata

if __name__ == '__main__':
    data_root = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[3] / 'data')) / 'train'

    print("="*80)
    print("Test Ordinary VehiclesGPS (Car10, Scene1)")
    print("="*80)
    positions, orientations, metadata = load_vehicle_gps(
        data_root,
        'Scene1',
        'Car10'
    )

    if positions is not None:
        print(f"Loaded successfully:")
        print(f"  Frames: {metadata['num_frames']}")
        print(f"  Is Air Vehicle: {metadata['is_aerial']}")
        print(f"  Height correction: {metadata['height_correction_applied']}")
        print(f"  Data Type: {metadata['data_type']}")
        print(f"\nago3FrameGPSLocation:")
        print(positions[:3])
        print(f"\nago3Frame Pose (Radian):")
        print(orientations[:3])

    print("\n" + "="*80)
    print("Test Air TaxiGPS (AT1, Scene3)")
    print("="*80)
    positions, orientations, metadata = load_vehicle_gps(
        data_root,
        'Scene3',
        'AT1'
    )

    if positions is not None:
        print(f"Loaded successfully:")
        print(f"  Frames: {metadata['num_frames']}")
        print(f"  Is Air Vehicle: {metadata['is_aerial']}")
        print(f"  Height correction: {metadata['height_correction_applied']}")
        print(f"  Data Type: {metadata['data_type']}")
        print(f"\nago3FrameGPSLocation:")
        print(positions[:3])
        print(f"\nago3Frame Pose (Radian):")
        print(orientations[:3])
