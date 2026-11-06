"""Select Kalman or Unscented Kalman by vehicle type as output smoothing. """

import torch
import numpy as np
from typing import Union, List, Optional
import warnings

try:
    from .kalman import KalmanFilter6D
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    warnings.warn("Kalman filter not available. Install filterpy: pip install filterpy")

try:
    from .ukf import UKF3DFilter
    UKF_AVAILABLE = True
except ImportError:
    UKF_AVAILABLE = False
    warnings.warn("UKF filter not available.")

class VehicleAdaptiveOutputFilter:

    def __init__(self, config):

        self.config = config
        self.device = None

        self.car_process_noise = getattr(config, 'output_car_kalman_process_noise', 0.10)
        self.car_measurement_noise = getattr(config, 'output_car_kalman_measurement_noise', 1.6)

        self.at_ukf_process_noise = getattr(config, 'output_at_ukf_process_noise', 0.10)
        self.at_ukf_measurement_noise = getattr(config, 'output_at_ukf_measurement_noise', 0.8)

        self._first_call = True

    def apply(self, poses_6d: torch.Tensor, vehicle_names: List[str]) -> torch.Tensor:

        if self.device is None:
            self.device = poses_6d.device

        if self._first_call:
            print(f"\n[VehicleAdaptiveOutputFilter] Initialization")
            print(f"  - Batch size: {poses_6d.shape[0]}")
            print(f"  - Time steps: {poses_6d.shape[1]}")
            print(f"  - Vehicles: {vehicle_names}")
            print(f"  - Car 2D KalmanParameter: process_noise={self.car_process_noise}, measurement_noise={self.car_measurement_noise}")
            print(f"  - AT 3D UKFParameter: process_noise={self.at_ukf_process_noise}, measurement_noise={self.at_ukf_measurement_noise}")
            self._first_call = False

        B, T, _ = poses_6d.shape
        assert len(vehicle_names) == B, f"vehicle_namesLength ({len(vehicle_names)}) Must be equal tobatch size ({B})"

        smoothed_batch = []
        for i in range(B):
            vehicle_name = vehicle_names[i]
            single_pose = poses_6d[i]

            if self._is_car(vehicle_name):
                smoothed = self._apply_car_filter(single_pose)
            elif self._is_at(vehicle_name):
                smoothed = self._apply_at_filter(single_pose)
            else:
                smoothed = single_pose

            smoothed_batch.append(smoothed)

        smoothed_poses_6d = torch.stack(smoothed_batch, dim=0)

        return smoothed_poses_6d

    def _is_car(self, vehicle_name: str) -> bool:

        return vehicle_name.lower().startswith('car')

    def _is_at(self, vehicle_name: str) -> bool:

        return vehicle_name.lower().startswith('at')

    def _apply_car_filter(self, pose_6d: torch.Tensor) -> torch.Tensor:

        if not KALMAN_AVAILABLE:
            return pose_6d

        T, _ = pose_6d.shape
        device = pose_6d.device

        pose_6d_np = pose_6d.detach().cpu().numpy()

        try:
            kf = KalmanFilter6D(
                process_noise=self.car_process_noise,
                measurement_noise=self.car_measurement_noise,
                smooth_orientation=False
            )

            smoothed_6d_np = kf.smooth(pose_6d_np)

            xy_smoothed = smoothed_6d_np[:, :2]
            z_original = pose_6d_np[:, 2:3]
            orientations_original = pose_6d_np[:, 3:]

            result_np = np.concatenate([
                xy_smoothed,
                z_original,
                orientations_original
            ], axis=-1)

            pose_smoothed = torch.from_numpy(result_np).float().to(device)

        except Exception as e:
            print(f"[WARNING] CarFailure to filter: {e}")
            pose_smoothed = pose_6d

        return pose_smoothed

    def _apply_at_filter(self, pose_6d: torch.Tensor) -> torch.Tensor:

        if not UKF_AVAILABLE:
            return pose_6d

        device = pose_6d.device

        try:
            pose_np = pose_6d.detach().cpu().numpy()

            positions = pose_np[:, :3]
            orientations = pose_np[:, 3:6]

            ukf = UKF3DFilter(
                process_noise=self.at_ukf_process_noise,
                measurement_noise=self.at_ukf_measurement_noise,
                dt=0.05
            )

            positions_smoothed = ukf.filter(positions)

            result_np = np.column_stack([
                positions_smoothed,
                orientations
            ])

            pose_smoothed = torch.from_numpy(result_np).float().to(device)

        except Exception as e:
            print(f"[WARNING] AT UKFFailure to filter: {e}")
            pose_smoothed = pose_6d

        return pose_smoothed

def create_adaptive_filter(config) -> Optional[VehicleAdaptiveOutputFilter]:

    if getattr(config, 'use_output_adaptive_filter', False):
        return VehicleAdaptiveOutputFilter(config)
    else:
        return None
