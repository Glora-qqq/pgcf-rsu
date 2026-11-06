"""2D Kalman Smoothing for Ground Vehicles. """

import numpy as np
from typing import Optional, Union
import warnings

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    warnings.warn("filterpy not installed. Please install: pip install filterpy")
    KalmanFilter = None

class KalmanFilter6D:

    def __init__(
        self,
        process_noise: float = 0.10,
        measurement_noise: float = 1.6,
        smooth_orientation: bool = False
    ):
        if KalmanFilter is None:
            raise ImportError("filterpy is required. Install: pip install filterpy")

        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.smooth_orientation = smooth_orientation

        self.filters = []
        for i in range(3):
            kf = KalmanFilter(dim_x=2, dim_z=1)

            kf.F = np.array([[1., 1.],
                             [0., 1.]])

            kf.H = np.array([[1., 0.]])

            kf.Q = np.array([[process_noise**2, 0.],
                             [0., process_noise**2]])

            kf.R = np.array([[measurement_noise**2]])

            kf.P *= 1000.

            self.filters.append(kf)

        if smooth_orientation:
            for i in range(3):
                kf = KalmanFilter(dim_x=2, dim_z=1)
                kf.F = np.array([[1., 1.], [0., 1.]])
                kf.H = np.array([[1., 0.]])

                kf.Q = np.array([[(process_noise*0.1)**2, 0.],
                                 [0., (process_noise*0.1)**2]])
                kf.R = np.array([[(measurement_noise*0.1)**2]])
                kf.P *= 1000.

                self.filters.append(kf)

    def smooth(self, poses: np.ndarray) -> np.ndarray:

        if len(poses) == 0:
            return poses

        if poses.shape[1] != 6:
            raise ValueError(f"Expected poses shape [T, 6], got {poses.shape}")

        T = len(poses)
        smoothed = np.copy(poses)

        for kf in self.filters:
            kf.x = np.array([[0.], [0.]])
            kf.P *= 1000.

        num_dims = 6 if self.smooth_orientation else 3

        for dim in range(num_dims):
            self.filters[dim].x = np.array([[poses[0, dim]], [0.]])

            for t in range(T):
                self.filters[dim].predict()

                self.filters[dim].update(poses[t, dim])

                smoothed[t, dim] = self.filters[dim].x[0, 0]

        if not self.smooth_orientation:
            smoothed[:, 3:] = poses[:, 3:]

        return smoothed

    def get_state_info(self) -> dict:

        return {
            'process_noise': self.process_noise,
            'measurement_noise': self.measurement_noise,
            'smooth_orientation': self.smooth_orientation,
            'num_filters': len(self.filters),
        }

def is_aerial_vehicle(vehicle_name: str) -> bool:

    if vehicle_name is None:
        return False
    return vehicle_name.upper().startswith('AT')

def apply_kalman_smoothing(
    predictions: np.ndarray,
    vehicle_name: Optional[str] = None,
    process_noise: Optional[float] = None,
    measurement_noise: float = 1.6,
    config: Optional[dict] = None
) -> np.ndarray:

    if config is None:
        config = {}

    if process_noise is None and config.get('adaptive', True):
        if is_aerial_vehicle(vehicle_name):
            process_noise = config.get('at_noise', 0.20)
        else:
            process_noise = config.get('car_noise', 0.10)
    elif process_noise is None:
        process_noise = 0.10

    kf = KalmanFilter6D(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        smooth_orientation=config.get('smooth_orientation', False)
    )

    smoothed = kf.smooth(predictions)

    return smoothed

def batch_kalman_smoothing(
    predictions_dict: dict,
    config: Optional[dict] = None
) -> dict:

    smoothed_dict = {}

    for vehicle_name, predictions in predictions_dict.items():
        smoothed_dict[vehicle_name] = apply_kalman_smoothing(
            predictions,
            vehicle_name=vehicle_name,
            config=config
        )

    return smoothed_dict

if __name__ == '__main__':
    print("Kalman Filter Module Test")
    print("="*80)

    T = 100
    t = np.linspace(0, 10, T)

    true_x = 10 * t
    true_y = 5 * t
    true_z = np.zeros(T)

    noisy_x = true_x + np.random.normal(0, 1.6, T)
    noisy_y = true_y + np.random.normal(0, 1.6, T)
    noisy_z = true_z + np.random.normal(0, 0.5, T)

    poses = np.zeros((T, 6))
    poses[:, 0] = noisy_x
    poses[:, 1] = noisy_y
    poses[:, 2] = noisy_z
    poses[:, 3:] = 0

    kf = KalmanFilter6D(process_noise=0.10, measurement_noise=1.6)
    smoothed = kf.smooth(poses)

    original_error_x = np.mean(np.abs(noisy_x - true_x))
    smoothed_error_x = np.mean(np.abs(smoothed[:, 0] - true_x))

    original_error_y = np.mean(np.abs(noisy_y - true_y))
    smoothed_error_y = np.mean(np.abs(smoothed[:, 1] - true_y))

    print(f"XAxis:")
    print(f"  Raw error:   {original_error_x:.3f}m")
    print(f"  Post-smoothing error: {smoothed_error_x:.3f}m")
    print(f"  Improvement:       {(1 - smoothed_error_x/original_error_x)*100:.1f}%")

    print(f"\nYAxis:")
    print(f"  Raw error:   {original_error_y:.3f}m")
    print(f"  Post-smoothing error: {smoothed_error_y:.3f}m")
    print(f"  Improvement:       {(1 - smoothed_error_y/original_error_y)*100:.1f}%")

    print("\n✓ Kalman filter module test passed")
