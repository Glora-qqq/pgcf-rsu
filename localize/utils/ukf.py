"""空中目标使用的无迹卡尔曼滤波。"""

import numpy as np

try:
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False
    print("⚠️  filterpy未安装，UKF滤波不可用")
    print("   安装命令: pip install filterpy")

class UKF3DFilter:

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 1.6,
        dt: float = 0.05
    ):

        if not HAS_FILTERPY:
            raise ImportError("UKF滤波需要filterpy库，请安装: pip install filterpy")

        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        points = MerweScaledSigmaPoints(
            n=6,
            alpha=0.1,
            beta=2.0,
            kappa=0.0
        )

        self.ukf = UnscentedKalmanFilter(
            dim_x=6,
            dim_z=3,
            dt=dt,
            fx=self._state_transition,
            hx=self._measurement_function,
            points=points
        )

        self.ukf.P *= 100.0

        q = process_noise ** 2
        self.ukf.Q = np.diag([
            q, q, q,
            q * 10, q * 10, q * 10
        ])

        r = measurement_noise ** 2
        self.ukf.R = np.diag([r, r, r])

    def _state_transition(self, x, dt):

        F = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ]
        ])
        return F @ x

    def _measurement_function(self, x):

        return x[:3]

    def filter_trajectory(self, gps_positions: np.ndarray) -> np.ndarray:

        T = len(gps_positions)
        filtered_positions = np.zeros_like(gps_positions)

        self.ukf.x = np.array([
            gps_positions[0, 0],
            gps_positions[0, 1],
            gps_positions[0, 2],
            0.0,
            0.0,
            0.0
        ])

        for t in range(T):
            self.ukf.predict()

            self.ukf.update(gps_positions[t])

            filtered_positions[t] = self.ukf.x[:3]

        return filtered_positions

    def reset(self):

        self.ukf.P *= 100.0

def apply_ukf_to_gps(
    gps_positions: np.ndarray,
    process_noise: float = 0.1,
    measurement_noise: float = 1.6,
    dt: float = 0.05
) -> np.ndarray:

    if not HAS_FILTERPY:
        print("⚠️  filterpy未安装，返回原始GPS数据")
        return gps_positions

    ukf = UKF3DFilter(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        dt=dt
    )

    return ukf.filter_trajectory(gps_positions)

def normalize_angle(angle_rad):

    while angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    while angle_rad < -np.pi:
        angle_rad += 2 * np.pi
    return angle_rad

class UKF6DFilter:

    def __init__(
        self,
        process_noise_pos: float = 0.1,
        process_noise_ori: float = 0.01,
        measurement_noise_pos: float = 1.6,
        measurement_noise_ori: float = 0.05,
        dt: float = 0.05
    ):

        if not HAS_FILTERPY:
            raise ImportError("UKF滤波需要filterpy库，请安装: pip install filterpy")

        self.dt = dt

        points = MerweScaledSigmaPoints(
            n=12,
            alpha=0.1,
            beta=2.0,
            kappa=0.0
        )

        self.ukf = UnscentedKalmanFilter(
            dim_x=12,
            dim_z=6,
            dt=dt,
            fx=self._state_transition,
            hx=self._measurement_function,
            points=points
        )

        self.ukf.P *= 100.0

        q_pos = process_noise_pos ** 2
        q_ori = process_noise_ori ** 2
        self.ukf.Q = np.diag([
            q_pos, q_pos, q_pos,
            q_ori, q_ori, q_ori,
            q_pos*10, q_pos*10, q_pos*10,
            q_ori*10, q_ori*10, q_ori*10
        ])

        r_pos = measurement_noise_pos ** 2
        r_ori = measurement_noise_ori ** 2
        self.ukf.R = np.diag([
            r_pos, r_pos, r_pos,
            r_ori, r_ori, r_ori
        ])

    def _state_transition(self, x, dt):

        F = np.array([
            [1, 0, 0, 0, 0, 0, dt, 0,  0,  0,  0,  0 ],
            [0, 1, 0, 0, 0, 0, 0,  dt, 0,  0,  0,  0 ],
            [0, 0, 1, 0, 0, 0, 0,  0,  dt, 0,  0,  0 ],
            [0, 0, 0, 1, 0, 0, 0,  0,  0,  dt, 0,  0 ],
            [0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  dt, 0 ],
            [0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0,  dt],
            [0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  0 ],
            [0, 0, 0, 0, 0, 0, 0,  1,  0,  0,  0,  0 ],
            [0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0,  0 ],
            [0, 0, 0, 0, 0, 0, 0,  0,  0,  1,  0,  0 ],
            [0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0 ],
            [0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1 ]
        ])

        new_state = F @ x

        new_state[3] = normalize_angle(new_state[3])
        new_state[4] = normalize_angle(new_state[4])
        new_state[5] = normalize_angle(new_state[5])

        return new_state

    def _measurement_function(self, x):

        return x[:6]

    def filter_trajectory(self, gps_poses_6d: np.ndarray) -> np.ndarray:

        T = len(gps_poses_6d)
        filtered_poses = np.zeros_like(gps_poses_6d)

        self.ukf.x = np.array([
            gps_poses_6d[0, 0],
            gps_poses_6d[0, 1],
            gps_poses_6d[0, 2],
            gps_poses_6d[0, 3],
            gps_poses_6d[0, 4],
            gps_poses_6d[0, 5],
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])

        for t in range(T):
            self.ukf.predict()

            self.ukf.update(gps_poses_6d[t])

            filtered_poses[t] = self.ukf.x[:6]

            filtered_poses[t, 3] = normalize_angle(filtered_poses[t, 3])
            filtered_poses[t, 4] = normalize_angle(filtered_poses[t, 4])
            filtered_poses[t, 5] = normalize_angle(filtered_poses[t, 5])

        return filtered_poses

    def reset(self):

        self.ukf.P *= 100.0

if __name__ == "__main__":
    np.random.seed(42)
    T = 100
    t = np.linspace(0, 10, T)

    gt = np.column_stack([
        np.sin(t),
        np.cos(t),
        t / 10
    ])

    noise = np.random.normal(0, 0.5, (T, 3))
    gps_noisy = gt + noise

    if HAS_FILTERPY:
        gps_filtered = apply_ukf_to_gps(gps_noisy)

        error_noisy = np.linalg.norm(gps_noisy - gt, axis=1).mean()
        error_filtered = np.linalg.norm(gps_filtered - gt, axis=1).mean()

        print(f"原始GPS误差: {error_noisy:.4f}")
        print(f"UKF滤波误差: {error_filtered:.4f}")
        print(f"改善: {(1 - error_filtered/error_noisy)*100:.1f}%")
    else:
        print("filterpy未安装，跳过测试")
