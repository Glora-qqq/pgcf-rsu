"""Multi-way side features for assembly testing, GPS and geometric quantities. """

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.coordinates import (
    convert_angles_deg_to_rad,
    apply_height_correction
)
from utils.constants import DEG_TO_RAD, is_aerial_vehicle
from models.geometry import extract_spatial_features_batch

class MultiRSFPose6DTestDataset(Dataset):

    def __init__(
        self,
        test_data_root: Path,
        wifo_features_root: Path,
        scene_name: str,
        vehicle_name: str,
        rsf_list: List[str],
        gps_file_suffix: str = '_test_cat_noise.txt',
        wifo_feature_template: str = "{vehicle}_{rsf}_wifo_features.npy",
        rsf_extrinsic_template: str = "RSF_position_{scene_id}.txt",
        use_input_kalman: bool = False,
        input_kalman_config: Optional[Dict] = None,
        config=None,
    ):

        self.test_data_root = Path(test_data_root)
        self.wifo_features_root = Path(wifo_features_root)
        self.scene_name = scene_name
        self.vehicle_name = vehicle_name
        self.rsf_list = rsf_list
        self.gps_file_suffix = gps_file_suffix
        self.wifo_feature_template = wifo_feature_template
        self.rsf_extrinsic_template = rsf_extrinsic_template

        self.use_input_kalman = use_input_kalman
        self.input_kalman_config = input_kalman_config or {
            'process_noise': 0.10,
            'measurement_noise': 1.6,
        }

        self.config = config

        self.scene_dir_mapping = {
            'Scene1': 'Scene_1',
            'Scene2': 'Scene_2',
            'Scene3': 'Scene_3',
        }

        self._load_data()

    def _load_data(self):

        self.gps_poses = self._load_gps_poses()

        self.wifo_features = self._load_wifo_features()

        self.rsf_extrinsics = self._load_rsf_extrinsics()

        self._validate_data()

        print(f"[{self.scene_name}/{self.vehicle_name}] Data loading complete:")
        print(f"  GPSPosture: {self.gps_poses.shape}")
        print(f"  WiFoFeatures: {[(rsf, feat.shape) for rsf, feat in self.wifo_features.items()]}")
        print(f"  RSFExternal Ginseng: {len(self.rsf_extrinsics)} PieceRSF")

    def _load_gps_poses(self) -> np.ndarray:

        scene_dir = self.scene_dir_mapping[self.scene_name]
        gps_file = (
            self.test_data_root /
            scene_dir /
            self.vehicle_name /
            f"{self.vehicle_name}{self.gps_file_suffix}"
        )

        if not gps_file.exists():
            raise FileNotFoundError(f"GPSFile does not exist: {gps_file}")

        gps_data = np.loadtxt(gps_file)

        positions = gps_data[:, :3]
        angles = gps_data[:, 3:6]
        frame_ids = gps_data[:, 6]

        processed_poses = []
        for i in range(len(gps_data)):
            pos = positions[i]
            roll, pitch, yaw = angles[i]
            frame_id = frame_ids[i]

            roll_rad, pitch_rad, yaw_rad = convert_angles_deg_to_rad(
                roll, pitch, yaw, self.vehicle_name
            )

            corrected_pos = apply_height_correction(
                pos, roll_rad, pitch_rad, yaw_rad,
                self.scene_name, self.vehicle_name
            )

            processed_poses.append([
                corrected_pos[0], corrected_pos[1], corrected_pos[2],
                roll_rad, pitch_rad, yaw_rad,
                frame_id
            ])

        gps_poses_array = np.array(processed_poses, dtype=np.float32)

        if self.use_input_kalman:
            gps_poses_6d = gps_poses_array[:, :6]

            smoothed_6d = self._apply_input_kalman(gps_poses_6d)

            gps_poses_array[:, :6] = smoothed_6d

        return gps_poses_array

    def _apply_input_kalman(self, gps_poses_6d: np.ndarray) -> np.ndarray:

        if self.vehicle_name.startswith('AT'):
            if hasattr(self, 'config') and getattr(self.config, 'use_at_ukf', False):
                try:
                    from utils.ukf import apply_ukf_to_gps

                    filter_orientation = getattr(self.config, 'at_ukf_orientation', False)
                    process_noise = getattr(self.config, 'at_ukf_process_noise', 0.1)
                    measurement_noise = getattr(self.config, 'at_ukf_measurement_noise', 1.6)
                    dt = getattr(self.config, 'at_ukf_dt', 0.05)

                    if filter_orientation:
                        process_noise_ori = getattr(self.config, 'at_ukf_process_noise_ori', 0.01)
                        measurement_noise_ori = getattr(self.config, 'at_ukf_measurement_noise_ori', 0.05)

                        smoothed_poses = apply_ukf_to_gps(
                            gps_poses_6d=gps_poses_6d,
                            process_noise=process_noise,
                            measurement_noise=measurement_noise,
                            dt=dt,
                            filter_orientation=True,
                            process_noise_ori=process_noise_ori,
                            measurement_noise_ori=measurement_noise_ori
                        )
                        print(f"  [UKF] {self.vehicle_name}: 6D UKFFilter complete (xyz+Posture) ")
                    else:
                        positions = gps_poses_6d[:, :3]
                        orientations = gps_poses_6d[:, 3:6]

                        positions_filtered = apply_ukf_to_gps(
                            gps_positions=positions,
                            process_noise=process_noise,
                            measurement_noise=measurement_noise,
                            dt=dt
                        )

                        smoothed_poses = np.column_stack([positions_filtered, orientations])
                        print(f"  [UKF] {self.vehicle_name}: 3D UKFFilter complete (xyz) ")

                    return smoothed_poses

                except Exception as e:
                    print(f"  [UKF] [WARNING] {self.vehicle_name} UKFFailure to filter: {e}, Use originalGPS")
                    return gps_poses_6d
            else:
                print(f"  [Info] {self.vehicle_name}: Not enabledUKF, Use originalGPS")
                return gps_poses_6d

        try:
            from filterpy.kalman import KalmanFilter
        except ImportError:
            print(f"  [Enter Kalman] filterpyNot Installed, Skip Kalman filtering")
            return gps_poses_6d

        T = len(gps_poses_6d)
        smoothed_poses = np.zeros_like(gps_poses_6d)

        process_noise = self.input_kalman_config.get('process_noise', 0.10)
        measurement_noise = self.input_kalman_config.get('measurement_noise', 1.6)

        print(f"  [Enter Kalman] {self.vehicle_name}: Apply (process_noise={process_noise})")
        print(f"  [Enter Kalman] ⚠️  Only forxyand angle application, KeepzUnchanged (Avoid breaking height correction) ")

        for dim in [0, 1, 3, 4, 5]:
            kf = KalmanFilter(dim_x=2, dim_z=1)
            kf.x = np.array([gps_poses_6d[0, dim], 0.0])
            kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
            kf.H = np.array([[1.0, 0.0]])
            kf.R = measurement_noise
            kf.Q = np.eye(2) * process_noise
            kf.P *= 10.0

            for t in range(T):
                kf.predict()
                kf.update(gps_poses_6d[t, dim])
                smoothed_poses[t, dim] = kf.x[0]

        smoothed_poses[:, 2] = gps_poses_6d[:, 2]

        return smoothed_poses

    def _load_wifo_features(self) -> Dict[str, np.ndarray]:

        wifo_features = {}

        for rsf_name in self.rsf_list:
            wifo_file = (
                self.wifo_features_root /
                self.scene_name /
                self.vehicle_name /
                self.wifo_feature_template.format(
                    vehicle=self.vehicle_name,
                    rsf=rsf_name
                )
            )

            if not wifo_file.exists():
                raise FileNotFoundError(f"WiFoFeature file does not exist: {wifo_file}")

            features = np.load(wifo_file)

            if features.ndim != 2 or features.shape[1] != 512:
                raise ValueError(
                    f"WiFoFeature dimension error: {wifo_file}\n"
                    f"Expectations: [T, 512], Actual: {features.shape}"
                )

            wifo_features[rsf_name] = features

        return wifo_features

    def _load_rsf_extrinsics(self) -> Dict[str, np.ndarray]:

        scene_id = self.scene_name.replace('Scene', '')

        scene_dir = self.scene_dir_mapping[self.scene_name]
        rsf_file = (
            self.test_data_root /
            scene_dir /
            self.rsf_extrinsic_template.format(scene_id=scene_id)
        )

        if not rsf_file.exists():
            raise FileNotFoundError(f"RSFExternal reference file does not exist: {rsf_file}")

        try:
            with open(rsf_file, 'r', encoding='utf-8') as f:
                rsf_dict = json.load(f)
        except Exception as e:
            raise ValueError(f"RSFExternal GinsengJSONParsing failed: {rsf_file}\nError: {e}")

        rsf_extrinsics = {}
        for rsf_name in self.rsf_list:
            if rsf_name not in rsf_dict:
                raise ValueError(f"RSFExternal parameters are missing{rsf_name}: {rsf_file}")

            rsf_data = rsf_dict[rsf_name]
            x = float(rsf_data['X'])
            y = float(rsf_data['Y'])
            z = float(rsf_data['Z'])
            roll = float(rsf_data['Roll']) * DEG_TO_RAD
            pitch = float(rsf_data['Pitch']) * DEG_TO_RAD
            yaw = float(rsf_data['Yaw']) * DEG_TO_RAD

            rsf_extrinsics[rsf_name] = np.array(
                [x, y, z, roll, pitch, yaw],
                dtype=np.float32
            )

        return rsf_extrinsics

    def _validate_data(self):

        num_gps_frames = len(self.gps_poses)

        for rsf_name, features in self.wifo_features.items():
            num_wifo_frames = len(features)

            if num_wifo_frames != num_gps_frames:
                print(
                    f"[WARNING] {self.scene_name}/{self.vehicle_name}/{rsf_name}: "
                    f"WiFoNumber of feature frames({num_wifo_frames}) != GPSFrames({num_gps_frames})"
                )

                min_frames = min(num_wifo_frames, num_gps_frames)
                self.gps_poses = self.gps_poses[:min_frames]
                self.wifo_features[rsf_name] = features[:min_frames]

                print(f"  truncated to {min_frames} Frame")

        if len(self.rsf_extrinsics) != len(self.rsf_list):
            raise ValueError(
                f"RSFExternal parameter quantity mismatch: "
                f"Required{len(self.rsf_list)}, Actual{len(self.rsf_extrinsics)}"
            )

    def __len__(self) -> int:

        return 1

    def __getitem__(self, idx: int) -> Dict:

        gps_poses = self.gps_poses

        wifo_features = {
            rsf_name: features
            for rsf_name, features in self.wifo_features.items()
        }

        rsf_extrinsics = self.rsf_extrinsics.copy()

        meta = {
            'scene': self.scene_name,
            'vehicle': self.vehicle_name,
            'num_frames': len(gps_poses),
        }

        return {
            'gps_pose': gps_poses,
            'wifo_features': wifo_features,
            'rsf_extrinsics': rsf_extrinsics,
            'meta': meta,
        }

def collate_test_batch(batch: List[Dict]) -> Dict:

    sample = batch[0]

    gps_poses_with_id = torch.tensor(sample['gps_pose'], dtype=torch.float32)
    gps_poses_6d = gps_poses_with_id[:, :6]
    frame_ids = gps_poses_with_id[:, 6]

    rsf_names = list(sample['wifo_features'].keys())
    T = gps_poses_6d.shape[0]
    N_rsf = len(rsf_names)

    all_rsf_extrinsics = []
    for rsf_name in rsf_names:
        rsf_ext = torch.tensor(sample['rsf_extrinsics'][rsf_name], dtype=torch.float32)
        all_rsf_extrinsics.append(rsf_ext)
    all_rsf_extrinsics = torch.stack(all_rsf_extrinsics)

    class SimpleConfig:
        def __init__(self):
            self.use_spatial_features = True

    simple_config = SimpleConfig()

    spatial_features_all, _ = extract_spatial_features_batch(
        gps_poses_6d.unsqueeze(0),
        all_rsf_extrinsics.unsqueeze(0),
        simple_config,
        scene_names=[sample['meta']['scene']]
    )
    spatial_features_all = spatial_features_all.squeeze(0)

    rsf_data_dict = {}
    for idx, rsf_name in enumerate(rsf_names):
        csi_features = torch.tensor(
            sample['wifo_features'][rsf_name],
            dtype=torch.float32
        )

        spatial_features = spatial_features_all[:, idx, :]

        rsf_data_dict[rsf_name] = {
            'csi_features': csi_features.unsqueeze(0),
            'spatial_features': spatial_features.unsqueeze(0)
        }

    return {
        'scene_name': sample['meta']['scene'],
        'rsf_data_dict': rsf_data_dict,
        'gps_poses_6d': gps_poses_6d.unsqueeze(0),
        'vehicle_names': [sample['meta']['vehicle']],
        'rsf_list': rsf_names,
        'n_rsf': len(rsf_names),
        'meta': [sample['meta']],
        'frame_ids': frame_ids,
    }

def create_test_dataloader(
    test_data_root: Path,
    wifo_features_root: Path,
    scene_name: str,
    vehicle_name: str,
    rsf_list: List[str],
    batch_size: int = 32,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:

    dataset = MultiRSFPose6DTestDataset(
        test_data_root=test_data_root,
        wifo_features_root=wifo_features_root,
        scene_name=scene_name,
        vehicle_name=vehicle_name,
        rsf_list=rsf_list,
        **kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_test_batch,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return dataloader

if __name__ == '__main__':

    import sys
    _project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(_project_root))

    from config_test import (
        TEST_DATA_ROOT,
        TEST_WIFO_FEATURES_ROOT,
        SCENE_RSF_CONFIG,
        TEST_VEHICLES,
    )

    print("=" * 60)
    print("Test Data Loader Validation")
    print("=" * 60)

    scene_name = 'Scene1'
    vehicle_name = 'Car1'
    rsf_list = SCENE_RSF_CONFIG[scene_name]

    print(f"\nTest Vehicle: {scene_name}/{vehicle_name}")
    print(f"RSFList: {rsf_list}")

    dataloader = create_test_dataloader(
        test_data_root=TEST_DATA_ROOT,
        wifo_features_root=TEST_WIFO_FEATURES_ROOT,
        scene_name=scene_name,
        vehicle_name=vehicle_name,
        rsf_list=rsf_list,
        batch_size=8,
    )

    print(f"\nDataset Size: {len(dataloader.dataset)} Frame")
    print(f"Batch Quantity: {len(dataloader)} Batch")

    print(f"\nBefore testing3batch:")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:
            break

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  GPSPosture: {batch['gps_pose'].shape}")
        print(f"  WiFoFeatures:")
        for rsf_name, features in batch['wifo_features'].items():
            print(f"    {rsf_name}: {features.shape}")
        print(f"  RSFExternal Ginseng:")
        for rsf_name, extrinsic in batch['rsf_extrinsics'].items():
            print(f"    {rsf_name}: {extrinsic.shape}")
        print(f"  FrameIDScope: {[m['frame_id'] for m in batch['meta']]}")

    print("\n" + "=" * 60)
    print("✓ Data loader test passed! ")
    print("=" * 60)
