"""组装训练用的多路侧特征、GPS、真值和几何量。"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import sys
import re

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import apply_height_correction, convert_angles_deg_to_rad, is_aerial_vehicle
from models.geometry import extract_spatial_features_batch
from data.loaders.features import load_vehicle_wifo_features
from data.loaders.ground_truth import load_vehicle_gt
from data.loaders.gps import load_vehicle_gps
from data.loaders.rsu import load_scene_rsf_positions, rsf_dict_to_numpy
from config import WIFO_FEATURES_ROOT, DATA_ROOT, PROJECT_ROOT
import pickle

class Pose6DLoader:

    @staticmethod
    def load_rgb_features(
        rgb_features_dir: Path,
        scene_name: str,
        rsf_name: str,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:

        filename = f"{scene_name}_{rsf_name}_features.npz"
        filepath = rgb_features_dir / filename

        if not filepath.exists():
            return None

        try:
            data = np.load(filepath, allow_pickle=True)
            rgb_features = data['fused']

            total_frames = 1500 if end_idx is None else max(end_idx, 1500)
            rgb_mask = np.zeros(total_frames, dtype=np.float32)
            rgb_mask[:750] = 1.0

            rgb_features_full = np.zeros((total_frames, 1024), dtype=np.float32)
            rgb_features_full[:750] = rgb_features

            if end_idx is None:
                end_idx = total_frames
            rgb_features_window = rgb_features_full[start_idx:end_idx]
            rgb_mask_window = rgb_mask[start_idx:end_idx]

            window_has_rgb_start = max(0, 750 - start_idx)
            window_has_rgb_end = min(rgb_mask_window.shape[0], window_has_rgb_start)

            rgb_mask_window = np.zeros(rgb_mask_window.shape[0], dtype=np.float32)
            if start_idx < 750:
                valid_rgb_end = min(750 - start_idx, rgb_mask_window.shape[0])
                rgb_mask_window[:valid_rgb_end] = 1.0

            return rgb_features_window, rgb_mask_window

        except Exception as e:
            print(f"加载RGB特征失败 {filepath}: {e}")
            return None

    @staticmethod
    def load_pose_6d_from_gt(data_root, scene_name, vehicle_name):

        positions, orientations, frame_ids, metadata = load_vehicle_gt(
            data_root=data_root,
            scene_name=scene_name,
            vehicle_name=vehicle_name,
            apply_correction=True
        )

        if positions is None:
            return None

        pose_6d = np.concatenate([positions, orientations], axis=1)

        if frame_ids is None:
            frame_ids = np.arange(len(positions))

        return {
            'positions': positions,
            'orientations': orientations,
            'pose_6d': pose_6d,
            'frame_ids': frame_ids
        }

    @staticmethod
    def load_gps_6d_from_noise(data_root, scene_name, vehicle_name, config=None):

        positions_gps, orientations_gps, metadata = load_vehicle_gps(
            data_root=data_root,
            scene_name=scene_name,
            vehicle_name=vehicle_name
        )

        _, _, frame_ids, _ = load_vehicle_gt(
            data_root=data_root,
            scene_name=scene_name,
            vehicle_name=vehicle_name,
            apply_correction=False
        )

        if frame_ids is None:
            frame_ids = np.arange(len(positions_gps))

        pose_6d = np.concatenate([positions_gps, orientations_gps], axis=1)

        return {
            'positions': positions_gps,
            'orientations': orientations_gps,
            'pose_6d': pose_6d,
            'frame_ids': frame_ids
        }

    @staticmethod
    def load_rsf_extrinsics_6d(data_root, scene_name):

        rsf_dict = load_scene_rsf_positions(data_root, scene_name)

        if not rsf_dict:
            return {}

        rsf_extrinsics_6d = {}
        for rsf_name, rsf_info in rsf_dict.items():
            position = [rsf_info['X'], rsf_info['Y'], rsf_info['Z']]
            orientation = [
                np.deg2rad(rsf_info['Roll']),
                np.deg2rad(rsf_info['Pitch']),
                np.deg2rad(rsf_info['Yaw'])
            ]

            rsf_6d = np.array(position + orientation)
            rsf_extrinsics_6d[rsf_name] = rsf_6d

        return rsf_extrinsics_6d

class MultiRSFPose6DDataset(Dataset):

    def __init__(
        self,
        data_root: Union[str, Path],
        scenes: List[str],
        rsf_config: Dict[str, List[str]],
        split: str = 'train',
        time_window: Optional[int] = None,
        stride: int = 64,
        config=None,
        use_input_kalman: bool = False,
        input_kalman_config: Optional[Dict] = None
    ):

        self.data_root = Path(data_root)
        self.scenes = scenes
        self.rsf_config = rsf_config
        self.split = split
        self.time_window = time_window
        self.stride = stride
        self.config = config if config is not None else self._get_default_config()

        self.use_input_kalman = use_input_kalman
        self.input_kalman_config = input_kalman_config or {
            'process_noise': 0.10,
            'measurement_noise': 1.6,
        }

        if self.use_input_kalman:
            print("="*60)
            print("[Kalman] Vehicle-Type Adaptive Kalman Filtering Strategy v2.0")
            print("="*60)
            print("[OK] Input Kalman filtering (data loading stage):")
            print(f"   - Car: Apply Kalman filtering (process_noise={self.input_kalman_config.get('process_noise', 0.10)})")
            print(f"   - AT: Skip Kalman filtering (flexible trajectory, keep original GPS)")
            print(f"   - Measurement noise: {self.input_kalman_config.get('measurement_noise', 1.6)}")
            print("="*60)

        self.samples = []

        self.scene_rsf_counts = self._analyze_scene_rsf_counts()

        self._load_all_data()

    def _analyze_scene_rsf_counts(self):

        scene_counts = {}
        for scene in self.scenes:
            rsf_list = self.rsf_config.get(scene, [])
            scene_counts[scene] = len(rsf_list)

        print(f"场景RSF数量分析: {scene_counts}")
        return scene_counts

    def _load_all_data(self):

        for scene_name in self.scenes:
            print(f"\n{'='*60}")
            print(f"正在加载场景: {scene_name}")
            print(f"{'='*60}")

            scene_path = self.data_root / scene_name

            if not scene_path.exists():
                print(f"[ERROR] Scene path does not exist: {scene_path}")
                continue

            rsf_list = self.rsf_config.get(scene_name, [])
            if not rsf_list:
                print(f"[WARNING] Scene {scene_name} has no RSF configuration")
                continue

            print(f"[OK] RSF config: {rsf_list}")

            rsf_extrinsics_6d = Pose6DLoader.load_rsf_extrinsics_6d(self.data_root, scene_name)
            print(f"[OK] RSF extrinsics loaded: {list(rsf_extrinsics_6d.keys())}")

            vehicle_dirs = [d for d in scene_path.iterdir()
                          if d.is_dir() and (d.name.startswith('Car') or d.name.startswith('AT') or d.name.startswith('UAV'))]

            print(f"[OK] Found {len(vehicle_dirs)} vehicles: {sorted([d.name for d in vehicle_dirs])}")

            for vehicle_dir in vehicle_dirs:
                vehicle_name = vehicle_dir.name

                gt_data = Pose6DLoader.load_pose_6d_from_gt(self.data_root, scene_name, vehicle_name)

                gps_data = Pose6DLoader.load_gps_6d_from_noise(self.data_root, scene_name, vehicle_name, config=self.config)

                if gt_data is None or gps_data is None:
                    print(f"  [WARNING] Skip vehicle {vehicle_name}: GT or GPS loading failed")
                    continue

                print(f"  [OK] {vehicle_name}: GT and GPS data loaded")

                available_rsfs = []
                for rsf_name in rsf_list:
                    if rsf_name not in rsf_extrinsics_6d:
                        print(f"    [WARNING] Skip {rsf_name}: Missing extrinsics")
                        continue

                    wifo_feature_file = WIFO_FEATURES_ROOT / scene_name / vehicle_name / f"{vehicle_name}_{rsf_name}_wifo_features.npy"
                    if not wifo_feature_file.exists():
                        print(f"    [WARNING] Skip {rsf_name}: WiFo feature file not found")
                        continue

                    available_rsfs.append(rsf_name)

                if not available_rsfs:
                    print(f"    [WARNING] Skip vehicle: No available RSF features")
                    continue

                print(f"    [OK] Loaded {len(available_rsfs)}/{len(rsf_list)} RSFs: {available_rsfs}")

                sample = {
                    'scene_name': scene_name,
                    'vehicle_name': vehicle_name,
                    'rsf_list': available_rsfs,
                    'gt_data': gt_data,
                    'gps_data': gps_data,
                    'rsf_extrinsics_6d': {rsf: rsf_extrinsics_6d[rsf] for rsf in available_rsfs},
                }

                if self.time_window is None:
                    self.samples.append(sample)
                else:
                    trajectory_length = len(gt_data['pose_6d'])
                    for start_idx in range(0, trajectory_length - self.time_window + 1, self.stride):
                        end_idx = start_idx + self.time_window

                        windowed_sample = sample.copy()
                        windowed_sample.update({
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'window_length': self.time_window
                        })

                        self.samples.append(windowed_sample)

        sample_stats = {}
        rsf_stats = {}
        scene_stats = {}

        for sample in self.samples:
            scene = sample['scene_name']
            vehicle = sample['vehicle_name']
            key = f"{scene}/{vehicle}"

            if key not in sample_stats:
                sample_stats[key] = 0
                rsf_stats[key] = sample['rsf_list']
            sample_stats[key] += 1

            if scene not in scene_stats:
                scene_stats[scene] = set()
            scene_stats[scene].add(vehicle)

        print(f"\n{'='*60}")
        print(f"成功加载 {len(self.samples)} 个样本")
        print(f"{'='*60}")

        for scene in sorted(scene_stats.keys()):
            vehicles = sorted(scene_stats[scene])
            print(f"\n【{scene}】: {len(vehicles)}辆车")
            print(f"  车辆列表: {vehicles}")

            expected_rsf = len(self.rsf_config.get(scene, []))
            print(f"  期望RSF数: {expected_rsf}")

            for vehicle in vehicles:
                key = f"{scene}/{vehicle}"
                rsf_list = rsf_stats[key]
                num_samples = sample_stats[key]
                print(f"    {vehicle}: {num_samples}个样本, {len(rsf_list)}个RSF {sorted(rsf_list)}")

        print(f"\n{'='*60}")
        print(f"总样本数: {len(self.samples)}")
        print(f"说明：完整轨迹模式下，每辆车=1个样本（包含所有RSF数据）")
        print(f"期望样本数: Scene1(6车) + Scene2(6车) + Scene3(10车) = 22辆车")
        print(f"{'='*60}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        try:
            scene_name = sample['scene_name']
            vehicle_name = sample['vehicle_name']
            rsf_list = sample['rsf_list']
            n_rsf = len(rsf_list)

            all_csi_features = []
            all_spatial_features = []
            all_rgb_features = []
            all_rgb_masks = []

            try:
                first_rsf = rsf_list[0]
                wifo_data_temp = load_vehicle_wifo_features(
                    features_root=WIFO_FEATURES_ROOT,
                    scene_name=scene_name,
                    vehicle_name=vehicle_name,
                    rsf_list=[first_rsf],
                    load_gps=False,
                    load_frame_info=True
                )
                frame_info = wifo_data_temp.get('frame_info', {})
                start_frame = frame_info.get('start_frame', 0)
                end_frame = frame_info.get('end_frame', start_frame)
            except:
                start_frame = 0
                end_frame = 0

            start_idx = sample.get('start_idx', 0)
            end_idx_orig = sample.get('end_idx', None)

            if end_idx_orig is None:
                actual_traj_length = len(sample['gt_data']['pose_6d'])
                rgb_start_frame = start_frame
                rgb_end_frame = start_frame + actual_traj_length
            else:
                rgb_start_frame = start_frame + start_idx
                rgb_end_frame = start_frame + end_idx_orig

            for idx, rsf_name in enumerate(rsf_list):
                try:
                    wifo_data = load_vehicle_wifo_features(
                        features_root=WIFO_FEATURES_ROOT,
                        scene_name=scene_name,
                        vehicle_name=vehicle_name,
                        rsf_list=[rsf_name],
                        load_gps=False,
                        load_frame_info=False
                    )

                    if 'start_idx' in sample:
                        csi_features = torch.from_numpy(
                            wifo_data['wifo_features'][0][start_idx:end_idx_orig]
                        ).float()
                    else:
                        csi_features = torch.from_numpy(
                            wifo_data['wifo_features'][0]
                        ).float()

                except Exception as e:
                    print(f"警告：WiFo特征加载失败 {scene_name}/{vehicle_name}/{rsf_name}: {e}")
                    if 'start_idx' in sample:
                        T = sample['window_length']
                    else:
                        T = len(sample['gt_data']['pose_6d'])
                    csi_features = torch.randn(T, 512)

                all_csi_features.append(csi_features)

                T_rgb = csi_features.shape[0]
                all_rgb_features.append(torch.zeros(T_rgb, 1024))
                all_rgb_masks.append(torch.zeros(T_rgb))

            all_csi_features = torch.stack(all_csi_features, dim=0)
            T = all_csi_features.shape[1]

            if len(all_rgb_features) > 0:
                all_rgb_features = torch.stack(all_rgb_features, dim=0)
                all_rgb_masks = torch.stack(all_rgb_masks, dim=0)
            else:
                all_rgb_features = None
                all_rgb_masks = None

            if 'start_idx' in sample:
                gt_poses_6d = torch.from_numpy(sample['gt_data']['pose_6d'][start_idx:end_idx_orig]).float()
                gps_poses_6d = torch.from_numpy(sample['gps_data']['pose_6d'][start_idx:end_idx_orig]).float()
            else:
                gt_poses_6d = torch.from_numpy(sample['gt_data']['pose_6d']).float()
                gps_poses_6d = torch.from_numpy(sample['gps_data']['pose_6d']).float()

            if self.use_input_kalman:
                gps_poses_6d = self._apply_input_kalman(gps_poses_6d, scene_name, vehicle_name)

            gps_trajectory_6d = gps_poses_6d.unsqueeze(0)

            all_rsf_extrinsics_6d = []
            for rsf_name in rsf_list:
                rsf_6d = sample['rsf_extrinsics_6d'][rsf_name]
                all_rsf_extrinsics_6d.append(torch.from_numpy(rsf_6d).float())
            all_rsf_extrinsics_6d = torch.stack(all_rsf_extrinsics_6d, dim=0)

            vehicle_type = 'AT' if 'UAV' in vehicle_name or 'AT' in vehicle_name else 'Car'

            spatial_features, _ = extract_spatial_features_batch(
                gps_trajectory_6d,
                all_rsf_extrinsics_6d.unsqueeze(0),
                self.config,
                scene_names=[scene_name]
            )

            spatial_features = spatial_features.squeeze(0)

            gt_frame_ids = sample['gt_data'].get('frame_ids', None)
            if gt_frame_ids is not None:
                if 'start_idx' in sample and sample.get('end_idx') is not None:
                    frame_ids = torch.from_numpy(gt_frame_ids[start_idx:end_idx_orig]).long()
                else:
                    frame_ids = torch.from_numpy(gt_frame_ids).long()
            else:
                base = int(start_frame if start_frame is not None else 0) + int(sample.get('start_idx', 0) or 0)
                frame_ids = torch.arange(T, dtype=torch.long) + base

            modality = str(getattr(self.config, "ablation_modality", "full") or "full").lower()
            if modality in ("no_csi_completion", "no_completion"):
                from utils.completeness import zero_csi_where_invalid, default_completeness_path
                comp_path = getattr(self.config, "completeness_json", None)
                comp_path = Path(comp_path) if comp_path else default_completeness_path()
                zeroed_total = 0
                for i_rsf, rsf_name in enumerate(rsf_list):
                    feat_i, n_z = zero_csi_where_invalid(
                        all_csi_features[i_rsf],
                        frame_ids,
                        scene_name,
                        vehicle_name,
                        rsf_name,
                        path=comp_path,
                    )
                    all_csi_features[i_rsf] = feat_i
                    zeroed_total += n_z
                if not hasattr(self, "_csi_mask_log_count"):
                    self._csi_mask_log_count = 0
                if self._csi_mask_log_count < 8:
                    print(
                        f"[no_csi_completion] {scene_name}/{vehicle_name}: "
                        f"zeroed {zeroed_total} frame-RSU slots (path={comp_path.name})"
                    )
                    self._csi_mask_log_count += 1

            return {
                'scene_name': scene_name,
                'vehicle_name': vehicle_name,
                'vehicle_type': vehicle_type,
                'rsf_list': rsf_list,
                'n_rsf': n_rsf,
                'csi_features': all_csi_features,
                'spatial_features': spatial_features,
                'rgb_features': all_rgb_features,
                'rgb_mask': all_rgb_masks,
                'gt_poses_6d': gt_poses_6d,
                'gps_poses_6d': gps_poses_6d,
                'frame_ids': frame_ids,
                'trajectory_length': T,
                'start_idx': sample.get('start_idx', 0),
                'end_idx': sample.get('end_idx', T),
                'start_frame': start_frame if start_frame is not None else 0,
                'end_frame': end_frame if end_frame is not None else (T - 1),
            }

        except Exception as e:
            print(f"加载样本失败 {idx}: {e}")
            return None

    def _apply_input_kalman(self, gps_poses_6d: torch.Tensor, scene_name: str, vehicle_name: str) -> torch.Tensor:

        is_tensor = isinstance(gps_poses_6d, torch.Tensor)
        if is_tensor:
            gps_np = gps_poses_6d.cpu().numpy()
        else:
            gps_np = gps_poses_6d

        if vehicle_name.startswith('AT'):
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

                        smoothed_np = apply_ukf_to_gps(
                            gps_poses_6d=gps_np,
                            process_noise=process_noise,
                            measurement_noise=measurement_noise,
                            dt=dt,
                            filter_orientation=True,
                            process_noise_ori=process_noise_ori,
                            measurement_noise_ori=measurement_noise_ori
                        )
                        print(f"  [UKF] AT车辆 {vehicle_name}: 6D UKF滤波完成（xyz+姿态）")
                    else:
                        positions = gps_np[:, :3]
                        orientations = gps_np[:, 3:6]

                        positions_filtered = apply_ukf_to_gps(
                            gps_positions=positions,
                            process_noise=process_noise,
                            measurement_noise=measurement_noise,
                            dt=dt
                        )

                        smoothed_np = np.column_stack([positions_filtered, orientations])
                        print(f"  [UKF] AT车辆 {vehicle_name}: 3D UKF滤波完成（xyz）")

                except Exception as e:
                    print(f"  [UKF] [WARNING] AT车辆 {vehicle_name} UKF滤波失败: {e}，使用原始GPS")
                    smoothed_np = gps_np
            else:
                smoothed_np = gps_np
                print(f"  [Info] AT车辆 {vehicle_name}: 未启用UKF，使用原始GPS")

        else:
            from utils.kalman import apply_kalman_smoothing

            process_noise = self.input_kalman_config.get('process_noise', 0.10)
            measurement_noise = self.input_kalman_config.get('measurement_noise', 1.6)

            smoothed_np = apply_kalman_smoothing(
                gps_np,
                vehicle_name=vehicle_name,
                process_noise=process_noise,
                measurement_noise=measurement_noise
            )

            print(f"  [Kalman] Car车辆 {vehicle_name}: 2D卡尔曼滤波完成（XY降噪）")

        if not vehicle_name.startswith('AT'):
            if hasattr(self, 'config') and getattr(self.config, 'use_orientation_noise', False):
                from data.loaders.gps import add_gps_orientation_noise

                orientations_noisy = add_gps_orientation_noise(
                    smoothed_np[:, 3:6],
                    vehicle_name,
                    config=self.config
                )
                smoothed_np[:, 3:6] = orientations_noisy
                if getattr(self.config, 'verbose', False):
                    from tqdm import tqdm
                    tqdm.write(f"  [Noise] Car车辆 {vehicle_name}: 姿态角噪声添加完成")

        if is_tensor:
            smoothed_gps = torch.from_numpy(smoothed_np).float()
        else:
            smoothed_gps = smoothed_np

        return smoothed_gps

class MultiRSFPose6DDataLoader:

    def __init__(
        self,
        data_root: Union[str, Path],
        scenes: List[str],
        rsf_config: Dict[str, List[str]],
        split: str = 'train',
        batch_size: int = 1,
        time_window: Optional[int] = None,
        stride: int = 64,
        num_workers: int = 0,
        config=None,
        use_input_kalman: bool = False,
        input_kalman_config: Optional[Dict] = None
    ):

        self.dataset = MultiRSFPose6DDataset(
            data_root=data_root,
            scenes=scenes,
            rsf_config=rsf_config,
            split=split,
            time_window=time_window,
            stride=stride,
            config=config,
            use_input_kalman=use_input_kalman,
            input_kalman_config=input_kalman_config
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            drop_last=False
        )

    def _collate_fn(self, batch):

        batch = [sample for sample in batch if sample is not None]

        if len(batch) == 0:
            return None

        if len(batch) > 1:
            print(f"警告：batch_size={len(batch)} > 1，轨迹长度可能不同，建议设置batch_size=1")

        sample = batch[0]

        scene_name = sample['scene_name']
        rsf_list = sample['rsf_list']
        n_rsf = sample['n_rsf']

        rsf_data_dict = {}

        for idx, rsf_name in enumerate(rsf_list):
            rsf_data = {
                'csi_features': sample['csi_features'][idx].unsqueeze(0),
                'spatial_features': sample['spatial_features'][:, idx, :].unsqueeze(0),
            }

            if sample['rgb_features'] is not None:
                rsf_data['rgb_features'] = sample['rgb_features'][idx].unsqueeze(0)
                rsf_data['rgb_mask'] = sample['rgb_mask'][idx].unsqueeze(0)

            rsf_data_dict[rsf_name] = rsf_data

        collated_batch = {
            'scene_name': scene_name,
            'rsf_data_dict': rsf_data_dict,
            'rsf_list': rsf_list,
            'n_rsf': n_rsf,

            'gt_poses_6d': sample['gt_poses_6d'].unsqueeze(0),
            'gps_poses_6d': sample['gps_poses_6d'].unsqueeze(0),
            'frame_ids': sample['frame_ids'].unsqueeze(0) if 'frame_ids' in sample else None,

            'vehicle_names': [sample['vehicle_name']],
            'vehicle_types': [sample['vehicle_type']],
            'trajectory_lengths': [sample['trajectory_length']],

            'start_frame': sample['start_frame'],
            'end_frame': sample['end_frame'],
        }

        return collated_batch

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

def create_train_set(config):

    return MultiRSFPose6DDataLoader(
        data_root=config.data_root,
        scenes=config.scenes,
        rsf_config=config.rsf_config,
        split='train',
        batch_size=config.batch_size,
        time_window=config.time_window,
        stride=config.stride,
        num_workers=config.num_workers,
        config=config
    )
