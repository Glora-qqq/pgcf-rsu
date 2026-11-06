"""Sliding Window Dataset on Test Set. """

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .csi import load_test_csi, TEST_SCENE_CSI_SUFFIX
from .mask_io import load_test_mask
from .rgb_io import load_test_rgb_features

class TestSlidingWindowDataset(Dataset):

    def __init__(
        self,
        data_root: Path,
        mask_root: Path,
        rgb_root: Path,
        scenes: List[str],
        rsf_config: Dict[str, List[str]],
        scene_name_mapping: Dict[str, str],
        window_size: int = 60,
        stride: int = 5,
        min_valid_frames: int = 12,
        rgb_strategy: str = 'center',
        **kwargs
    ):

        self.data_root = Path(data_root)
        self.mask_root = Path(mask_root)
        self.rgb_root = Path(rgb_root)
        self.window_size = window_size
        self.stride = stride
        self.min_valid_frames = min_valid_frames
        self.rgb_strategy = rgb_strategy

        self.windows = []
        self._scan_samples(scenes, rsf_config, scene_name_mapping)

        print(f"[TestDataset] Scan Complete:")
        print(f"  Number of scenes: {len(scenes)}")
        print(f"  Total number of windows: {len(self.windows)}")

    def _scan_samples(
        self,
        scenes: List[str],
        rsf_config: Dict[str, List[str]],
        scene_name_mapping: Dict[str, str]
    ):

        for test_scene in scenes:
            train_scene = scene_name_mapping.get(test_scene, test_scene.replace('_', ''))

            if train_scene not in rsf_config:
                print(f"[WARNING] Scene {train_scene} Not inRSFConfiguring, Skip")
                continue

            rsf_list = rsf_config[train_scene]

            for train_rsf in rsf_list:
                test_rsf = train_rsf

                rsf_dir = self.data_root / test_scene / test_rsf / 'CSI'
                if not rsf_dir.exists():
                    continue

                csi_suffix = TEST_SCENE_CSI_SUFFIX.get(test_scene, 'high_fencha')

                for csi_file in rsf_dir.glob(f'{test_rsf}_*_{csi_suffix}_CSI.mat'):
                    parts = csi_file.stem.split('_')
                    vehicle = parts[1]

                    self._generate_windows(
                        test_scene=test_scene,
                        test_rsf=test_rsf,
                        train_scene=train_scene,
                        train_rsf=train_rsf,
                        vehicle=vehicle,
                    )

    def _generate_windows(
        self,
        test_scene: str,
        test_rsf: str,
        train_scene: str,
        train_rsf: str,
        vehicle: str,
    ):

        mask = load_test_mask(
            mask_root=self.mask_root,
            scene=train_scene,
            rsf=train_rsf,
            vehicle=vehicle,
        )

        num_frames = 1500
        max_start = num_frames - self.window_size + 1

        for start_idx in range(0, max_start, self.stride):
            end_idx = start_idx + self.window_size

            window_mask = mask[start_idx:end_idx]
            num_valid = np.sum(window_mask == 0)

            if num_valid >= self.min_valid_frames:
                self.windows.append({
                    'test_scene': test_scene,
                    'test_rsf': test_rsf,
                    'train_scene': train_scene,
                    'train_rsf': train_rsf,
                    'vehicle': vehicle,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'num_valid': int(num_valid),
                })

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict:

        window_info = self.windows[idx]

        test_scene = window_info['test_scene']
        test_rsf = window_info['test_rsf']
        train_scene = window_info['train_scene']
        train_rsf = window_info['train_rsf']
        vehicle = window_info['vehicle']
        start_idx = window_info['start_idx']
        end_idx = window_info['end_idx']

        csi_full = load_test_csi(
            data_root=self.data_root,
            scene=test_scene,
            rsf=test_rsf,
            vehicle=vehicle,
        )

        if csi_full is None:
            csi_window = np.zeros((2, self.window_size, 128, 32), dtype=np.float32)
        else:
            csi_window = csi_full[:, start_idx:end_idx, :, :]

        mask_full = load_test_mask(
            mask_root=self.mask_root,
            scene=train_scene,
            rsf=train_rsf,
            vehicle=vehicle,
        )
        mask_window = mask_full[start_idx:end_idx]

        rgb_full = load_test_rgb_features(
            rgb_root=self.rgb_root,
            scene=train_scene,
            rsf=train_rsf,
            padding_strategy='mean',
        )

        if rgb_full is None:
            rgb_global = np.zeros(1024, dtype=np.float32)
        else:
            rgb_window = rgb_full[start_idx:end_idx, :]

            if self.rgb_strategy == 'center':
                center_idx = self.window_size // 2
                rgb_global = rgb_window[center_idx, :]
            elif self.rgb_strategy == 'mean':
                valid_mask = (mask_window == 0)
                if valid_mask.sum() > 0:
                    rgb_global = rgb_window[valid_mask].mean(axis=0)
                else:
                    rgb_global = rgb_window.mean(axis=0)
            else:
                raise ValueError(f"UnknownRGBStrategy: {self.rgb_strategy}")

        sample = {
            'csi': torch.from_numpy(csi_window),
            'mask': torch.from_numpy(mask_window),
            'rgb_global': torch.from_numpy(rgb_global),
            'window_info': {
                'scene': train_scene,
                'rsf': train_rsf,
                'vehicle': vehicle,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'num_valid': window_info['num_valid'],
            }
        }

        return sample

def create_test_dataloader(
    data_root: Path,
    mask_root: Path,
    rgb_root: Path,
    scenes: List[str],
    rsf_config: Dict[str, List[str]],
    scene_name_mapping: Dict[str, str],
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:

    dataset = TestSlidingWindowDataset(
        data_root=data_root,
        mask_root=mask_root,
        rgb_root=rgb_root,
        scenes=scenes,
        rsf_config=rsf_config,
        scene_name_mapping=scene_name_mapping,
        **dataset_kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader

def verify_test_dataset():

    from config_test import (
        TEST_DATA_ROOT, TEST_MASK_ROOT, TEST_RGB_FEATURE_ROOT,
        TEST_SCENES, TEST_RSF_CONFIG, SCENE_NAME_MAPPING,
        TEST_WINDOW_CONFIG
    )

    print("=" * 60)
    print("Validate Test Dataset")
    print("=" * 60)

    dataset = TestSlidingWindowDataset(
        data_root=TEST_DATA_ROOT,
        mask_root=TEST_MASK_ROOT,
        rgb_root=TEST_RGB_FEATURE_ROOT,
        scenes=TEST_SCENES,
        rsf_config=TEST_RSF_CONFIG,
        scene_name_mapping=SCENE_NAME_MAPPING,
        **TEST_WINDOW_CONFIG
    )

    print(f"\n[Dataset Statistics]")
    print(f"  Total number of windows: {len(dataset)}")

    if len(dataset) > 0:
        print(f"\n[Test the first sample]")
        sample = dataset[0]

        print(f"  CSIShape: {sample['csi'].shape}")
        print(f"  MaskShape: {sample['mask'].shape}")
        print(f"  RGBShape: {sample['rgb_global'].shape}")
        print(f"  Window Information: {sample['window_info']}")
        print(f"  Effective Frames: {(sample['mask'] == 0).sum().item()}")

        print(f"\n[TestDataLoader]")
        dataloader = create_test_dataloader(
            data_root=TEST_DATA_ROOT,
            mask_root=TEST_MASK_ROOT,
            rgb_root=TEST_RGB_FEATURE_ROOT,
            scenes=TEST_SCENES,
            rsf_config=TEST_RSF_CONFIG,
            scene_name_mapping=SCENE_NAME_MAPPING,
            batch_size=4,
            num_workers=0,
            **TEST_WINDOW_CONFIG
        )

        batch = next(iter(dataloader))
        print(f"  Batch CSIShape: {batch['csi'].shape}")
        print(f"  Batch MaskShape: {batch['mask'].shape}")
        print(f"  Batch RGBShape: {batch['rgb_global'].shape}")
        print(f"  ✓ DataLoaderWorks fine")

    print("=" * 60)

if __name__ == '__main__':
    verify_test_dataset()
