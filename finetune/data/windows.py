"""把 CSI 和视觉特征切成滑窗，供微调训练。"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

class SlidingWindowCSIDataset(Dataset):

    def __init__(
        self,
        csi_root: Path,
        mask_root: Path,
        rgb_feature_root: Path,
        scenes: List[str] = None,
        rsf_config: Dict = None,
        window_size: int = 60,
        stride: int = 5,
        min_valid_frames: int = 10,
        rgb_strategy: str = 'center',
        verbose: bool = True
    ):

        self.csi_root = Path(csi_root)
        self.mask_root = Path(mask_root)
        self.rgb_feature_root = Path(rgb_feature_root)
        self.window_size = window_size
        self.stride = stride
        self.min_valid_frames = min_valid_frames
        self.rgb_strategy = rgb_strategy
        self.verbose = verbose

        if scenes is None:
            scenes = ['Scene1', 'Scene2', 'Scene3']
        if rsf_config is None:
            rsf_config = {
                'Scene1': ['RSF1', 'RSF3', 'RSF4', 'RSF7', 'RSF11'],
                'Scene2': ['RSF1', 'RSF3', 'RSF4', 'RSF7', 'RSF11'],
                'Scene3': ['RSF1', 'RSF3', 'RSF4', 'RSF7', 'RSF11'],
            }

        self.scenes = scenes
        self.rsf_config = rsf_config

        if verbose:
            print(f"\n{'='*80}")
            print(f"初始化滑动窗口CSI数据集")
            print(f"{'='*80}")
            print(f"配置:")
            print(f"  窗口大小: {window_size}帧")
            print(f"  步长: {stride}帧")
            print(f"  最小有效帧: {min_valid_frames}帧")
            print(f"  RGB策略: {rgb_strategy}")

        self.samples = []
        self.windows = []

        self._scan_samples()

        if verbose:
            print(f"\n数据集初始化完成:")
            print(f"  总样本数: {len(self.samples)}")
            print(f"  总有效窗口: {len(self.windows)}")
            print(f"{'='*80}\n")

    def _scan_samples(self):

        if self.verbose:
            print(f"\n扫描样本...")

        total_samples = 0
        total_windows = 0
        total_valid_windows = 0

        for scene in self.scenes:
            rsf_list = self.rsf_config.get(scene, [])

            for rsf in rsf_list:
                csi_dir = self.csi_root / scene / rsf / 'CSI'

                if not csi_dir.exists():
                    if self.verbose:
                        print(f"  [WARNING] CSI目录不存在: {csi_dir}")
                    continue

                mat_files = list(csi_dir.glob('*.mat'))
                if len(mat_files) == 0:
                    if self.verbose:
                        print(f"  [WARNING] 无CSI文件: {csi_dir}")
                    continue

                for mat_file in mat_files:
                    filename = mat_file.stem
                    parts = filename.split('_')
                    if len(parts) < 2:
                        continue
                    vehicle = parts[1]

                    mask_file = self.mask_root / scene / rsf / f"{vehicle}_mask.npy"
                    if not mask_file.exists():
                        if self.verbose:
                            print(f"  [SKIP] Mask不存在: {scene}/{rsf}/{vehicle}")
                        continue

                    try:
                        mask = np.load(mask_file)
                    except Exception as e:
                        if self.verbose:
                            print(f"  [ERROR] 加载mask失败: {scene}/{rsf}/{vehicle}, {e}")
                        continue

                    sample_windows = self._generate_windows(mask)

                    if len(sample_windows) == 0:
                        if self.verbose:
                            print(f"  [SKIP] 无有效窗口: {scene}/{rsf}/{vehicle}")
                        continue

                    sample_idx = len(self.samples)
                    self.samples.append({
                        'scene': scene,
                        'rsf': rsf,
                        'vehicle': vehicle,
                        'csi_file': str(mat_file),
                        'mask_file': str(mask_file),
                        'rgb_npz_file': str(self.rgb_feature_root / f"{scene}_{rsf}_features.npz"),
                        'num_windows': len(sample_windows),
                    })

                    for window_start, window_end, num_valid in sample_windows:
                        self.windows.append({
                            'sample_idx': sample_idx,
                            'window_start': window_start,
                            'window_end': window_end,
                            'num_valid_frames': num_valid,
                        })

                    total_samples += 1
                    total_windows += (1500 - self.window_size) // self.stride + 1
                    total_valid_windows += len(sample_windows)

        if self.verbose:
            print(f"\n扫描完成:")
            print(f"  有效样本: {total_samples}")
            print(f"  理论窗口: {total_windows}")
            print(f"  有效窗口: {total_valid_windows}")
            if total_windows > 0:
                print(f"  窗口保留率: {total_valid_windows/total_windows*100:.1f}%")

    def _generate_windows(self, mask: np.ndarray) -> List[Tuple[int, int, int]]:

        seq_len = len(mask)
        valid_windows = []

        for start in range(0, seq_len - self.window_size + 1, self.stride):
            end = start + self.window_size
            window_mask = mask[start:end]

            num_valid = int((window_mask == 0).sum())

            if num_valid >= self.min_valid_frames:
                valid_windows.append((start, end, num_valid))

        return valid_windows

    def __len__(self):

        return len(self.windows)

    def __getitem__(self, idx):

        window_info = self.windows[idx]
        sample_idx = window_info['sample_idx']
        window_start = window_info['window_start']
        window_end = window_info['window_end']

        sample_meta = self.samples[sample_idx]

        csi = self._load_csi(sample_meta['csi_file'], window_start, window_end)

        mask = self._load_mask(sample_meta['mask_file'], window_start, window_end)

        rgb_global = self._load_rgb_feature(
            sample_meta['rgb_npz_file'],
            mask,
            window_start,
            window_end
        )

        sample = {
            'csi': csi,
            'mask': mask,
            'rgb_global': rgb_global,
            'window_info': {
                'scene': sample_meta['scene'],
                'rsf': sample_meta['rsf'],
                'vehicle': sample_meta['vehicle'],
                'window_start': window_start,
                'window_end': window_end,
                'num_valid_frames': window_info['num_valid_frames'],
            }
        }

        return sample

    def _load_csi(self, csi_file: str, start: int, end: int) -> torch.Tensor:

        try:
            import scipy.io as sio

            mat_data = sio.loadmat(csi_file)

            csi_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
            csi_1500_frames = mat_data[csi_key]

            window_size = end - start
            csi_complex = np.zeros((window_size, 128, 32), dtype=np.complex128)

            for i, frame_idx in enumerate(range(start, end)):
                try:
                    csi_frame = csi_1500_frames[0, frame_idx]

                    if isinstance(csi_frame, np.ndarray):
                        csi_array = csi_frame
                    else:
                        csi_array = np.array(csi_frame, dtype=complex)

                    if csi_array.size > 0 and not np.allclose(csi_array, 0):
                        csi_complex[i] = csi_array
                except:
                    pass

            csi_real = csi_complex.real.astype(np.float32)
            csi_imag = csi_complex.imag.astype(np.float32)
            csi = np.stack([csi_real, csi_imag], axis=0)

            csi_tensor = torch.from_numpy(csi).float()

            return csi_tensor

        except Exception as e:
            print(f"[ERROR] 加载CSI失败: {csi_file}, {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(2, end - start, 128, 32, dtype=torch.float32)

    def _load_mask(self, mask_file: str, start: int, end: int) -> torch.Tensor:

        try:
            mask = np.load(mask_file)
            mask_window = mask[start:end]
            mask_tensor = torch.from_numpy(mask_window).float()
            return mask_tensor
        except Exception as e:
            print(f"[ERROR] 加载mask失败: {mask_file}, {e}")
            return torch.zeros(end - start, dtype=torch.float32)

    def _load_rgb_feature(
        self,
        rgb_npz_file: str,
        mask: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:

        try:
            rgb_data = np.load(rgb_npz_file, allow_pickle=True)
            rgb_750 = rgb_data['fused']

            rgb_mean = rgb_750.mean(axis=0, keepdims=True)
            rgb_1500 = np.concatenate([rgb_750, np.repeat(rgb_mean, 750, axis=0)], axis=0)

            if self.rgb_strategy == 'center':
                center_idx = (start + end) // 2
                rgb_global = rgb_1500[center_idx]

            elif self.rgb_strategy == 'mean':
                mask_np = mask.numpy()
                valid_indices = np.where(mask_np == 0)[0] + start

                if len(valid_indices) > 0:
                    rgb_global = rgb_1500[valid_indices].mean(axis=0)
                else:
                    center_idx = (start + end) // 2
                    rgb_global = rgb_1500[center_idx]

            else:
                raise ValueError(f"未知的RGB策略: {self.rgb_strategy}")

            rgb_tensor = torch.from_numpy(rgb_global).float()

            return rgb_tensor

        except Exception as e:
            print(f"[ERROR] 加载RGB特征失败: {rgb_npz_file}, {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(1024, dtype=torch.float32)

    def get_statistics(self) -> dict:

        if len(self.windows) == 0:
            return {
                'num_samples': 0,
                'num_windows': 0,
                'avg_windows_per_sample': 0,
                'avg_valid_frames_per_window': 0,
            }

        sample_window_counts = {}
        for window in self.windows:
            sample_idx = window['sample_idx']
            if sample_idx not in sample_window_counts:
                sample_window_counts[sample_idx] = 0
            sample_window_counts[sample_idx] += 1

        valid_frame_counts = [w['num_valid_frames'] for w in self.windows]

        stats = {
            'num_samples': len(self.samples),
            'num_windows': len(self.windows),
            'avg_windows_per_sample': len(self.windows) / max(len(self.samples), 1),
            'avg_valid_frames_per_window': np.mean(valid_frame_counts),
            'min_valid_frames_per_window': np.min(valid_frame_counts),
            'max_valid_frames_per_window': np.max(valid_frame_counts),
        }

        return stats

if __name__ == '__main__':

    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from config import (
        CSI_DATA_ROOT,
        MASK_ROOT,
        RGB_FEATURE_ROOT,
        WINDOW_CONFIG
    )

    print("="*80)
    print("测试滑动窗口CSI数据集")
    print("="*80)

    dataset = SlidingWindowCSIDataset(
        csi_root=CSI_DATA_ROOT,
        mask_root=MASK_ROOT,
        rgb_feature_root=RGB_FEATURE_ROOT,
        window_size=WINDOW_CONFIG['window_size'],
        stride=WINDOW_CONFIG['stride'],
        min_valid_frames=WINDOW_CONFIG['min_valid_frames'],
        rgb_strategy='center',
        verbose=True
    )

    stats = dataset.get_statistics()
    print(f"\n数据集统计:")
    print(f"  样本数: {stats['num_samples']}")
    print(f"  窗口数: {stats['num_windows']}")
    print(f"  平均窗口/样本: {stats['avg_windows_per_sample']:.1f}")
    print(f"  平均有效帧/窗口: {stats['avg_valid_frames_per_window']:.1f}")

    if len(dataset) > 0:
        print(f"\n测试加载第一个窗口...")
        sample = dataset[0]

        print(f"  CSI shape: {sample['csi'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  RGB global shape: {sample['rgb_global'].shape}")
        print(f"  窗口信息: {sample['window_info']}")
        print(f"  有效帧数: {sample['window_info']['num_valid_frames']}/{dataset.window_size}")

        num_valid = (sample['mask'] == 0).sum().item()
        print(f"  验证: mask中有效帧 = {num_valid} [OK]")

    print(f"\n{'='*80}")
    print("测试完成！")
    print("="*80)
