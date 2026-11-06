"""读取测试集 CSI 有效帧掩码。"""

import numpy as np
from pathlib import Path
from typing import Optional

def load_test_mask(
    mask_root: Path,
    scene: str,
    rsf: str,
    vehicle: str,
    num_frames_original: int = 750,
    num_frames_padded: int = 1500,
    force_mask_start: int = 750,
) -> np.ndarray:

    VEHICLE_NAME_MAPPING = {
        'UAV1': 'AT1', 'UAV2': 'AT2', 'UAV3': 'AT3',
        'UAV4': 'AT4', 'UAV5': 'AT5', 'UAV6': 'AT6',
    }
    reverse_mapping = {v: k for k, v in VEHICLE_NAME_MAPPING.items()}
    vehicle_mask = reverse_mapping.get(vehicle, vehicle)

    mask_file = mask_root / scene / rsf / f'{vehicle_mask}_mask.npy'

    if mask_file.exists():
        try:
            mask = np.load(mask_file).astype(np.float32)

            if len(mask) == num_frames_original:
                mask_extended = np.ones(num_frames_padded, dtype=np.float32)
                mask_extended[:num_frames_original] = mask
            elif len(mask) == num_frames_padded:
                mask_extended = mask.copy()
            else:
                print(f"[WARNING] Mask长度异常: {len(mask)}, 使用默认mask")
                mask_extended = np.zeros(num_frames_padded, dtype=np.float32)

            mask_extended[force_mask_start:] = 1.0

            return mask_extended

        except Exception as e:
            print(f"[WARNING] Mask加载失败 {mask_file}: {e}, 使用默认mask")

    mask = np.zeros(num_frames_padded, dtype=np.float32)
    mask[force_mask_start:] = 1.0

    return mask

def verify_test_mask_loading():

    from config_test import TEST_MASK_ROOT

    print("=" * 60)
    print("验证测试Mask加载")
    print("=" * 60)

    test_scene = 'Scene1'
    test_rsf = 'RSF1'
    test_vehicle = 'Car1'

    mask = load_test_mask(
        mask_root=TEST_MASK_ROOT,
        scene=test_scene,
        rsf=test_rsf,
        vehicle=test_vehicle,
    )

    print(f"✓ Mask加载完成")
    print(f"  形状: {mask.shape}")
    print(f"  数据类型: {mask.dtype}")
    print(f"  前750帧:")
    print(f"    有效帧数: {np.sum(mask[:750] == 0)}")
    print(f"    缺失帧数: {np.sum(mask[:750] == 1)}")
    print(f"  后750帧:")
    print(f"    有效帧数: {np.sum(mask[750:] == 0)}")
    print(f"    缺失帧数: {np.sum(mask[750:] == 1)}")
    print(f"  后750帧全1检查: {np.all(mask[750:] == 1)}")

    print("=" * 60)

if __name__ == '__main__':
    verify_test_mask_loading()
