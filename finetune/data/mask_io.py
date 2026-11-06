"""Read test set CSI Effective Frame Mask. """

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
                print(f"[WARNING] MaskAbnormal length: {len(mask)}, Use defaultmask")
                mask_extended = np.zeros(num_frames_padded, dtype=np.float32)

            mask_extended[force_mask_start:] = 1.0

            return mask_extended

        except Exception as e:
            print(f"[WARNING] MaskLoading failed {mask_file}: {e}, Use defaultmask")

    mask = np.zeros(num_frames_padded, dtype=np.float32)
    mask[force_mask_start:] = 1.0

    return mask
