"""Read Test Set Visual Fusion npz, and make it to the model length.. """

import numpy as np
from pathlib import Path
from typing import Optional

def load_test_rgb_features(
    rgb_root: Path,
    scene: str,
    rsf: str,
    num_frames_original: int = 750,
    num_frames_padded: int = 1500,
    padding_strategy: str = 'mean',
    feature_key: str = 'fused',
) -> Optional[np.ndarray]:

    rgb_file = rgb_root / f'{scene}_{rsf}_features.npz'

    if not rgb_file.exists():
        print(f"[WARNING] RGBFeature file does not exist: {rgb_file}")
        return None

    try:
        npz_data = np.load(rgb_file)

        if feature_key not in npz_data:
            print(f"[WARNING] Key '{feature_key}' Absent, Availablekeys: {list(npz_data.keys())}")
            return None

        rgb_750 = npz_data[feature_key].astype(np.float32)

        if rgb_750.shape[0] != num_frames_original:
            print(f"[WARNING] RGBFrame number mismatch: Expectations{num_frames_original}, Actual{rgb_750.shape[0]}")
            if rgb_750.shape[0] > num_frames_original:
                rgb_750 = rgb_750[:num_frames_original]
            else:
                pad_len = num_frames_original - rgb_750.shape[0]
                rgb_750 = np.pad(
                    rgb_750,
                    ((0, pad_len), (0, 0)),
                    mode='edge'
                )

        if padding_strategy == 'mean':
            rgb_mean = rgb_750.mean(axis=0, keepdims=True)
            rgb_padding = np.repeat(rgb_mean, num_frames_padded - num_frames_original, axis=0)
        elif padding_strategy == 'repeat':
            last_frame = rgb_750[-1:, :]
            rgb_padding = np.repeat(last_frame, num_frames_padded - num_frames_original, axis=0)
        else:
            raise ValueError(f"UnknownpaddingStrategy: {padding_strategy}")

        rgb_1500 = np.concatenate([rgb_750, rgb_padding], axis=0)

        return rgb_1500.astype(np.float32)

    except Exception as e:
        print(f"[ERROR] RGBFeature loading failed {rgb_file}: {e}")
        return None
