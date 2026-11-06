"""Read in full sequence CSI, Masks and visual features, For export. """

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional

from .csi import load_test_csi
from .mask_io import load_test_mask
from .rgb_io import load_test_rgb_features

SCENE_CSI_SUFFIX = {
    'Scene1': 'high_fencha',
    'Scene2': 'widelane_low',
    'Scene3': 'widelane_high',
    'Scene_1': 'high_fencha',
    'Scene_2': 'widelane_low',
    'Scene_3': 'widelane_high',
}

VEHICLE_NAME_MAPPING = {
    'UAV1': 'AT1',
    'UAV2': 'AT2',
    'UAV3': 'AT3',
    'UAV4': 'AT4',
    'UAV5': 'AT5',
    'UAV6': 'AT6',
}

def load_train_csi(
    data_root: Path,
    scene: str,
    rsf: str,
    vehicle: str,
    num_frames: int = 1500,
) -> Optional[np.ndarray]:

    import scipy.io as sio

    reverse_mapping = {v: k for k, v in VEHICLE_NAME_MAPPING.items()}
    vehicle_csi = reverse_mapping.get(vehicle, vehicle)

    csi_dir = data_root / scene / rsf / 'CSI'

    suffix = SCENE_CSI_SUFFIX.get(scene, '')
    if suffix:
        csi_file = csi_dir / f'{rsf}_{vehicle_csi}_{suffix}_CSI.mat'
    else:
        csi_file = csi_dir / f'{rsf}_{vehicle_csi}_CSI.mat'

    if not csi_file.exists():
        print(f"[WARNING] CSIFile does not exist: {csi_file}")
        return None

    try:
        mat_data = sio.loadmat(csi_file)

        csi_key = None
        csi_raw = None

        if 'CSI' in mat_data:
            csi_key = 'CSI'
            csi_raw = mat_data['CSI']
        elif 'csi' in mat_data:
            csi_key = 'csi'
            csi_raw = mat_data['csi']
        else:
            for key in mat_data.keys():
                if not key.startswith('__'):
                    csi_key = key
                    csi_raw = mat_data[key]
                    break

        if csi_raw is None:
            print(f"[WARNING] Not foundCSIData")
            return None

        if csi_raw.dtype == object:
            if csi_raw.ndim == 2 and csi_raw.shape[0] == 1:
                num_frames_raw = csi_raw.shape[1]
                frame_list = []
                for i in range(num_frames_raw):
                    frame_data = csi_raw[0, i]
                    if isinstance(frame_data, np.ndarray) and frame_data.size > 0:
                        if frame_data.shape == (128, 32):
                            frame_list.append(frame_data)
                        elif frame_data.shape == (32, 128):
                            frame_list.append(frame_data.T)
                        else:
                            print(f"[WARNING] Frame{i}Dimension Exception: {frame_data.shape}")
                            frame_list.append(np.zeros((128, 32), dtype=complex))
                    else:
                        frame_list.append(np.zeros((128, 32), dtype=complex))

                if len(frame_list) > 0:
                    csi_data = np.stack(frame_list, axis=0)
                else:
                    print(f"[WARNING] No valid frame")
                    return None
            else:
                print(f"[WARNING] objectArray format not supported: {csi_raw.shape}")
                return None

        elif csi_raw.ndim == 3:
            if csi_raw.shape[0] == 128:
                csi_data = np.transpose(csi_raw, (2, 0, 1))
            elif csi_raw.shape[2] == 128:
                csi_data = np.transpose(csi_raw, (0, 2, 1))
            else:
                csi_data = csi_raw

        else:
            print(f"[WARNING] CSIDimension does not meet expectations: {csi_raw.shape}, dtype={csi_raw.dtype}")
            return None

        T = csi_data.shape[0]
        if T < num_frames:
            pad_width = ((0, num_frames - T), (0, 0), (0, 0))
            csi_data = np.pad(csi_data, pad_width, mode='constant', constant_values=0)
        elif T > num_frames:
            csi_data = csi_data[:num_frames]

        csi_real = np.real(csi_data)
        csi_imag = np.imag(csi_data)
        csi_output = np.stack([csi_real, csi_imag], axis=0).astype(np.float32)

        return csi_output

    except Exception as e:
        print(f"[ERROR] LoadCSIFailed: {e}")
        return None

def load_train_mask(
    mask_root: Path,
    scene: str,
    rsf: str,
    vehicle: str,
    num_frames: int = 1500,
) -> Optional[np.ndarray]:

    reverse_mapping = {v: k for k, v in VEHICLE_NAME_MAPPING.items()}
    vehicle_mask = reverse_mapping.get(vehicle, vehicle)

    mask_file = mask_root / scene / rsf / f'{vehicle_mask}_mask.npy'

    if not mask_file.exists():
        print(f"[WARNING] MaskFile does not exist: {mask_file}")
        return np.zeros(num_frames, dtype=np.float32)

    try:
        mask = np.load(mask_file)

        if len(mask) < num_frames:
            mask = np.pad(mask, (0, num_frames - len(mask)),
                         mode='constant', constant_values=1.0)
        elif len(mask) > num_frames:
            mask = mask[:num_frames]

        return mask.astype(np.float32)

    except Exception as e:
        print(f"[ERROR] LoadMaskFailed: {e}")
        return np.zeros(num_frames, dtype=np.float32)

def load_train_rgb(
    rgb_root: Path,
    scene: str,
    rsf: str,
    vehicle: str = None,
    frame_range: tuple = None,
    feature_dim: int = 1024,
) -> Optional[np.ndarray]:

    rgb_file = rgb_root / f'{scene}_{rsf}_features.npz'

    if not rgb_file.exists():
        print(f"[WARNING] RGBFile does not exist: {rgb_file}")
        if frame_range:
            T = frame_range[1] - frame_range[0] + 1
            return np.zeros((T, feature_dim), dtype=np.float32)
        else:
            return np.zeros(feature_dim, dtype=np.float32)

    try:
        data = np.load(rgb_file)

        if 'fused' not in data:
            print(f"[WARNING] Not found'fused'Features, Availablekeys: {list(data.keys())}")
            if frame_range:
                T = frame_range[1] - frame_range[0] + 1
                return np.zeros((T, feature_dim), dtype=np.float32)
            else:
                return np.zeros(feature_dim, dtype=np.float32)

        rgb_features = data['fused']

        if frame_range is not None:
            start_frame, end_frame = frame_range
            rgb_features = rgb_features[start_frame:end_frame+1]

        return rgb_features.astype(np.float32)

    except Exception as e:
        print(f"[ERROR] LoadRGBFailed: {e}")
        if frame_range:
            T = frame_range[1] - frame_range[0] + 1
            return np.zeros((T, feature_dim), dtype=np.float32)
        else:
            return np.zeros(feature_dim, dtype=np.float32)

def load_vehicle_data_for_extraction(
    scene: str,
    rsf: str,
    vehicle: str,
    data_root: Path,
    mask_root: Path,
    rgb_root: Path,
    is_test: bool = False,
    num_frames: int = 1500,
    rgb_feature_dim: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:

    if is_test:
        csi_scene = 'Scene_' + scene.replace('Scene', '')
        csi_data = load_test_csi(
            data_root, csi_scene, rsf, vehicle,
            num_frames_original=750,
            num_frames_padded=num_frames
        )
    else:
        csi_data = load_train_csi(
            data_root, scene, rsf, vehicle,
            num_frames=num_frames
        )

    if csi_data is None:
        raise RuntimeError(f"Unable to loadCSIData: {scene}/{rsf}/{vehicle}")

    if is_test:
        mask_data = load_test_mask(
            mask_root, scene, rsf, vehicle,
            num_frames_original=750,
            num_frames_padded=num_frames
        )
    else:
        mask_data = load_train_mask(
            mask_root, scene, rsf, vehicle,
            num_frames=num_frames
        )

    if mask_data is None:
        mask_data = np.zeros(num_frames, dtype=np.float32)

    if is_test:
        rgb_data = load_test_rgb_features(
            rgb_root, scene, rsf,
            num_frames_original=750,
            num_frames_padded=num_frames
        )
        if rgb_data is not None and rgb_data.ndim == 2:
            rgb_data = np.mean(rgb_data, axis=0)
    else:
        rgb_data = load_train_rgb(
            rgb_root, scene, rsf, vehicle,
            frame_range=None,
            feature_dim=rgb_feature_dim
        )

        if rgb_data is not None and rgb_data.ndim == 2:
            rgb_data = np.mean(rgb_data, axis=0)

    if rgb_data is None:
        rgb_data = np.zeros(rgb_feature_dim, dtype=np.float32)

    csi_tensor = torch.from_numpy(csi_data)
    mask_tensor = torch.from_numpy(mask_data)
    rgb_tensor = torch.from_numpy(rgb_data)

    meta = {
        'scene': scene,
        'rsf': rsf,
        'vehicle': vehicle,
        'is_test': is_test,
        'num_frames': num_frames,
    }

    return csi_tensor, mask_tensor, rgb_tensor, meta
