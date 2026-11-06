"""Read test set CSI mat. """

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Optional, Dict

TEST_SCENE_CSI_SUFFIX = {
    'Scene_1': 'high_fencha',
    'Scene_2': 'widelane_low',
    'Scene_3': 'widelane_high',
}

def load_test_csi(
    data_root: Path,
    scene: str,
    rsf: str,
    vehicle: str,
    num_frames_original: int = 750,
    num_frames_padded: int = 1500,
) -> Optional[np.ndarray]:

    VEHICLE_NAME_MAPPING = {
        'UAV1': 'AT1', 'UAV2': 'AT2', 'UAV3': 'AT3',
        'UAV4': 'AT4', 'UAV5': 'AT5', 'UAV6': 'AT6',
    }
    reverse_mapping = {v: k for k, v in VEHICLE_NAME_MAPPING.items()}
    vehicle_csi = reverse_mapping.get(vehicle, vehicle)

    csi_dir = data_root / scene / rsf / 'CSI'
    suffix = TEST_SCENE_CSI_SUFFIX.get(scene, 'high_fencha')
    csi_file = csi_dir / f'{rsf}_{vehicle_csi}_{suffix}_CSI.mat'

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
                if not key.startswith('__') and ('CIR' in key or 'CSI' in key.upper()):
                    csi_key = key
                    csi_raw = mat_data[key]
                    break

            if csi_key is None:
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        csi_key = key
                        csi_raw = mat_data[key]
                        break

        if csi_raw is None:
            print(f"[WARNING] Not foundCSIData, Availablekeys: {mat_data.keys()}")
            return None

        if csi_raw.dtype == object:
            if csi_raw.ndim == 2:
                num_frames = csi_raw.shape[1]
                frame_list = []
                for i in range(num_frames):
                    frame_data = csi_raw[0, i]
                    if isinstance(frame_data, np.ndarray) and frame_data.size > 0:
                        frame_list.append(frame_data)
                    else:
                        frame_list.append(np.zeros((128, 32), dtype=complex))

                csi_complex = np.stack(frame_list, axis=0)
            else:
                print(f"[WARNING] UnknownobjectArray Format: {csi_raw.shape}")
                return None
        else:
            if csi_raw.ndim == 3 and csi_raw.shape[2] > csi_raw.shape[0]:
                csi_complex = np.transpose(csi_raw, (2, 0, 1))
            else:
                csi_complex = csi_raw

        if csi_complex.shape[0] != num_frames_original:
            if csi_complex.shape[0] > num_frames_original:
                csi_complex = csi_complex[:num_frames_original]
            else:
                pad_len = num_frames_original - csi_complex.shape[0]
                csi_complex = np.pad(
                    csi_complex,
                    ((0, pad_len), (0, 0), (0, 0)),
                    mode='constant',
                    constant_values=0
                )

        csi_real = csi_complex.real.astype(np.float32)
        csi_imag = csi_complex.imag.astype(np.float32)
        csi_separated = np.stack([csi_real, csi_imag], axis=0)

        pad_len = num_frames_padded - num_frames_original
        csi_final = np.pad(
            csi_separated,
            ((0, 0), (0, pad_len), (0, 0), (0, 0)),
            mode='constant',
            constant_values=0
        )

        assert csi_final.shape == (2, num_frames_padded, 128, 32),\
            f"CSIWrong shape: {csi_final.shape}, Expectations: (2, {num_frames_padded}, 128, 32)"

        return csi_final.astype(np.float32)

    except Exception as e:
        print(f"[ERROR] LoadCSIFailed {csi_file}: {e}")
        return None

def verify_test_csi_loading():

    from config_test import TEST_DATA_ROOT, TEST_SCENES, SCENE_NAME_MAPPING

    print("=" * 60)
    print("Validation testCSILoad")
    print("=" * 60)

    test_scene = 'Scene_1'
    test_rsf = 'RSF_1'
    test_vehicle = 'Car1'

    csi = load_test_csi(
        data_root=TEST_DATA_ROOT,
        scene=test_scene,
        rsf=test_rsf,
        vehicle=test_vehicle,
    )

    if csi is not None:
        print(f"✓ CSILoaded successfully")
        print(f"  Shape: {csi.shape}")
        print(f"  Data Type: {csi.dtype}")
        print(f"  ago750Frame Range: [{csi[:, :750].min():.4f}, {csi[:, :750].max():.4f}]")
        print(f"  After750Frame Range: [{csi[:, 750:].min():.4f}, {csi[:, 750:].max():.4f}]")
        print(f"  After750Frame Full0Check: {np.all(csi[:, 750:] == 0)}")
    else:
        print("✗ CSILoading failed")

    print("=" * 60)

if __name__ == '__main__':
    verify_test_csi_loading()
