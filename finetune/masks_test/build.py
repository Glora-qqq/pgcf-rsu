"""According to the test set CSI Generate Mask, and mark the second half as to be predicted. """

import os
import sys
import numpy as np
import scipy.io as sio
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def load_test_csi(csi_file_path):

    try:
        data = sio.loadmat(str(csi_file_path))

        csi_key = None
        for key in data.keys():
            if not key.startswith('__') and ('CIR' in key or 'CSI' in key or 'csi' in key):
                csi_key = key
                break

        if csi_key is None:
            for key in data.keys():
                if not key.startswith('__'):
                    csi_key = key
                    break

        if csi_key is None:
            return None, None

        csi_data = data[csi_key]
        return csi_data, csi_key

    except Exception as e:
        return None, None

def analyze_test_csi_file(csi_file_path, test_frames=750, total_frames=1500):

    try:
        csi_data, csi_key = load_test_csi(csi_file_path)

        if csi_data is None:
            manual_mask = np.ones(total_frames, dtype=np.float32)
            return manual_mask, [], 0, total_frames

        if csi_data.dtype != object:
            print(f"[WARN] Unexpected CSI format: {csi_data.dtype}, expected object")
            manual_mask = np.ones(total_frames, dtype=np.float32)
            return manual_mask, [], 0, total_frames

        if csi_data.ndim == 2:
            actual_frames = csi_data.shape[1]
        else:
            actual_frames = csi_data.size

        if actual_frames != test_frames:
            print(f"[WARN] Expected {test_frames} frames, got {actual_frames}")

        frame_status = []
        manual_mask = np.zeros(total_frames, dtype=np.float32)

        for i in range(min(actual_frames, test_frames)):
            try:
                if csi_data.ndim == 2:
                    frame_data = csi_data[0, i]
                else:
                    frame_data = csi_data.flat[i]

                if frame_data is None:
                    frame_status.append('missing')
                    manual_mask[i] = 1.0
                elif isinstance(frame_data, np.ndarray):
                    if np.all(frame_data == 0) or np.all(np.isnan(frame_data)):
                        frame_status.append('missing')
                        manual_mask[i] = 1.0
                    elif frame_data.size < 10:
                        frame_status.append('missing')
                        manual_mask[i] = 1.0
                    else:
                        frame_status.append('valid')
                        manual_mask[i] = 0.0
                else:
                    frame_status.append('missing')
                    manual_mask[i] = 1.0

            except Exception as e:
                frame_status.append('error')
                manual_mask[i] = 1.0

        for i in range(test_frames, total_frames):
            frame_status.append('padding')
            manual_mask[i] = 1.0

        valid_count = sum(1 for s in frame_status if s == 'valid')
        missing_count = total_frames - valid_count

        return manual_mask, frame_status, valid_count, missing_count

    except Exception as e:
        print(f"[ERROR] Failed to analyze {csi_file_path}: {e}")
        manual_mask = np.ones(total_frames, dtype=np.float32)
        return manual_mask, [], 0, total_frames

def build(test_root, output_root):

    print("=" * 80)
    print("Test setCSIMask Generation (Scene -> RSF -> Vehicle)")
    print("=" * 80)

    test_root = Path(test_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_samples': 0,
        'total_valid_frames': 0,
        'total_missing_frames': 0,
        'samples': [],
        'scene_stats': defaultdict(lambda: defaultdict(dict)),
    }

    for scene_dir in sorted(test_root.glob('Scene_*')):
        scene_name = scene_dir.name

        scene_name_train = scene_name.replace('_', '')

        print(f"\nProcessing {scene_name}...")

        for rsf_dir in sorted(scene_dir.glob('RSF*')):
            rsf_name = rsf_dir.name

            csi_dir = rsf_dir / 'CSI'
            if not csi_dir.exists():
                print(f"  [WARN] {scene_name}/{rsf_name}/CSI not found")
                continue

            csi_files = list(csi_dir.glob('*.mat'))
            if not csi_files:
                print(f"  [WARN] No CSI files in {scene_name}/{rsf_name}/CSI")
                continue

            print(f"  {rsf_name}: {len(csi_files)} vehicles")

            output_dir = output_root / scene_name_train / rsf_name
            output_dir.mkdir(parents=True, exist_ok=True)

            for csi_file in tqdm(sorted(csi_files), desc=f"    {rsf_name}"):
                parts = csi_file.stem.split('_')
                vehicle_name = parts[1]

                manual_mask, frame_status, valid_count, missing_count = analyze_test_csi_file(csi_file)

                mask_file = output_dir / f'{vehicle_name}_mask.npy'
                np.save(mask_file, manual_mask)

                stats['total_samples'] += 1
                stats['total_valid_frames'] += valid_count
                stats['total_missing_frames'] += missing_count

                sample_info = {
                    'scene': scene_name,
                    'rsf': rsf_name,
                    'vehicle': vehicle_name,
                    'csi_file': str(csi_file.relative_to(test_root)),
                    'mask_file': str(mask_file.relative_to(output_root)),
                    'valid_frames': valid_count,
                    'missing_frames': missing_count,
                    'missing_ratio': missing_count / 1500,
                }
                stats['samples'].append(sample_info)

                stats['scene_stats'][scene_name][rsf_name][vehicle_name] = {
                    'valid': valid_count,
                    'missing': missing_count,
                }

    print_summary(stats, output_root)

    stats_file = output_root / 'mask_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVE] Statistics saved to: {stats_file}")

    return stats

def print_summary(stats, output_root):

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"Total valid frames: {stats['total_valid_frames']:,}")
    print(f"Total missing frames: {stats['total_missing_frames']:,}")

    if stats['total_samples'] > 0:
        avg_valid = stats['total_valid_frames'] / stats['total_samples']
        avg_missing = stats['total_missing_frames'] / stats['total_samples']
        avg_valid_ratio = avg_valid / 1500

        print(f"\nAverage per sample:")
        print(f"  Valid frames: {avg_valid:.1f} ({avg_valid_ratio*100:.1f}%)")
        print(f"  Missing frames: {avg_missing:.1f} ({(1-avg_valid_ratio)*100:.1f}%)")

    print(f"\n[SAVE] Masks saved to: {output_root}/")
    print(f"  Structure: Scene*/RSF*/Vehicle_mask.npy")

if __name__ == '__main__':
    test_root = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[2] / 'data')) / 'test'
    output_root = Path(__file__).resolve().parent

    if not test_root.exists():
        print(f"[ERROR] Test data not found: {test_root}")
        sys.exit(1)

    print(f"Test data root: {test_root}")
    print(f"Output root: {output_root}\n")

    stats = build(test_root, output_root)

    print("\n" + "=" * 80)
    print("Test mask generation completed!")
    print("=" * 80)
