"""According to the training set CSI Valid or not, Generate Fine Tuning Mask. """

import os
import sys
import numpy as np
import scipy.io as sio
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def analyze_csi_file(csi_file_path, expected_frames=1500, zero_threshold=1e-6):

    try:
        csi_mat = sio.loadmat(str(csi_file_path))

        csi_key = [k for k in csi_mat.keys() if not k.startswith('__')][0]
        csi_1500_frames = csi_mat[csi_key]

        frame_status = []
        manual_mask = np.zeros(expected_frames, dtype=np.float32)

        for i in range(min(1500, expected_frames)):
            try:
                csi_frame = csi_1500_frames[0, i]

                if isinstance(csi_frame, np.ndarray):
                    csi_array = csi_frame
                else:
                    csi_array = np.array(csi_frame, dtype=complex)

                if csi_array.size == 0 or np.allclose(csi_array, 0):
                    frame_status.append('missing')
                    manual_mask[i] = 1.0
                else:
                    frame_status.append('valid')
                    manual_mask[i] = 0.0
            except Exception as e:
                frame_status.append('error')
                manual_mask[i] = 1.0

        if len(frame_status) < expected_frames:
            for i in range(len(frame_status), expected_frames):
                frame_status.append('padding')
                manual_mask[i] = 1.0

        valid_count = sum(1 for s in frame_status if s == 'valid')
        missing_count = expected_frames - valid_count

        return manual_mask, frame_status, valid_count, missing_count

    except Exception as e:
        print(f"[ERROR] Analytics CSI File failed: {csi_file_path}, {e}")
        manual_mask = np.ones(expected_frames, dtype=np.float32)
        return manual_mask, [], 0, expected_frames

def generate_all_masks(data_root='train', output_root='WiFo_CSI/masks'):

    print("=" * 80)
    print("CSI Mask Generation (By Scene -> RSF -> Vehicle Hierarchy) ")
    print("=" * 80)

    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_files': 0,
        'total_valid_frames': 0,
        'total_missing_frames': 0,
        'by_scene': {},
        'by_rsf': defaultdict(lambda: {'files': 0, 'valid': 0, 'missing': 0}),
    }

    for scene_dir in sorted(data_root.glob('Scene*')):
        scene_name = scene_dir.name
        print(f"\n{scene_name}:")

        stats['by_scene'][scene_name] = {
            'files': 0,
            'valid': 0,
            'missing': 0,
            'rsfs': {}
        }

        rsf_dirs = sorted([d for d in scene_dir.glob('RSF*') if d.is_dir()])

        for rsf_dir in rsf_dirs:
            rsf_name = rsf_dir.name
            csi_dir = rsf_dir / 'CSI'

            if not csi_dir.exists():
                continue

            print(f"  {rsf_name}:")

            stats['by_scene'][scene_name]['rsfs'][rsf_name] = {
                'files': 0,
                'valid': 0,
                'missing': 0
            }

            output_dir = output_root / scene_name / rsf_name
            output_dir.mkdir(parents=True, exist_ok=True)

            for csi_file in sorted(csi_dir.glob('*.mat')):
                filename = csi_file.stem
                parts = filename.split('_')
                if len(parts) >= 2:
                    vehicle_name = parts[1]
                else:
                    print(f"[WARN] Could not fetch car number from filename: {filename}")
                    continue

                manual_mask, frame_status, valid_count, missing_count = analyze_csi_file(
                    csi_file,
                    expected_frames=1500,
                    zero_threshold=1e-6
                )

                output_file = output_dir / f"{vehicle_name}_mask.npy"
                np.save(output_file, manual_mask)

                stats['total_files'] += 1
                stats['total_valid_frames'] += valid_count
                stats['total_missing_frames'] += missing_count

                stats['by_scene'][scene_name]['files'] += 1
                stats['by_scene'][scene_name]['valid'] += valid_count
                stats['by_scene'][scene_name]['missing'] += missing_count

                stats['by_scene'][scene_name]['rsfs'][rsf_name]['files'] += 1
                stats['by_scene'][scene_name]['rsfs'][rsf_name]['valid'] += valid_count
                stats['by_scene'][scene_name]['rsfs'][rsf_name]['missing'] += missing_count

                stats['by_rsf'][rsf_name]['files'] += 1
                stats['by_rsf'][rsf_name]['valid'] += valid_count
                stats['by_rsf'][rsf_name]['missing'] += missing_count

                valid_ratio = valid_count / 1500 * 100
                print(f"    {vehicle_name}: {valid_count}/1500 Valid Frames ({valid_ratio:.1f}%) -> {output_file}")

    print("\n" + "=" * 80)
    print("Mask generation complete")
    print("=" * 80)
    print(f"\nPresidential Statistics:")
    print(f"  Total Files: {stats['total_files']}")
    print(f"  Total Effective Frames: {stats['total_valid_frames']}")
    print(f"  Total Missing Frames: {stats['total_missing_frames']}")
    print(f"  Average effective rate: {stats['total_valid_frames'] / (stats['total_files'] * 1500) * 100:.1f}%")

    print(f"\nBy RSF Statistics:")
    for rsf_name, rsf_stats in sorted(stats['by_rsf'].items()):
        if rsf_stats['files'] > 0:
            valid_ratio = rsf_stats['valid'] / (rsf_stats['files'] * 1500) * 100
            print(f"  {rsf_name}: {rsf_stats['files']} File, {rsf_stats['valid']}/{rsf_stats['files']*1500} Valid Frames ({valid_ratio:.1f}%)")

    stats_file = output_root / 'mask_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        average_valid_ratio = stats['total_valid_frames'] / (stats['total_files'] * 1500) * 100 if stats['total_files'] > 0 else 0
        stats_dict = {
            'total_files': stats['total_files'],
            'total_valid_frames': stats['total_valid_frames'],
            'total_missing_frames': stats['total_missing_frames'],
            'average_valid_ratio': average_valid_ratio,
            'by_scene': stats['by_scene'],
            'by_rsf': dict(stats['by_rsf'])
        }
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)

    print(f"\nStats saved: {stats_file}")
    print(f"Mask file saved to: {output_root}")

if __name__ == '__main__':
    data_root = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[2] / 'data')) / 'train'
    output_root = Path(__file__).resolve().parent

    generate_all_masks(str(data_root), str(output_root))
