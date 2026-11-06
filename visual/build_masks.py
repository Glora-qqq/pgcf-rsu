"""Generate from Coverage Text RGB Missing Frame Mask. """

import numpy as np
import os
import re
from collections import defaultdict

def parse_missing_string(missing_str):

    ranges = []
    parts = missing_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            ranges.append((start, end))

    return ranges

def parse_coverage_file(file_path):

    missing_info = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_scene = None
    current_rsf = None
    processed_rsf = set()

    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith('Scene'):
            current_scene = line.rstrip(':')
            processed_rsf.clear()

        elif current_scene and line.endswith(':') and 'RSF' in line:
            current_rsf = line.rstrip(':').strip()

        elif current_scene and current_rsf:
            view_match = re.match(r'left\s*:\s*\d+\s*snapshots.*missing\s*\d+', line)

            if view_match:
                key = (current_scene, current_rsf)

                if key in processed_rsf:
                    continue

                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()

                    if next_line.startswith('Missing:'):
                        missing_str = next_line.replace('Missing:', '').strip()
                        missing_ranges = parse_missing_string(missing_str)
                        missing_info[key] = missing_ranges
                        processed_rsf.add(key)

    return dict(missing_info)

def create_mask_array(missing_ranges, total_frames, snapshot_offset=0):

    mask = np.zeros(total_frames, dtype=bool)

    for start, end in missing_ranges:
        mask_start = start - 1 - snapshot_offset
        mask_end = end - snapshot_offset

        if mask_start < 0 or mask_end > total_frames:
            print(f"⚠️  Warning: Interval {start}-{end} Out of range, Skip")
            continue

        mask[mask_start:mask_end] = True

    return mask

def generate_masks(coverage_file, output_dir, total_frames, dataset_name, snapshot_offset=0):

    print(f"\n{'='*70}")
    print(f"Generate{dataset_name}SetRGB mask")
    print(f"{'='*70}")

    missing_info = parse_coverage_file(coverage_file)

    print(f"\nDetected {len(missing_info)} PieceScene-RSFCombination")

    os.makedirs(output_dir, exist_ok=True)

    scenes = defaultdict(list)
    for (scene, rsf), ranges in missing_info.items():
        scenes[scene].append((rsf, ranges))

    total_missing = 0

    for scene in sorted(scenes.keys()):
        scene_dir = os.path.join(output_dir, scene)
        os.makedirs(scene_dir, exist_ok=True)

        print(f"\n{scene}:")

        for rsf, missing_ranges in sorted(scenes[scene]):
            mask = create_mask_array(missing_ranges, total_frames, snapshot_offset)

            filename = f"{scene}_{rsf}_RGBmask.npy"
            filepath = os.path.join(scene_dir, filename)

            np.save(filepath, mask)

            num_missing = mask.sum()
            total_missing += num_missing
            missing_rate = (num_missing / total_frames) * 100

            print(f"  {rsf}: {num_missing}/{total_frames} Frame Missing ({missing_rate:.1f}%)")
            print(f"       Missing interval: {missing_ranges}")
            print(f"       ✅ Save to: {filepath}")

    total_frames_all = len(missing_info) * total_frames
    overall_missing_rate = (total_missing / total_frames_all) * 100

    print(f"\n{'='*70}")
    print(f"{dataset_name}Set Stats:")
    print(f"{'='*70}")
    print(f"  Scene-RSFNumber of combinations: {len(missing_info)}")
    print(f"  Total frames: {total_frames_all}")
    print(f"  Missing frames: {total_missing}")
    print(f"  Missing rate: {overall_missing_rate:.2f}%")

    return missing_info

def verify_mask(mask_file):

    mask = np.load(mask_file)

    print(f"\nMaskVerify: {os.path.basename(mask_file)}")
    print(f"  Shape: {mask.shape}")
    print(f"  Data Type: {mask.dtype}")
    print(f"  Missing frames: {mask.sum()}")
    print(f"  Missing rate: {mask.sum() / len(mask) * 100:.2f}%")

    missing_frames = np.where(mask)[0] + 1
    if len(missing_frames) > 0:
        diff = np.diff(missing_frames)
        breaks = np.where(diff > 1)[0]

        ranges = []
        start_idx = 0
        for break_idx in breaks:
            ranges.append((missing_frames[start_idx], missing_frames[break_idx]))
            start_idx = break_idx + 1
        ranges.append((missing_frames[start_idx], missing_frames[-1]))

        print(f"  Missing interval: {ranges}")

def main():

    print("=" * 70)
    print("GenerateRGBMissing FrameMaskFile")
    print("=" * 70)

    base_dir = os.environ.get('RGB_WORK_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs'))
    os.makedirs(base_dir, exist_ok=True)
    data_root = os.environ.get('DATA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

    train_output_dir = os.path.join(base_dir, 'mask-RGB-train')
    train_missing = generate_masks(
        coverage_file=os.environ.get('RGB_COVERAGE_TRAIN', os.path.join(data_root, 'rgb_coverage_train_output.txt')),
        output_dir=train_output_dir,
        total_frames=750,
        dataset_name='Training',
        snapshot_offset=0
    )

    test_output_dir = os.path.join(base_dir, 'mask-RGB-test')
    test_missing = generate_masks(
        coverage_file=os.environ.get('RGB_COVERAGE_TEST', os.path.join(data_root, 'rgb_coverage_test_output.txt')),
        output_dir=test_output_dir,
        total_frames=750,
        dataset_name='Test',
        snapshot_offset=750
    )

    print("\n" + "=" * 70)
    print("Examples of validationMaskFile")
    print("=" * 70)

    if train_missing:
        first_scene, first_rsf = list(train_missing.keys())[0]
        example_file = os.path.join(
            train_output_dir,
            first_scene,
            f"{first_scene}_{first_rsf}_RGBmask.npy"
        )
        if os.path.exists(example_file):
            verify_mask(example_file)

    readme_path = os.path.join(base_dir, 'README_mask.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("RGBMissing FrameMaskDocument description\n")
        f.write("="*70 + "\n\n")
        f.write("Directory Structure:\n")
        f.write("env_feature/\n")
        f.write("├── mask-RGB-train/\n")
        f.write("│   ├── Scene1/\n")
        f.write("│   │   ├── Scene1_RSF1_RGBmask.npy\n")
        f.write("│   │   ├── Scene1_RSF4_RGBmask.npy\n")
        f.write("│   │   └── Scene1_RSF7_RGBmask.npy\n")
        f.write("│   ├── Scene2/\n")
        f.write("│   └── Scene3/\n")
        f.write("└── mask-RGB-test/\n")
        f.write("    └── (Ditto structure)\n\n")
        f.write("File Format:\n")
        f.write("- Filename: SceneX_RSFY_RGBmask.npy\n")
        f.write("- Data Type: numpy.ndarray, dtype=bool\n")
        f.write("- Shape: [750] (750Frame)\n")
        f.write("- Value: True=Missing Frame, False=Valid Frames\n")
        f.write("- Index: 0-based (mask[0]Corresponding to1Frame)\n\n")
        f.write("Usage example:\n")
        f.write("mask = np.load('mask-RGB-train/Scene1/Scene1_RSF1_RGBmask.npy')\n")
        f.write("missing_frame_ids = np.where(mask)[0] + 1\n")
        f.write("valid_frame_ids = np.where(~mask)[0] + 1\n")

    print(f"\n✅ Instructions saved to: {readme_path}")

    print("\n" + "=" * 70)
    print("🎉 Done! MaskFile Generated")
    print("=" * 70)
    print(f"\nTraining Setsmask: {train_output_dir}")
    print(f"Test setmask: {test_output_dir}")
    print("\nNext: Use thesemaskFile ProceedDINOv2Feature extraction")

if __name__ == '__main__':
    main()
