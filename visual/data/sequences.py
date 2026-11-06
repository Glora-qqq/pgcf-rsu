"""Read in chronological order RGB Frame, For complementary model training. """

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class RGBSequenceDataset(Dataset):

    def __init__(self, image_dir, mask_dir, scene, rsf, view, transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.scene = scene
        self.rsf = rsf
        self.view = view

        mask_path = get_mask_path(mask_dir, scene, rsf)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"MaskFile does not exist: {mask_path}")

        self.mask = np.load(mask_path)

        assert len(self.mask) == TOTAL_FRAMES,\
            f"MaskWrong length: {len(self.mask)} != {TOTAL_FRAMES}"

        self.frame_paths = []
        for frame_id in range(1, TOTAL_FRAMES + 1):
            img_path = get_image_path(image_dir, scene, rsf, frame_id, view)
            self.frame_paths.append(img_path)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.transform = transform

        self.num_missing = self.mask.sum()
        self.num_valid = TOTAL_FRAMES - self.num_missing

    def __len__(self):

        return TOTAL_FRAMES

    def __getitem__(self, idx):

        frame_id = idx + 1
        is_missing = self.mask[idx]

        if is_missing:
            return None, frame_id, True
        else:
            img_path = self.frame_paths[idx]

            if not os.path.exists(img_path):
                print(f"⚠️ Warning: Image file does not exist {img_path}")
                return None, frame_id, True

            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                return image, frame_id, False

            except Exception as e:
                print(f"⚠️ Warning: Image failed to load {img_path}: {e}")
                return None, frame_id, True

    def get_sample_info(self):

        missing_frames = np.where(self.mask)[0] + 1

        missing_ranges = []
        if len(missing_frames) > 0:
            diff = np.diff(missing_frames)
            breaks = np.where(diff > 1)[0]

            start_idx = 0
            for break_idx in breaks:
                missing_ranges.append((
                    int(missing_frames[start_idx]),
                    int(missing_frames[break_idx])
                ))
                start_idx = break_idx + 1
            missing_ranges.append((
                int(missing_frames[start_idx]),
                int(missing_frames[-1])
            ))

        return {
            'scene': self.scene,
            'rsf': self.rsf,
            'view': self.view,
            'total_frames': TOTAL_FRAMES,
            'valid_frames': int(self.num_valid),
            'missing_frames': int(self.num_missing),
            'missing_rate': float(self.num_missing / TOTAL_FRAMES),
            'missing_ranges': missing_ranges
        }

    def __repr__(self):
        info = self.get_sample_info()
        return (f"RGBSequenceDataset({info['scene']}_{info['rsf']}_{info['view']}, "
                f"{info['valid_frames']}/{info['total_frames']} frames, "
                f"missing={info['missing_rate']:.1%})")

def create_sample_info(image_dir, mask_dir):

    samples = get_all_samples(image_dir, mask_dir)

    sample_infos = []
    for scene, rsf, view in samples:
        try:
            dataset = RGBSequenceDataset(image_dir, mask_dir, scene, rsf, view)
            info = dataset.get_sample_info()
            sample_infos.append(info)
        except Exception as e:
            print(f"⚠️ Warning: Unable to create sample {scene}_{rsf}_{view}: {e}")

    return sample_infos

def test_dataset():

    print("=" * 70)
    print("TestRGBSequenceDataset")
    print("=" * 70)

    samples = get_all_samples(IMAGE_DIR_TRAIN, MASK_DIR_TRAIN)

    if len(samples) == 0:
        print("❌ Error: No samples found")
        print(f"Please checkmaskContents: {MASK_DIR_TRAIN}")
        return

    print(f"\nFound {len(samples)} samples")
    print(f"First sample: {samples[0]}")

    scene, rsf, view = samples[0]
    dataset = RGBSequenceDataset(IMAGE_DIR_TRAIN, MASK_DIR_TRAIN, scene, rsf, view)

    print(f"\n{dataset}")

    info = dataset.get_sample_info()
    print("\nSample Details:")
    print(f"  Scenario: {info['scene']}")
    print(f"  RSF: {info['rsf']}")
    print(f"  Angle of View: {info['view']}")
    print(f"  Total frames: {info['total_frames']}")
    print(f"  Effective Frames: {info['valid_frames']}")
    print(f"  Missing frames: {info['missing_frames']}")
    print(f"  Missing rate: {info['missing_rate']:.2%}")
    print(f"  Missing interval: {info['missing_ranges']}")

    print("\nBefore test read10Frame:")
    for i in range(10):
        image, frame_id, is_missing = dataset[i]

        if is_missing:
            print(f"  Frame {frame_id}: ❌ Missing")
        else:
            print(f"  Frame {frame_id}: ✅ Active, shape={image.shape}")

    if len(info['missing_ranges']) > 0:
        first_missing_start, first_missing_end = info['missing_ranges'][0]
        print(f"\nTest first missing interval [{first_missing_start}, {first_missing_end}]:")

        image, frame_id, is_missing = dataset[first_missing_start - 2]
        print(f"  Frame {frame_id} (Before deletion): {'❌ Missing' if is_missing else '✅ Active'}")

        image, frame_id, is_missing = dataset[first_missing_start - 1]
        print(f"  Frame {frame_id} (Missing start): {'❌ Missing' if is_missing else '✅ Active'}")

        image, frame_id, is_missing = dataset[first_missing_end - 1]
        print(f"  Frame {frame_id} (Missing End): {'❌ Missing' if is_missing else '✅ Active'}")

        if first_missing_end < TOTAL_FRAMES:
            image, frame_id, is_missing = dataset[first_missing_end]
            print(f"  Frame {frame_id} (After Missing): {'❌ Missing' if is_missing else '✅ Active'}")

    print("\n" + "=" * 70)
    print("Count all samples")
    print("=" * 70)

    all_infos = create_sample_info(IMAGE_DIR_TRAIN, MASK_DIR_TRAIN)

    total_frames = sum(info['total_frames'] for info in all_infos)
    total_valid = sum(info['valid_frames'] for info in all_infos)
    total_missing = sum(info['missing_frames'] for info in all_infos)

    print(f"\nNumber of samples: {len(all_infos)}")
    print(f"Total frames: {total_frames}")
    print(f"Effective Frames: {total_valid}")
    print(f"Missing frames: {total_missing}")
    print(f"Total Missing Rate: {total_missing/total_frames:.2%}")

    print("\nStatistics by Scenario:")
    for scene in SCENES:
        scene_infos = [info for info in all_infos if info['scene'] == scene]
        if len(scene_infos) > 0:
            scene_total = sum(info['total_frames'] for info in scene_infos)
            scene_missing = sum(info['missing_frames'] for info in scene_infos)
            print(f"  {scene}: {len(scene_infos)} Sample, "
                  f"{scene_missing}/{scene_total} Missing "
                  f"({scene_missing/scene_total:.1%})")

    print("\n✅ Dataset Test Complete! ")

if __name__ == '__main__':
    test_dataset()
