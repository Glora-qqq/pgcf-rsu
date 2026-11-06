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
