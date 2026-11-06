"""Organize completion reasoning samples by scene and roadside unit. """

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List
import torch
from torchvision import transforms

class ExtractSample:

    def __init__(self, scene: str, rsf: str, view: str,
                 images_dir: str, mask_dir: str,
                 snapshot_start: int, snapshot_end: int,
                 scene_prefix: str,
                 image_scene_format: str = None):

        self.scene = scene
        self.rsf = rsf
        self.view = view
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.snapshot_start = snapshot_start
        self.snapshot_end = snapshot_end
        self.scene_prefix = scene_prefix
        self.image_scene_format = image_scene_format

        if image_scene_format and 'Scene' in scene:
            scene_num = scene.replace('Scene', '')
            self.image_scene_name = image_scene_format.format(scene_num=scene_num)
        else:
            self.image_scene_name = scene

        self.seq_length = snapshot_end - snapshot_start + 1
        assert self.seq_length == 750, f"Sequence length must be750, Current: {self.seq_length}"

        self.mask = self._load_mask()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_mask(self) -> np.ndarray:

        mask_path = os.path.join(
            self.mask_dir,
            self.scene,
            f"{self.scene}_{self.rsf}_RGBmask.npy"
        )

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"MaskFile does not exist: {mask_path}")

        mask = np.load(mask_path)

        if mask.shape[0] != 750:
            raise ValueError(f"MaskWrong length: {mask.shape[0]}, Expectations750")

        return mask

    def _get_image_path(self, frame_idx: int) -> str:

        snapshot_num = self.snapshot_start + frame_idx

        filename = f"{self.image_scene_name}_{self.rsf}_snapshot{snapshot_num}_{self.view}_RGB.png"

        image_path = os.path.join(self.images_dir, filename)

        return image_path

    def load_images(self) -> Tuple[torch.Tensor, torch.Tensor]:

        images = []
        valid_mask = []

        for frame_idx in range(750):
            image_path = self._get_image_path(frame_idx)

            if not os.path.exists(image_path):
                images.append(torch.zeros(3, 224, 224))
                valid_mask.append(False)
            else:
                try:
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = self.transform(image)
                    images.append(image_tensor)
                    valid_mask.append(True)
                except Exception as e:
                    print(f"⚠️  Failed to load image: {image_path}, Error: {e}")
                    images.append(torch.zeros(3, 224, 224))
                    valid_mask.append(False)

        images_tensor = torch.stack(images)
        valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool)

        return images_tensor, valid_mask_tensor

    def get_info(self) -> Dict:

        return {
            'scene': self.scene,
            'rsf': self.rsf,
            'view': self.view,
            'seq_length': self.seq_length,
            'missing_frames': (~self.mask).sum(),
            'valid_frames': self.mask.sum()
        }

    def __repr__(self):
        info = self.get_info()
        return f"ExtractSample({info['scene']}/{info['rsf']}/{info['view']}, " \
               f"Missing Frame: {info['missing_frames']}/750)"

def scan_samples(mask_dir: str, views: List[str]) -> List[Tuple[str, str, str]]:

    samples = []
    mask_dir_path = Path(mask_dir)

    if not mask_dir_path.exists():
        raise FileNotFoundError(f"MaskDirectory does not exist: {mask_dir}")

    for scene_dir in sorted(mask_dir_path.iterdir()):
        if not scene_dir.is_dir():
            continue

        scene = scene_dir.name

        rsf_set = set()
        for mask_file in scene_dir.glob("*_RGBmask.npy"):
            filename = mask_file.stem
            parts = filename.rsplit('_', 2)

            if len(parts) >= 3:
                rsf = parts[-2]
                rsf_set.add(rsf)

        for rsf in sorted(rsf_set):
            for view in views:
                samples.append((scene, rsf, view))

    return samples

def create_extract_samples(dataset_config: Dict, views: List[str]) -> List[ExtractSample]:

    sample_tuples = scan_samples(dataset_config['mask_dir'], views)

    print(f"📊 Scan To {len(sample_tuples)} samples")

    samples = []
    for scene, rsf, view in sample_tuples:
        try:
            sample = ExtractSample(
                scene=scene,
                rsf=rsf,
                view=view,
                images_dir=dataset_config['images_dir'],
                mask_dir=dataset_config['mask_dir'],
                snapshot_start=dataset_config['snapshot_start'],
                snapshot_end=dataset_config['snapshot_end'],
                scene_prefix=dataset_config['scene_prefix'],
                image_scene_format=dataset_config.get('image_scene_format', None)
            )
            samples.append(sample)
        except Exception as e:
            print(f"⚠️  Failed to create sample: {scene}/{rsf}/{view}, Error: {e}")

    print(f"✅ Successfully created {len(samples)} samples")

    return samples
