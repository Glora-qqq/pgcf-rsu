"""Provides paired feature samples for triple vision fusion. """

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

class MultiViewFusionDataset(Dataset):

    def __init__(
        self,
        feature_dir: str,
        samples: List[Tuple[str, str, str]],
        views: List[str] = ['left', 'middle', 'right']
    ):

        self.feature_dir = feature_dir
        self.samples = samples
        self.views = views

        print(f"Initialize Dataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], str, str, str]:

        scene_name, rsf_name, scene_type = self.samples[idx]

        view_features = []
        for view in self.views:
            filename = f"{scene_name}_{rsf_name}_{view}_env_feature.npy"
            filepath = os.path.join(self.feature_dir, scene_name, rsf_name, filename)

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Feature file does not exist: {filepath}")

            feature = np.load(filepath)
            feature_tensor = torch.from_numpy(feature).float()
            view_features.append(feature_tensor)

        return view_features, scene_name, rsf_name, scene_type

    def get_sample_info(self, idx: int) -> Dict:

        scene_name, rsf_name, scene_type = self.samples[idx]
        return {
            'scene': scene_name,
            'rsf': rsf_name,
            'scene_type': scene_type
        }

def collate_fn(batch):

    view_features_list = [item[0] for item in batch]
    scene_names = [item[1] for item in batch]
    rsf_names = [item[2] for item in batch]
    scene_types = [item[3] for item in batch]

    num_views = len(view_features_list[0])
    batched_views = []

    for view_idx in range(num_views):
        view_batch = [item[view_idx] for item in view_features_list]
        view_tensor = torch.stack(view_batch, dim=0)
        batched_views.append(view_tensor)

    return batched_views, scene_names, rsf_names, scene_types

def create_dataloaders(
    train_samples: List,
    val_samples: List,
    feature_dir: str,
    batch_size: int = 4,
    num_workers: int = 0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    train_dataset = MultiViewFusionDataset(
        feature_dir=feature_dir,
        samples=train_samples
    )

    val_dataset = MultiViewFusionDataset(
        feature_dir=feature_dir,
        samples=val_samples
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"\nData Loader Creation Complete:")
    print(f"  Training Samples: {len(train_dataset)}")
    print(f"  Validate Sample: {len(val_dataset)}")
    print(f"  Batch Size: {batch_size}")

    return train_loader, val_loader
