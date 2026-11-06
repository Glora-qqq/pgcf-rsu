"""Random Seeds, Angle of view, Losses and checkpoint Read/Write. """

import os
import random
import numpy as np
import torch
import math
from typing import Dict, List, Tuple

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_view_angles(scene_type: str) -> Dict[str, float]:

    if scene_type == 'intersection':
        angles = {
            'left': -52.5 * math.pi / 180,
            'middle': 0.0,
            'right': 52.5 * math.pi / 180
        }
    else:
        angles = {
            'left': -90.0 * math.pi / 180,
            'middle': 0.0,
            'right': 90.0 * math.pi / 180
        }

    return angles

def create_sample_list(scenes_config: dict, feature_dir: str) -> List[Tuple[str, str, str]]:

    samples = []

    for scene_name, scene_config in scenes_config.items():
        scene_type = scene_config['scene_type']
        rsf_list = scene_config['rsf_list']

        scene_dir = os.path.join(feature_dir, scene_name)
        if not os.path.exists(scene_dir):
            print(f"⚠️  Scene directory does not exist: {scene_dir}")
            continue

        for rsf_name in rsf_list:
            rsf_dir = os.path.join(scene_dir, rsf_name)
            if not os.path.exists(rsf_dir):
                print(f"⚠️  RSFDirectory does not exist: {rsf_dir}")
                continue

            views_exist = all([
                os.path.exists(os.path.join(rsf_dir, f"{scene_name}_{rsf_name}_{view}_env_feature.npy"))
                for view in ['left', 'middle', 'right']
            ])

            if views_exist:
                samples.append((scene_name, rsf_name, scene_type))
            else:
                print(f"⚠️  Missing full perspective: {scene_name}/{rsf_name}")

    return samples

def split_train_val(samples: List, val_ratio: float = 0.15, seed: int = 42) -> Tuple[List, List]:

    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]

    return train_samples, val_samples

def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loss: float,
    val_loss: float,
    checkpoint_path: str,
    is_best: bool = False
):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss
    }

    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer=None, scheduler=None):

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('val_loss', float('inf'))

    return start_epoch, best_val_loss

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_temporal_consistency_loss(views: List[torch.Tensor]) -> torch.Tensor:

    left, middle, right = views

    def cosine_sim(x, y):
        x_norm = torch.nn.functional.normalize(x, dim=-1)
        y_norm = torch.nn.functional.normalize(y, dim=-1)
        return (x_norm * y_norm).sum(dim=-1).mean()

    sim_lm = cosine_sim(left, middle)
    sim_mr = cosine_sim(middle, right)
    sim_lr = cosine_sim(left, right)

    loss = -(sim_lm + sim_mr + sim_lr) / 3.0

    return loss

def compute_contrastive_loss(fused: torch.Tensor, views: List[torch.Tensor], temperature: float = 0.1) -> torch.Tensor:

    B, T, D = fused.shape

    fused_flat = fused.reshape(B * T, D)

    views_flat = torch.stack([v.reshape(B * T, D) for v in views], dim=0)

    fused_norm = torch.nn.functional.normalize(fused_flat, dim=-1)
    views_norm = torch.nn.functional.normalize(views_flat, dim=-1)

    views_concat = views_norm.reshape(-1, D)
    sim_matrix = torch.matmul(fused_norm, views_concat.t()) / temperature

    positive_mask = torch.zeros(B * T, 3 * B * T, device=fused.device)
    for i in range(B * T):
        positive_mask[i, i] = 1
        positive_mask[i, i + B * T] = 1
        positive_mask[i, i + 2 * B * T] = 1

    exp_sim = torch.exp(sim_matrix)
    positive_sim = (exp_sim * positive_mask).sum(dim=1)
    total_sim = exp_sim.sum(dim=1)

    loss = -torch.log(positive_sim / (total_sim + 1e-8)).mean()

    return loss

def format_time(seconds: float) -> str:

    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
