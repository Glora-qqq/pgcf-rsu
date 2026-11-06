"""Training Geometry Mass Fusion and GPS Guided Six-Dimensional Pose Correction. """

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import time
from tqdm import tqdm
import json
from datetime import datetime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Logger:

    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout

        self.log = open(self.log_file, 'w', encoding='utf-8')

        header = f"{'='*80}\n"
        header += f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"{'='*80}\n"
        self.log.write(header)
        self.log.flush()

    def write(self, message):

        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):

        self.terminal.flush()
        self.log.flush()

    def close(self):

        footer = f"\n{'='*80}\n"
        footer += f"End of training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        footer += f"{'='*80}\n"
        self.log.write(footer)
        self.log.close()
        sys.stdout = self.terminal

sys.path.append(str(Path(__file__).parent))

from config import TrainConfig, SCENE_RSF_CONFIG
from data.train_set import MultiRSFPose6DDataLoader
from models.quality import SpatialQualityFusion, SpatialQualityEstimator
from models.concat import ConcatProjCSIFusion
from models.temporal import TransformerTemporalModel
from models.regression import PoseHead, PoseEvaluator

class PoseFusionModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ablation_modality = str(getattr(config, 'ablation_modality', 'full') or 'full').lower()

        strategy = str(getattr(config, "csi_fusion_strategy", "dynamic") or "dynamic").lower()
        if strategy == "concat_proj":
            self.multi_rsf_fusion = ConcatProjCSIFusion(config)
            print("[INFO] Use ConcatProjCSIFusion (CSI Stitching→Projection 512, None 12 Dimension Quality Score) ")
        else:
            self.multi_rsf_fusion = SpatialQualityFusion(config)
            print("[INFO] Use space quality fusion")

        self.geo_quality = None
        self.geo_to_wifo = None
        if self.ablation_modality == 'gps_geo':
            hidden = int(getattr(config, 'geo_proj_hidden', 128) or 128)
            self.geo_quality = SpatialQualityEstimator(config)
            self.geo_to_wifo = nn.Sequential(
                nn.Linear(12, hidden),
                nn.ReLU(),
                nn.Linear(hidden, config.wifo_dim),
            )
            print(f"[INFO] ablation_modality=gps_geo: Geometric bypass geo_proj_hidden={hidden} (Disconnect CSI/RGB) ")
        elif self.ablation_modality == 'placebo_shuffle':
            print("[INFO] ablation_modality=placebo_shuffle: CSI/WiFo Time dimension disorder (GPS And 12 Dimension Geometry Immovable) ")
        elif self.ablation_modality in ('no_csi_completion', 'no_completion'):
            print("[INFO] ablation_modality=no_csi_completion: Original CSI Missing frame feature set to zero (dataloader Pinch off; RGB/Geometry/GPS Unchanged) ")

        self.temporal_model = TransformerTemporalModel(
            input_dim=config.wifo_dim,
            d_model=config.wifo_dim,
            nhead=config.temporal_nhead,
            num_layers=config.temporal_num_layers,
            dim_feedforward=config.temporal_hidden_dim * 2,
            dropout=config.temporal_dropout
        )

        self.pose_head = PoseHead(
            input_dim=config.wifo_dim,
            hidden_dim=config.position_hidden_dim,
            config=config
        )
        print("[INFO] UseMLPReturning Head (Supports motion-aware output) ")

    def _forward_gps_geo(self, rsf_data_dict, scene_names=None):

        keys = sorted(rsf_data_dict.keys())
        spatial = torch.stack(
            [rsf_data_dict[k]['spatial_features'] for k in keys], dim=2
        )
        quality_scores, interpretable_info = self.geo_quality(spatial, scene_names)
        weights = F.softmax(quality_scores.squeeze(-1), dim=-1)
        pooled = (weights.unsqueeze(-1) * spatial).sum(dim=2)
        fused = self.geo_to_wifo(pooled)
        interpretable_info = dict(interpretable_info or {})
        interpretable_info.update({
            'ablation_modality': 'gps_geo',
            'rsf_keys': keys,
            'attention_weights': weights,
            'quality_scores': quality_scores.squeeze(-1),
            'n_rsf': len(keys),
        })
        return fused, weights, interpretable_info

    @staticmethod
    def _placebo_shuffle_csi(rsf_data_dict, T):

        first = next(iter(rsf_data_dict.values()))
        device = first['csi_features'].device
        perm = torch.randperm(T, device=device)
        out = {}
        for k, v in rsf_data_dict.items():
            out[k] = {
                'csi_features': v['csi_features'][:, perm],
                'spatial_features': v['spatial_features'],
            }
            for kk, vv in v.items():
                if kk not in out[k]:
                    out[k][kk] = vv
        return out

    def forward(self, rsf_data_dict, gps_poses_6d, scene_name=None, vehicle_names=None):

        B, T = gps_poses_6d.shape[:2]

        if isinstance(scene_name, str):
            scene_names = [scene_name] * B
        elif isinstance(scene_name, list):
            scene_names = scene_name
        else:
            scene_names = None

        if self.ablation_modality == 'placebo_shuffle' and self.training:
            rsf_data_dict = self._placebo_shuffle_csi(rsf_data_dict, T)

        if self.ablation_modality == 'gps_geo':
            fused_csi_features, rsf_attention_weights, interpretable_info =\
                self._forward_gps_geo(rsf_data_dict, scene_names)
        else:
            fused_csi_features, rsf_attention_weights, interpretable_info =\
                self.multi_rsf_fusion(rsf_data_dict, scene_names)
            if self.ablation_modality == 'placebo_shuffle':
                interpretable_info = dict(interpretable_info or {})
                interpretable_info['ablation_modality'] = 'placebo_shuffle'

        temporal_features = self.temporal_model(fused_csi_features)

        if vehicle_names is not None and len(vehicle_names) > 0:
            self.config.current_vehicle = vehicle_names[0] if isinstance(vehicle_names, list) else vehicle_names

        regression_results = self.pose_head(
            temporal_features,
            gps_poses_6d,
            scene_name
        )

        regression_results['interpretable_info'] = interpretable_info
        regression_results['interpretable_info']['rsf_attention_weights'] = rsf_attention_weights

        return regression_results

class Trainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        self._set_seed(config.random_seed)

        self.model = PoseFusionModel(config).to(self.device)

        self.best_val_loss = float('inf')
        self.best_epoch = 0

        self.optimizer = self._create_optimizer()

        from utils.losses import MultiLevel6DPoseLoss

        self.criterion = MultiLevel6DPoseLoss(
            alpha=self.config.loss_alpha,
            omega=self.config.loss_omega,
            beta=self.config.loss_beta,
            gamma=self.config.loss_gamma,
            delta=self.config.loss_delta,
            position_loss_type=self.config.position_loss_type,
            orientation_loss_type=self.config.orientation_loss_type,
        )

        print("\n✅ UseMultiLevel6DPoseLossLoss function (5losses) ")
        print(f"  - Location (α={self.config.loss_alpha}) | Direction (ω={self.config.loss_omega}) | GPSCorrection (β={self.config.loss_beta})")
        print(f"  - Smoothing (γ={self.config.loss_gamma}) | Confidence (δ={self.config.loss_delta})")

        if getattr(self.config, 'use_output_adaptive_filter', False):
            from utils.adaptive_filter import VehicleAdaptiveOutputFilter
            self.output_filter = VehicleAdaptiveOutputFilter(self.config)
            print("\n✅ Output Vehicle Adaptive Filtering Enabled")
            print(f"  - CarVehicle: xyFor2D Kalman (Q={self.config.output_car_kalman_process_noise}, R={self.config.output_car_kalman_measurement_noise})")
            print(f"  - ATVehicle: xyzFor3D UKF (Q={self.config.output_at_ukf_process_noise}, R={self.config.output_at_ukf_measurement_noise})")
        else:
            self.output_filter = None
            print("\n❌ Output filtering is off")

        self.evaluator = PoseEvaluator()

        self.scheduler = self._create_scheduler()

        from config import LOG_ROOT
        self.run_log_dir = LOG_ROOT / f"train_{time.strftime('%Y%m%d_%H%M%S')}"
        self.run_log_dir.mkdir(parents=True, exist_ok=True)
        self.run_ckpt_dir = self.run_log_dir / 'checkpoints'
        self.run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.train_history = {'losses': [], 'metrics': []}

        from config import CHECKPOINT_ROOT, LOG_ROOT
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"\n[Random Seeds] Set to {seed} (Ensure the training is reproducible) ")

    def _create_optimizer(self):

        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            print(f"✅ UseAdamOptimizer: lr={self.config.learning_rate}, weight_decay={self.config.weight_decay}")
        elif self.config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            print(f"✅ UseAdamWOptimizer: lr={self.config.learning_rate}, weight_decay={self.config.weight_decay}")
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        return optimizer

    def _create_scheduler(self):

        if self.config.lr_scheduler is None:
            print(f"⚪ Use fixed learning rate: {self.config.learning_rate}")
            return None

        if self.config.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.cosine_min_lr
            )
            print(f"✅ Learning rate using cosine annealing: {self.config.learning_rate} → {self.config.cosine_min_lr}")
            return scheduler

        elif self.config.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.5
            )
            print(f"✅ Use tiered learning rates: Every50Roundx0.5")
            return scheduler

        elif self.config.lr_scheduler == 'warmup_cosine':
            def warmup_cosine_schedule(epoch):
                if epoch < self.config.cosine_warmup_epochs:
                    return (epoch + 1) / self.config.cosine_warmup_epochs
                else:
                    progress = (epoch - self.config.cosine_warmup_epochs) /\
                              (self.config.num_epochs - self.config.cosine_warmup_epochs)
                    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                    min_lr_ratio = self.config.cosine_min_lr / self.config.learning_rate
                    return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_cosine_schedule)
            print(f"✅ Use preheat+Cosine annealing learning rate:")
            print(f"   Warmup: Epoch 0-{self.config.cosine_warmup_epochs} (0 → {self.config.learning_rate:.2e})")
            print(f"   Cosine: Epoch {self.config.cosine_warmup_epochs}-{self.config.num_epochs} ({self.config.learning_rate:.2e} → {self.config.cosine_min_lr:.2e})")
            return scheduler
        else:
            raise ValueError(f"Unsupported Learning Rate Scheduler: {self.config.lr_scheduler}")

    def load_checkpoint(self, checkpoint_path):

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

        print(f"\n{'='*80}")
        print(f"💾 Load checkpoint: {checkpoint_path}")
        print(f"{'='*80}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model weight loaded (epoch {checkpoint['epoch']}) ")

        if self.config.resume_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ Optimizer status loaded")
        else:
            print(f"⚠️  Use new optimizer state (Learning Rate: {self.config.learning_rate}) ")

        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
            print(f"✅ Training history loaded ({len(self.train_history.get('losses', []))} epochs) ")

        start_epoch = checkpoint['epoch'] + 1
        print(f"📍 will start from Epoch {start_epoch} Continue training")
        print(f"{'='*80}\n")

        return start_epoch

    def train_epoch(self, train_loader, epoch):

        self.model.train()

        total_loss = 0.0
        num_batches = 0

        accumulated_loss_dict = {
            'position': 0.0,
            'orientation': 0.0,
            'gps_correction': 0.0,
            'smooth': 0.0,
            'confidence_reg': 0.0,
            'pos_x': 0.0,
            'pos_y': 0.0,
            'pos_z': 0.0,
        }

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            rsf_data_dict = {}
            for rsf_name, rsf_data in batch['rsf_data_dict'].items():
                rsf_data_dict[rsf_name] = {
                    'csi_features': rsf_data['csi_features'].to(self.device),
                    'spatial_features': rsf_data['spatial_features'].to(self.device)
                }

            gt_poses_6d = batch['gt_poses_6d'].to(self.device)
            gps_poses_6d = batch['gps_poses_6d'].to(self.device)
            scene_name = batch['scene_name']
            vehicle_names = batch['vehicle_names']
            frame_ids = batch.get('frame_ids', None)

            self.optimizer.zero_grad()
            outputs = self.model(rsf_data_dict, gps_poses_6d, scene_name, vehicle_names)

            predicted_poses = outputs['corrected_poses_6d']
            correction_deltas = outputs['correction_deltas']
            confidences = outputs['confidences']

            if hasattr(self, 'output_filter') and self.output_filter is not None:
                predicted_poses = self.output_filter.apply(predicted_poses, vehicle_names)

            total_loss_value, loss_dict = self.criterion(
                pred_poses_6d=predicted_poses,
                gt_poses_6d=gt_poses_6d,
                gps_poses_6d=gps_poses_6d,
                correction_delta=correction_deltas,
                confidence=confidences
            )

            total_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip_norm
            )
            self.optimizer.step()

            total_loss += total_loss_value.item()
            num_batches += 1

            for key in accumulated_loss_dict.keys():
                accumulated_loss_dict[key] += loss_dict.get(key, 0)

            postfix_dict = {
                'Total': f'{total_loss_value.item():.4f}',
                'Pos': f'{loss_dict.get("position", 0.0):.4f}',
                'X': f'{loss_dict.get("pos_x", 0.0):.3f}',
                'Y': f'{loss_dict.get("pos_y", 0.0):.3f}',
                'Z': f'{loss_dict.get("pos_z", 0.0):.3f}',
                'Ori': f'{loss_dict.get("orientation", 0.0):.3f}',
                'Avg': f'{total_loss/num_batches:.4f}'
            }
            pbar.set_postfix(postfix_dict)

        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_loss_dict = {k: v / num_batches if num_batches > 0 else 0.0
                        for k, v in accumulated_loss_dict.items()}

        return avg_total_loss, avg_loss_dict

    def validate(self, val_loader, epoch):

        self.model.eval()

        total_loss = 0.0
        all_metrics = []
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                if batch is None:
                    continue

                rsf_data_dict = {}
                for rsf_name, rsf_data in batch['rsf_data_dict'].items():
                    rsf_data_dict[rsf_name] = {
                        'csi_features': rsf_data['csi_features'].to(self.device),
                        'spatial_features': rsf_data['spatial_features'].to(self.device)
                    }

                gt_poses_6d = batch['gt_poses_6d'].to(self.device)
                gps_poses_6d = batch['gps_poses_6d'].to(self.device)
                scene_name = batch['scene_name']
                vehicle_names = batch['vehicle_names']

                outputs = self.model(rsf_data_dict, gps_poses_6d, scene_name, vehicle_names)

                predicted_poses = outputs['corrected_poses_6d']

                if hasattr(self, 'output_filter') and self.output_filter is not None:
                    predicted_poses = self.output_filter.apply(predicted_poses, vehicle_names)

                loss_value, _ = self.criterion(
                    pred_poses_6d=predicted_poses,
                    gt_poses_6d=gt_poses_6d,
                    gps_poses_6d=gps_poses_6d,
                    correction_delta=outputs['correction_deltas'],
                    confidence=outputs['confidences']
                )

                total_loss += loss_value.item()

                metrics = self.evaluator.compute_pose_errors(
                    predicted_poses, gt_poses_6d
                )

                gps_improvement = self.evaluator.compute_gps_improvement(
                    predicted_poses, gt_poses_6d, gps_poses_6d
                )

                metrics.update(gps_improvement)

                if 'smoothed_poses_6d' in outputs:
                    smoothed_metrics = self.evaluator.compute_pose_errors(
                        outputs['smoothed_poses_6d'], gt_poses_6d
                    )
                    smoothed_improvement = self.evaluator.compute_gps_improvement(
                        outputs['smoothed_poses_6d'], gt_poses_6d, gps_poses_6d
                    )

                    for key, val in smoothed_metrics.items():
                        metrics[f'kalman_{key}'] = val
                    for key, val in smoothed_improvement.items():
                        metrics[f'kalman_{key}'] = val

                all_metrics.append(metrics)
                num_batches += 1

        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return avg_loss, avg_metrics

    def train(self, train_loader, val_loader=None):

        print("Start PGCF-RSU Positioning training")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Equipment: {self.device}")
        print(f"Number of training rounds: {self.config.num_epochs}")

        start_epoch = 1
        if self.config.resume_from is not None:
            try:
                start_epoch = self.load_checkpoint(self.config.resume_from)
                print(f"✅ From Epoch {start_epoch} Resume training\n")
            except Exception as e:
                print(f"❌ LoadcheckpointFailed: {e}")
                print(f"⚠️  Will start training from scratch\n")
                import traceback
                traceback.print_exc()
                start_epoch = 1

        best_val_loss = float('inf')

        for epoch in range(start_epoch, start_epoch + self.config.num_epochs):
            current_training_epoch = epoch - start_epoch + 1

            train_loss, loss_dict = self.train_epoch(train_loader, epoch)
            self.train_history['losses'].append(train_loss)

            print(f"\nEpoch {current_training_epoch}/{self.config.num_epochs} (Total Rounds: {epoch})")
            print(f"Training Losses: {train_loss:.6f}")
            print(f"  - Loss of position (Total): {loss_dict.get('position', 0.0):.4f}")
            print(f"    └─ X: {loss_dict.get('pos_x', 0.0):.4f} | Y: {loss_dict.get('pos_y', 0.0):.4f} | Z: {loss_dict.get('pos_z', 0.0):.4f}")
            print(f"  - Direction Loss: {loss_dict.get('orientation', 0.0):.4f} | GPSCorrection: {loss_dict.get('gps_correction', 0.0):.4f}")
            print(f"  - Smooth Loss: {loss_dict.get('smooth', 0.0):.4f} | Confidence: {loss_dict.get('confidence_reg', 0.0):.4f}")

            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"📉 Current Learning Rate: {current_lr:.2e}")

            current_loss = train_loss
            if val_loader is not None:
                val_loss, val_metrics = self.validate(val_loader, epoch)
                self.train_history['metrics'].append(val_metrics)
                current_loss = val_loss

                print(f"Validate loss: {val_loss:.6f}")
                print(f"LocationMAE: {val_metrics.get('position_mae', 0):.3f}m")
                print(f"GPSImprovement: {val_metrics.get('improvement_ratio', 0):.1%}")

                if getattr(self.config, 'use_output_kalman', False) and 'kalman_position_mae' in val_metrics:
                    print(f"  [Kalman filtering] LocationMAE: {val_metrics.get('kalman_position_mae', 0):.3f}m")
                    print(f"  [Kalman filtering] GPSImprovement: {val_metrics.get('kalman_improvement_ratio', 0):.1%}")

            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                self.best_epoch = epoch
                best_model_path = self.run_ckpt_dir / 'pose_best.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': current_loss if val_loader else None,
                    'best_loss': self.best_val_loss,
                    'spatial_feature_mask': getattr(self.config, 'spatial_feature_mask', None),
                    'csi_fusion_strategy': getattr(self.config, 'csi_fusion_strategy', None),
                    'loss_weights': {
                        'alpha': self.config.loss_alpha,
                        'omega': self.config.loss_omega,
                        'beta': self.config.loss_beta,
                        'gamma': self.config.loss_gamma,
                        'delta': self.config.loss_delta,
                    },
                }, best_model_path)
                print(f"[OK] Save the best model (Epoch {epoch}, Loss={current_loss:.4f}): {best_model_path}")

            if epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch, 'regular')

        print("Training Complete")

    def _save_checkpoint(self, epoch, checkpoint_type):

        checkpoint_dir = Path(getattr(self, 'run_ckpt_dir', Path('./checkpoints')))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'train_history': self.train_history
        }

        if checkpoint_type == 'best':
            path = checkpoint_dir / 'pose_best.pt'
        else:
            path = checkpoint_dir / f'pose_epoch_{epoch}.pt'

        torch.save(checkpoint, path)
        if self.config.verbose:
            print(f"Save Checkpoint: {path}")

def create_data_loaders(config):

    print("Create6Dimension Pose Data Loader...")

    from utils.constants import SCENE_RSF_CONFIG
    rsf_config = SCENE_RSF_CONFIG

    from config import DATA_ROOT
    data_root = DATA_ROOT

    train_loader = MultiRSFPose6DDataLoader(
        data_root=data_root,
        scenes=config.scenes,
        rsf_config=rsf_config,
        split='train',
        batch_size=1,
        time_window=config.time_window,
        stride=config.stride,
        num_workers=config.num_workers,
        config=config,
        use_input_kalman=getattr(config, 'use_input_kalman', False),
        input_kalman_config={
            'process_noise': getattr(config, 'input_kalman_process_noise', 0.10),
            'measurement_noise': getattr(config, 'input_kalman_measurement_noise', 1.6),
        }
    )

    val_loader = None

    return train_loader, val_loader

def main():

    config = TrainConfig()

    from config import LOG_ROOT, CHECKPOINT_ROOT
    log_file = LOG_ROOT / 'training_log.txt'
    logger = Logger(log_file)
    sys.stdout = logger
    print(f"✅ Logging Enabled: {log_file}")

    print("\n" + "="*80)
    print("Training Configuration")
    print("="*80)
    print(f"Scenario: {config.scenes}")
    print(f"Time window: {config.time_window} (None=Full trajectory)")
    print(f"BatchSize: {config.batch_size}")
    print(f"Number of training rounds: {config.num_epochs}")
    print(f"6dimensional posture: {config.use_6d_pose}")
    print(f"Position Correction Upper Limit: {config.max_position_correction}m")
    print(f"Upper Posture Correction Limit: {config.max_orientation_correction} rad ({config.max_orientation_correction*57.3:.1f}°)")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Equipment: {config.device if config.device else 'Automatic selection'}")
    print("-" * 80)
    print("🔬 Vehicle type adaptive Kalman filtering strategy v2.0")
    print("-" * 80)

    print(f"\n📥 Input vehicle adaptive filtering (Data loading phase, GPSPretreatment) :")
    if getattr(config, 'use_adaptive_kalman', True):
        print(f"  ✅ Enabled (Vehicle type adaptive) ")

        if getattr(config, 'use_input_kalman', False):
            print(f"     - CarVehicle: 2D KalmanFilter (xy) ")
            print(f"       · process_noise={getattr(config, 'input_kalman_process_noise', 0.10)}")
            print(f"       · measurement_noise={getattr(config, 'input_kalman_measurement_noise', 1.6)}")

        if getattr(config, 'use_at_ukf', False):
            print(f"     - ATVehicle: 3D UKFFilter (xyz) ✅")
            print(f"       · process_noise: {getattr(config, 'at_ukf_process_noise', 0.1)}")
            print(f"       · measurement_noise: {getattr(config, 'at_ukf_measurement_noise', 1.6)}")
            print(f"       · dt: {getattr(config, 'at_ukf_dt', 0.05)}Seconds@20Hz")
            if getattr(config, 'at_ukf_orientation', False):
                print(f"       · Attitude filtering: Enable")
            else:
                print(f"       · Attitude filtering: Close (Filter onlyxyz) ")
        else:
            print(f"     - ATVehicle: No filtering (Use originalGPS) ")
    else:
        print(f"  ❌ Closed")

    print(f"\n📤 Output Vehicle Adaptive Filtering (After model prediction, Before loss calculation) :")
    if getattr(config, 'use_output_adaptive_filter', False):
        print(f"  ✅ Enabled")
        print(f"     - CarVehicle: xyFor2D Kalman (Q={getattr(config, 'output_car_kalman_process_noise', 0.10)}, R={getattr(config, 'output_car_kalman_measurement_noise', 1.6)})")
        print(f"     - ATVehicle: xyzFor3D UKF (Q={getattr(config, 'output_at_ukf_process_noise', 0.10)}, R={getattr(config, 'output_at_ukf_measurement_noise', 0.8)})")
    else:
        print(f"  ❌ Closed (Loss calculation using model output directly, Gradient acts directly on the model) ")
    print("="*80 + "\n")

    try:
        train_loader, val_loader = create_data_loaders(config)

        trainer = Trainer(config)

        trainer.train(train_loader, val_loader)

    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
