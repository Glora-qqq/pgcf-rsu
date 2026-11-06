"""训练几何质量融合和 GPS 引导的六维位姿修正。"""

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
        header += f"训练日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
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
        footer += f"训练结束 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
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
            print("[INFO] 使用 ConcatProjCSIFusion（CSI 拼接→投影 512，无 12 维质量评分）")
        else:
            self.multi_rsf_fusion = SpatialQualityFusion(config)
            print("[INFO] 使用空间质量融合")

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
            print(f"[INFO] ablation_modality=gps_geo：几何旁路 geo_proj_hidden={hidden}（切断 CSI/RGB）")
        elif self.ablation_modality == 'placebo_shuffle':
            print("[INFO] ablation_modality=placebo_shuffle：CSI/WiFo 时间维乱序（GPS 与 12 维几何不动）")
        elif self.ablation_modality in ('no_csi_completion', 'no_completion'):
            print("[INFO] ablation_modality=no_csi_completion：原始 CSI 缺失帧特征置零（dataloader 掐断；RGB/几何/GPS 不变）")

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
        print("[INFO] 使用MLP回归头（支持运动感知输出）")

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

        print("\n✅ 使用MultiLevel6DPoseLoss损失函数（5个损失）")
        print(f"  - 位置 (α={self.config.loss_alpha}) | 方向 (ω={self.config.loss_omega}) | GPS修正 (β={self.config.loss_beta})")
        print(f"  - 平滑 (γ={self.config.loss_gamma}) | 置信度 (δ={self.config.loss_delta})")

        if getattr(self.config, 'use_output_adaptive_filter', False):
            from utils.adaptive_filter import VehicleAdaptiveOutputFilter
            self.output_filter = VehicleAdaptiveOutputFilter(self.config)
            print("\n✅ 输出端车辆自适应滤波已启用")
            print(f"  - Car车辆: xy用2D Kalman (Q={self.config.output_car_kalman_process_noise}, R={self.config.output_car_kalman_measurement_noise})")
            print(f"  - AT车辆: xyz用3D UKF (Q={self.config.output_at_ukf_process_noise}, R={self.config.output_at_ukf_measurement_noise})")
        else:
            self.output_filter = None
            print("\n❌ 输出端滤波已关闭")

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
        print(f"\n[随机种子] 已设置为 {seed}（确保训练可复现）")

    def _create_optimizer(self):

        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            print(f"✅ 使用Adam优化器: lr={self.config.learning_rate}, weight_decay={self.config.weight_decay}")
        elif self.config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            print(f"✅ 使用AdamW优化器: lr={self.config.learning_rate}, weight_decay={self.config.weight_decay}")
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")

        return optimizer

    def _create_scheduler(self):

        if self.config.lr_scheduler is None:
            print(f"⚪ 使用固定学习率: {self.config.learning_rate}")
            return None

        if self.config.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.cosine_min_lr
            )
            print(f"✅ 使用余弦退火学习率: {self.config.learning_rate} → {self.config.cosine_min_lr}")
            return scheduler

        elif self.config.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.5
            )
            print(f"✅ 使用阶梯式学习率: 每50轮×0.5")
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
            print(f"✅ 使用预热+余弦退火学习率:")
            print(f"   Warmup: Epoch 0-{self.config.cosine_warmup_epochs} (0 → {self.config.learning_rate:.2e})")
            print(f"   Cosine: Epoch {self.config.cosine_warmup_epochs}-{self.config.num_epochs} ({self.config.learning_rate:.2e} → {self.config.cosine_min_lr:.2e})")
            return scheduler
        else:
            raise ValueError(f"不支持的学习率调度器: {self.config.lr_scheduler}")

    def load_checkpoint(self, checkpoint_path):

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        print(f"\n{'='*80}")
        print(f"💾 加载检查点: {checkpoint_path}")
        print(f"{'='*80}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 模型权重已加载（epoch {checkpoint['epoch']}）")

        if self.config.resume_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ 优化器状态已加载")
        else:
            print(f"⚠️  使用新的优化器状态（学习率: {self.config.learning_rate}）")

        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
            print(f"✅ 训练历史已加载（{len(self.train_history.get('losses', []))} epochs）")

        start_epoch = checkpoint['epoch'] + 1
        print(f"📍 将从 Epoch {start_epoch} 继续训练")
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

        print("开始 PGCF-RSU 定位训练")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"设备: {self.device}")
        print(f"训练轮数: {self.config.num_epochs}")

        start_epoch = 1
        if self.config.resume_from is not None:
            try:
                start_epoch = self.load_checkpoint(self.config.resume_from)
                print(f"✅ 从 Epoch {start_epoch} 恢复训练\n")
            except Exception as e:
                print(f"❌ 加载checkpoint失败: {e}")
                print(f"⚠️  将从头开始训练\n")
                import traceback
                traceback.print_exc()
                start_epoch = 1

        best_val_loss = float('inf')

        for epoch in range(start_epoch, start_epoch + self.config.num_epochs):
            current_training_epoch = epoch - start_epoch + 1

            train_loss, loss_dict = self.train_epoch(train_loader, epoch)
            self.train_history['losses'].append(train_loss)

            print(f"\nEpoch {current_training_epoch}/{self.config.num_epochs} (总轮次: {epoch})")
            print(f"训练损失: {train_loss:.6f}")
            print(f"  - 位置损失 (总): {loss_dict.get('position', 0.0):.4f}")
            print(f"    └─ X: {loss_dict.get('pos_x', 0.0):.4f} | Y: {loss_dict.get('pos_y', 0.0):.4f} | Z: {loss_dict.get('pos_z', 0.0):.4f}")
            print(f"  - 方向损失: {loss_dict.get('orientation', 0.0):.4f} | GPS修正: {loss_dict.get('gps_correction', 0.0):.4f}")
            print(f"  - 平滑损失: {loss_dict.get('smooth', 0.0):.4f} | 置信度: {loss_dict.get('confidence_reg', 0.0):.4f}")

            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"📉 当前学习率: {current_lr:.2e}")

            current_loss = train_loss
            if val_loader is not None:
                val_loss, val_metrics = self.validate(val_loader, epoch)
                self.train_history['metrics'].append(val_metrics)
                current_loss = val_loss

                print(f"验证损失: {val_loss:.6f}")
                print(f"位置MAE: {val_metrics.get('position_mae', 0):.3f}m")
                print(f"GPS改进: {val_metrics.get('improvement_ratio', 0):.1%}")

                if getattr(self.config, 'use_output_kalman', False) and 'kalman_position_mae' in val_metrics:
                    print(f"  [卡尔曼滤波] 位置MAE: {val_metrics.get('kalman_position_mae', 0):.3f}m")
                    print(f"  [卡尔曼滤波] GPS改进: {val_metrics.get('kalman_improvement_ratio', 0):.1%}")

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
                print(f"[OK] 保存最佳模型 (Epoch {epoch}, Loss={current_loss:.4f}): {best_model_path}")

            if epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch, 'regular')

        print("训练完成")

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
            print(f"保存检查点: {path}")

def create_data_loaders(config):

    print("创建6维姿态数据加载器...")

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
    print(f"✅ 日志记录已启用: {log_file}")

    print("\n" + "="*80)
    print("训练配置")
    print("="*80)
    print(f"场景: {config.scenes}")
    print(f"时间窗口: {config.time_window} (None=完整轨迹)")
    print(f"Batch大小: {config.batch_size}")
    print(f"训练轮数: {config.num_epochs}")
    print(f"6维姿态: {config.use_6d_pose}")
    print(f"位置修正上限: {config.max_position_correction}m")
    print(f"姿态修正上限: {config.max_orientation_correction} rad ({config.max_orientation_correction*57.3:.1f}°)")
    print(f"学习率: {config.learning_rate}")
    print(f"设备: {config.device if config.device else '自动选择'}")
    print("-" * 80)
    print("🔬 车辆类型自适应卡尔曼滤波策略 v2.0")
    print("-" * 80)

    print(f"\n📥 输入端车辆自适应滤波（数据加载阶段，GPS预处理）:")
    if getattr(config, 'use_adaptive_kalman', True):
        print(f"  ✅ 已启用（车辆类型自适应）")

        if getattr(config, 'use_input_kalman', False):
            print(f"     - Car车辆：2D Kalman滤波（xy）")
            print(f"       · process_noise={getattr(config, 'input_kalman_process_noise', 0.10)}")
            print(f"       · measurement_noise={getattr(config, 'input_kalman_measurement_noise', 1.6)}")

        if getattr(config, 'use_at_ukf', False):
            print(f"     - AT车辆：3D UKF滤波（xyz）✅")
            print(f"       · process_noise: {getattr(config, 'at_ukf_process_noise', 0.1)}")
            print(f"       · measurement_noise: {getattr(config, 'at_ukf_measurement_noise', 1.6)}")
            print(f"       · dt: {getattr(config, 'at_ukf_dt', 0.05)}秒@20Hz")
            if getattr(config, 'at_ukf_orientation', False):
                print(f"       · 姿态滤波: 启用")
            else:
                print(f"       · 姿态滤波: 关闭（只滤波xyz）")
        else:
            print(f"     - AT车辆：无滤波（使用原始GPS）")
    else:
        print(f"  ❌ 已关闭")

    print(f"\n📤 输出端车辆自适应滤波（模型预测后，损失计算前）:")
    if getattr(config, 'use_output_adaptive_filter', False):
        print(f"  ✅ 已启用")
        print(f"     - Car车辆: xy用2D Kalman (Q={getattr(config, 'output_car_kalman_process_noise', 0.10)}, R={getattr(config, 'output_car_kalman_measurement_noise', 1.6)})")
        print(f"     - AT车辆: xyz用3D UKF (Q={getattr(config, 'output_at_ukf_process_noise', 0.10)}, R={getattr(config, 'output_at_ukf_measurement_noise', 0.8)})")
    else:
        print(f"  ❌ 已关闭（直接使用模型输出进行损失计算，梯度直接作用于模型）")
    print("="*80 + "\n")

    try:
        train_loader, val_loader = create_data_loaders(config)

        trainer = Trainer(config)

        trainer.train(train_loader, val_loader)

    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
