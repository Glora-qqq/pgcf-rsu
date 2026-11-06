"""冻结 WiFo 编码器，训练视觉投影和渐进门控。"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import time
import logging
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent))

from models import create_network, pretrained_weights
from data import SlidingWindowCSIDataset
import config as cfg

class Logger:

    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"="*70 + "\n")
            f.write("PGCF-RSU finetune log\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"="*70 + "\n\n")

    def log(self, message):

        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def __call__(self, message):

        self.log(message)

class GatedWiFoWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        if isinstance(outputs, tuple) and len(outputs) == 6:
            return outputs[:5]
        return outputs

def parse_args():
    parser = argparse.ArgumentParser(description='PGCF-RSU finetune')

    parser.add_argument('--epochs', type=int, default=cfg.TRAIN_CONFIG['num_epochs'],
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=cfg.TRAIN_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=cfg.TRAIN_CONFIG['learning_rate'],
                        help='学习率')
    parser.add_argument('--mask_ratio', type=float, default=cfg.MODEL_CONFIG['mask_ratio'],
                        help=f"WiFo Mask比例（当前配置: {cfg.MODEL_CONFIG['mask_ratio']}）")

    parser.add_argument('--model_size', type=str,
                        default=cfg.MODEL_CONFIG['model_size'],
                        choices=['tiny', 'little', 'small', 'base'])
    parser.add_argument('--pretrained_path', type=str,
                        default=str(cfg.WIFO_RGB_PRETRAIN_PATH))
    freeze_encoder = cfg.FREEZE_CONFIG['freeze_encoder']
    freeze_decoder = cfg.FREEZE_CONFIG['freeze_decoder']
    parser.add_argument('--freeze_encoder', action='store_true',
                        default=freeze_encoder,
                        help=f"冻结Encoder（当前配置: {freeze_encoder}）")
    parser.add_argument('--freeze_decoder_layers', type=int,
                        default=4 if freeze_decoder else 0,
                        help=f"冻结Decoder前N层（当前配置: {'全部冻结' if freeze_decoder else '不冻结'}）")

    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint继续训练的路径（例如: checkpoints/best_model.pth）')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='起始epoch（从checkpoint继续训练时使用）')

    parser.add_argument('--use_grouped_lr', action='store_true',
                        help='使用分组学习率（RGB模块用更高学习率）')
    parser.add_argument('--rgb_gate_lr_multiplier', type=float, default=20.0,
                        help='RGB Gate学习率倍数（默认20倍）')
    parser.add_argument('--rgb_proj_lr_multiplier', type=float, default=10.0,
                        help='RGB投影层学习率倍数（默认10倍）')

    parser.add_argument('--checkpoint_dir', type=str,
                        default=str(cfg.CHECKPOINT_ROOT),
                        help=f"检查点保存目录（默认: {cfg.CHECKPOINT_ROOT}）")

    parser.add_argument('--num_workers', type=int,
                        default=cfg.TRAIN_CONFIG['num_workers'])
    parser.add_argument('--device', type=str,
                        default=cfg.TRAIN_CONFIG['device'])

    return parser.parse_args()

def train_one_epoch(model, dataloader, optimizer, device, epoch, args):

    model.train()

    real_model = model
    if isinstance(model, nn.DataParallel):
        real_model = model.module
    if isinstance(real_model, GatedWiFoWrapper):
        real_model = real_model.model

    total_loss = 0.0
    total_loss_masked = 0.0
    total_loss_visible = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

    for batch_idx, batch in enumerate(pbar):
        csi = batch['csi'].to(device)
        rgb = batch['rgb_global'].to(device)
        mask = batch['mask'].to(device)

        loss_masked, loss_visible, pred_csi, target_csi, wifo_mask = model(
            csi,
            rgb_global=rgb,
            manual_mask=mask,
            mask_ratio=args.mask_ratio,
            add_noise=True,
            snr_db=20,
        )

        if loss_masked.dim() > 0:
            loss_masked = loss_masked.mean()
        if loss_visible.dim() > 0:
            loss_visible = loss_visible.mean()

        if args.mask_ratio == 0.0:
            loss = loss_visible
        else:
            loss = loss_masked

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_loss_masked += loss_masked.item()
        total_loss_visible += loss_visible.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.3e}",
            'loss_m': f"{loss_masked.item():.3e}",
            'loss_v': f"{loss_visible.item():.3e}",
        })

        if batch_idx % 100 == 0 and batch_idx > 0:
            gate_val = real_model.rgb_gate.item()
            tqdm.write(f"  [Batch {batch_idx}] RGB Gate: {gate_val:.4f}")

    avg_loss = total_loss / num_batches
    avg_loss_masked = total_loss_masked / num_batches
    avg_loss_visible = total_loss_visible / num_batches

    gate_val = real_model.rgb_gate.item()

    if hasattr(cfg, 'RESUME_CONFIG'):
        gate_target = cfg.RESUME_CONFIG.get('gate_target', 0.25)
        gate_warning = cfg.RESUME_CONFIG.get('gate_warning_threshold', 0.35)
        gate_critical = cfg.RESUME_CONFIG.get('gate_critical_threshold', 0.50)

        if gate_val >= gate_critical:
            status = f"⚠️  严重 (>{gate_critical}，RGB过度主导！)"
        elif gate_val >= gate_warning:
            status = f"⚠️  警告 (>{gate_warning}，接近过度主导)"
        elif gate_val >= gate_target * 0.8:
            status = f"✅ 接近目标 (目标:{gate_target})"
        else:
            status = "📈 增长中"
    else:
        status = ""

    print(f"\n📊 Epoch {epoch} - RGB Gate: {gate_val:.4f} {status}")

    return avg_loss, avg_loss_masked, avg_loss_visible

def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.LOG_ROOT).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    logger = Logger(log_file)

    logger(f"\n{'='*60}")
    logger("PGCF-RSU 微调")
    logger(f"{'='*60}")
    logger(f"{'='*60}\n")

    logger(f"✅ 输出目录已创建:")
    logger(f"   检查点: {checkpoint_dir}")
    logger(f"   日志: {log_file}")
    logger("")

    logger("\n" + "="*60)
    logger("完整配置检查")
    logger("="*60)
    logger(f"数据配置:")
    logger(f"  场景: {cfg.SCENES}")
    logger(f"  RSF配置: {cfg.SCENE_RSF_CONFIG}")
    logger(f"  窗口大小: {cfg.WINDOW_CONFIG['window_size']}")
    logger(f"  步长: {cfg.WINDOW_CONFIG['stride']}")
    logger(f"  最小有效帧: {cfg.WINDOW_CONFIG['min_valid_frames']}")
    logger(f"  RGB策略: {cfg.RGB_CONFIG['aggregation_strategy']}")
    logger(f"\n模型配置:")
    logger(f"  模型大小: {cfg.MODEL_CONFIG['model_size']}")
    logger(f"  Embed维度: {cfg.MODEL_CONFIG['embed_dim']}")
    logger(f"  Encoder层数: {cfg.MODEL_CONFIG['depth']}")
    logger(f"  Decoder层数: {cfg.MODEL_CONFIG['decoder_depth']}")
    logger(f"  WiFo Mask比例: {cfg.MODEL_CONFIG['mask_ratio']}")
    logger(f"\n训练配置:")
    logger(f"  Batch size: {cfg.TRAIN_CONFIG['batch_size']}")
    logger(f"  Epochs: {cfg.TRAIN_CONFIG['num_epochs']}")
    logger(f"  学习率: {cfg.TRAIN_CONFIG['learning_rate']}")
    logger(f"  Weight decay: {cfg.TRAIN_CONFIG['weight_decay']}")
    logger(f"  Num workers: {cfg.TRAIN_CONFIG['num_workers']}")
    logger(f"  GPU IDs: {cfg.TRAIN_CONFIG['gpu_ids']}")
    logger(f"\n冻结配置:")
    logger(f"  Freeze Encoder: {cfg.FREEZE_CONFIG['freeze_encoder']}")
    logger(f"  Freeze Decoder: {cfg.FREEZE_CONFIG['freeze_decoder']}")
    logger(f"  Freeze RGB Proj: {cfg.FREEZE_CONFIG['freeze_rgb_proj']}")
    logger("="*60 + "\n")

    logger("加载数据集...")
    dataset = SlidingWindowCSIDataset(
        csi_root=str(cfg.CSI_DATA_ROOT),
        mask_root=str(cfg.MASK_ROOT),
        rgb_feature_root=str(cfg.RGB_FEATURE_ROOT),
        scenes=cfg.SCENES,
        rsf_config=cfg.SCENE_RSF_CONFIG,
        window_size=cfg.WINDOW_CONFIG['window_size'],
        stride=cfg.WINDOW_CONFIG['stride'],
        min_valid_frames=cfg.WINDOW_CONFIG['min_valid_frames'],
        rgb_strategy=cfg.RGB_CONFIG['aggregation_strategy'],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger(f"✓ 数据集: {len(dataset)} 样本, {len(dataloader)} batches")

    logger("\n创建模型...")
    model = create_network(model_size=args.model_size)

    if Path(args.pretrained_path).exists():
        logger(f"加载预训练权重: {args.pretrained_path}")
        model, _ = pretrained_weights(
            model,
            args.pretrained_path,
            freeze_encoder=args.freeze_encoder,
            freeze_decoder_layers=args.freeze_decoder_layers,
            verbose=False
        )
    else:
        logger(f"⚠️  预训练权重未找到，使用随机初始化")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    gpu_ids = cfg.TRAIN_CONFIG.get('gpu_ids', [0])

    model = model.to(device)

    if torch.cuda.is_available() and len(gpu_ids) > 1:
        logger(f"✓ 使用多 GPU训练: {gpu_ids}")
        model = GatedWiFoWrapper(model)
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logger(f"  主 GPU: cuda:{gpu_ids[0]}")
        logger(f"  总 GPU数: {len(gpu_ids)}")
        logger(f"  ✅ 已使用wrapper兼容DataParallel")
    else:
        model = GatedWiFoWrapper(model)
        logger(f"✓ 模型已就绪 (设备: {device})")

    real_model = model
    if isinstance(model, nn.DataParallel):
        real_model = model.module
    if isinstance(real_model, GatedWiFoWrapper):
        real_model = real_model.model

    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger(f"✓ 可训练参数: {trainable_count:,} / {total_params:,} ({100*trainable_count/total_params:.1f}%)")

    if args.use_grouped_lr:
        param_groups = []

        param_groups.append({
            'params': [real_model.rgb_gate],
            'lr': args.lr * args.rgb_gate_lr_multiplier,
            'name': 'rgb_gate'
        })

        param_groups.append({
            'params': real_model.dinov2_proj.parameters(),
            'lr': args.lr * args.rgb_proj_lr_multiplier,
            'name': 'rgb_projection'
        })

        added_param_ids = set()
        for group in param_groups:
            if isinstance(group['params'], list):
                for p in group['params']:
                    added_param_ids.add(id(p))
            else:
                temp_list = list(group['params'])
                group['params'] = temp_list
                for p in temp_list:
                    added_param_ids.add(id(p))

        other_trainable = []
        for name, param in model.named_parameters():
            if param.requires_grad and id(param) not in added_param_ids:
                other_trainable.append(param)

        if other_trainable:
            param_groups.append({
                'params': other_trainable,
                'lr': args.lr,
                'name': 'other_trainable'
            })

        optimizer = optim.AdamW(
            param_groups,
            weight_decay=cfg.TRAIN_CONFIG['weight_decay']
        )

        logger(f"\n✓ 分组学习率配置:")
        logger(f"   RGB Gate:      {args.lr * args.rgb_gate_lr_multiplier:.2e} (基础lr × {args.rgb_gate_lr_multiplier:.0f})")
        logger(f"   RGB投影层:     {args.lr * args.rgb_proj_lr_multiplier:.2e} (基础lr × {args.rgb_proj_lr_multiplier:.0f})")
        logger(f"   其他参数:       {args.lr:.2e} (基础lr)")
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=cfg.TRAIN_CONFIG['weight_decay']
        )
        logger(f"\n✓ 统一学习率: {args.lr:.2e}")

    logger(f"\n{'='*60}")
    logger(f"训练配置")
    logger(f"{'='*60}")
    if len(gpu_ids) > 1:
        logger(f"  GPU配置: 多GPU并行 ({len(gpu_ids)}张卡)")
        logger(f"    - GPU IDs: {gpu_ids}")
        logger(f"    - 主GPU: cuda:{gpu_ids[0]}")
        logger(f"    - Loss自动平均到主GPU")
    else:
        logger(f"  GPU配置: 单GPU (cuda:{gpu_ids[0]})")
    logger(f"  WiFo Mask比例: {args.mask_ratio}")
    if args.mask_ratio == 0.0:
        logger(f"  优化目标: loss_visible (所有有效帧)")
        logger(f"  ✅ 不使用WiFo掩码，最大化数据利用")
    else:
        logger(f"  优化目标: loss_masked (被掩码部分)")
        logger(f"  ⚠️  使用WiFo掩码，{args.mask_ratio*100:.0f}%被掩码")
    logger(f"  学习率: {args.lr}")
    logger(f"  Weight decay: {cfg.TRAIN_CONFIG['weight_decay']}")
    logger(f"  Batch size: {args.batch_size} ({'总batch: ' + str(args.batch_size * len(gpu_ids)) if len(gpu_ids) > 1 else ''})")
    logger(f"  训练轮数: {args.epochs}")
    logger(f"  最小有效帧: {cfg.WINDOW_CONFIG['min_valid_frames']}")

    start_epoch = args.start_epoch
    best_loss = float('inf')
    gate_history = []

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger(f"\n{'='*60}")
            logger(f"从checkpoint继续训练")
            logger(f"{'='*60}")
            logger(f"加载checkpoint: {resume_path}")

            checkpoint = torch.load(resume_path, map_location=device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                real_model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger(f"✓ 已加载优化器状态")
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    logger(f"✓ 从epoch {start_epoch}继续")
                if 'best_loss' in checkpoint:
                    best_loss = checkpoint['best_loss']
                    logger(f"✓ 当前最佳loss: {best_loss:.4e}")
                if 'gate_history' in checkpoint:
                    gate_history = checkpoint['gate_history']
                    logger(f"✓ 已加载gate历史（{len(gate_history)}轮）")
                    logger(f"✓ 最后一轮gate值: {gate_history[-1]:.4f}")
            else:
                real_model.load_state_dict(checkpoint)
                logger(f"✓ 已加载模型权重")
                logger(f"⚠️  未找到优化器状态，使用新的优化器")

            logger(f"{'='*60}\n")
        else:
            logger(f"\n⚠️  Checkpoint未找到: {resume_path}")
            logger(f"   从头开始训练\n")

    logger(f"\n{'='*60}")
    logger(f"开始训练...")
    logger(f"{'='*60}")
    logger(f"起始epoch: {start_epoch}")
    logger(f"结束epoch: {args.epochs}")
    logger(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        avg_loss, avg_loss_masked, avg_loss_visible = train_one_epoch(
            model, dataloader, optimizer, device, epoch, args
        )

        epoch_time = time.time() - start_time

        logger(f"\nEpoch {epoch}/{args.epochs} 完成:")
        logger(f"  Loss: {avg_loss:.4e}")
        logger(f"  Loss (Masked): {avg_loss_masked:.4e}")
        logger(f"  Loss (Visible): {avg_loss_visible:.4e}")
        logger(f"  时间: {epoch_time:.1f}s")

        real_model = model.module.model if isinstance(model, nn.DataParallel) else model.model
        gate_val = real_model.rgb_gate.item()
        gate_history.append(gate_val)

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            save_model = model
            if isinstance(save_model, nn.DataParallel):
                save_model = save_model.module
            if isinstance(save_model, GatedWiFoWrapper):
                save_model = save_model.model

            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'gate_history': gate_history,
                'args': vars(args),
            }
            torch.save(checkpoint_dict, checkpoint_path)
            logger(f"  ✅ 最佳模型已保存 (loss: {best_loss:.4e}): {checkpoint_path}")

        logger("")

    logger(f"\n{'='*60}")
    logger(f"训练完成!")
    logger(f"{'='*60}")
    logger(f"最佳Loss: {best_loss:.4e}")

    final_gate = gate_history[-1]
    initial_gate = gate_history[0] if gate_history else 0.01
    logger(f"最终RGB Gate: {final_gate:.4f} (初始: {initial_gate:.4f})")

    if hasattr(cfg, 'RESUME_CONFIG'):
        gate_target = cfg.RESUME_CONFIG.get('gate_target', 0.25)
        gate_warning = cfg.RESUME_CONFIG.get('gate_warning_threshold', 0.35)
        gate_critical = cfg.RESUME_CONFIG.get('gate_critical_threshold', 0.50)

        if final_gate >= gate_critical:
            logger(f"⚠️  Gate状态: 严重 - RGB过度主导 (>{gate_critical})")
            logger(f"   建议: 降低学习率倍数或停止训练")
        elif final_gate >= gate_warning:
            logger(f"⚠️  Gate状态: 警告 - 接近过度主导 ({gate_warning}-{gate_critical})")
            logger(f"   建议: 密切监控，考虑降低学习率")
        elif final_gate >= gate_target * 0.8:
            logger(f"✅ Gate状态: 理想 - 达到平衡点 (~{gate_target})")
            logger(f"   RGB提供有效增强，但不主导CSI特征")
        else:
            logger(f"📈 Gate状态: 偏低 - 仍有增长空间 (<{gate_target*0.8:.2f})")
            logger(f"   建议: 可以继续训练或提高学习率")
    logger(f"模型保存: {checkpoint_dir}/best_model.pth")
    logger(f"日志保存: {log_file}")
    logger(f"{'='*60}\n")

if __name__ == '__main__':
    main()
