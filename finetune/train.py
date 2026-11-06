"""Freeze WiFo Encoder, Train Visual Projection and Progressive Gating. """

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
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
                        help='Number of training rounds')
    parser.add_argument('--batch_size', type=int, default=cfg.TRAIN_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=cfg.TRAIN_CONFIG['learning_rate'],
                        help='Learning Rate')
    parser.add_argument('--mask_ratio', type=float, default=cfg.MODEL_CONFIG['mask_ratio'],
                        help=f"WiFo MaskRatio (Current Configuration: {cfg.MODEL_CONFIG['mask_ratio']}) ")

    parser.add_argument('--model_size', type=str,
                        default=cfg.MODEL_CONFIG['model_size'],
                        choices=['tiny', 'little', 'small', 'base'])
    parser.add_argument('--pretrained_path', type=str,
                        default=str(cfg.WIFO_RGB_PRETRAIN_PATH))
    freeze_encoder = cfg.FREEZE_CONFIG['freeze_encoder']
    freeze_decoder = cfg.FREEZE_CONFIG['freeze_decoder']
    parser.add_argument('--freeze_encoder', action='store_true',
                        default=freeze_encoder,
                        help=f"FreezeEncoder (Current Configuration: {freeze_encoder}) ")
    parser.add_argument('--freeze_decoder_layers', type=int,
                        default=4 if freeze_decoder else 0,
                        help=f"FreezeDecoderagoNLayer (Current Configuration: {'Freeze All' if freeze_decoder else 'Do not freeze'}) ")

    parser.add_argument('--resume', type=str, default=None,
                        help='FromcheckpointPath to continue training (Example: checkpoints/best_model.pth) ')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='Startepoch (FromcheckpointUse when resuming training) ')

    parser.add_argument('--use_grouped_lr', action='store_true',
                        help='Use group learning rate (RGBHigher learning rate for modules) ')
    parser.add_argument('--rgb_gate_lr_multiplier', type=float, default=20.0,
                        help='RGB GateLearning Rate Multiplier (Default20times) ')
    parser.add_argument('--rgb_proj_lr_multiplier', type=float, default=10.0,
                        help='RGBProjection Layer Learning Rate Multiple (Default10times) ')

    parser.add_argument('--checkpoint_dir', type=str,
                        default=str(cfg.CHECKPOINT_ROOT),
                        help=f"Checkpoint Save Directory (Default: {cfg.CHECKPOINT_ROOT}) ")

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
            status = f"⚠️  Critical (>{gate_critical}, RGBExcessive dominance! )"
        elif gate_val >= gate_warning:
            status = f"⚠️  Warning (>{gate_warning}, Approaching excessive dominance)"
        elif gate_val >= gate_target * 0.8:
            status = f"✅ Approaching Target (Objectives:{gate_target})"
        else:
            status = "📈 Growing"
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
    logger("PGCF-RSU Fine-tuning")
    logger(f"{'='*60}")
    logger(f"{'='*60}\n")

    logger(f"✅ Output directory created:")
    logger(f"   Checkpoint: {checkpoint_dir}")
    logger(f"   Log: {log_file}")
    logger("")

    logger("\n" + "="*60)
    logger("Complete Configuration Check")
    logger("="*60)
    logger(f"Data Configuration:")
    logger(f"  Scenario: {cfg.SCENES}")
    logger(f"  RSFConfiguration: {cfg.SCENE_RSF_CONFIG}")
    logger(f"  Window size: {cfg.WINDOW_CONFIG['window_size']}")
    logger(f"  Step: {cfg.WINDOW_CONFIG['stride']}")
    logger(f"  Minimum Effective Frame: {cfg.WINDOW_CONFIG['min_valid_frames']}")
    logger(f"  RGBStrategy: {cfg.RGB_CONFIG['aggregation_strategy']}")
    logger(f"\nModel Configuration:")
    logger(f"  Model Size: {cfg.MODEL_CONFIG['model_size']}")
    logger(f"  EmbedDimension: {cfg.MODEL_CONFIG['embed_dim']}")
    logger(f"  EncoderNumber of layers: {cfg.MODEL_CONFIG['depth']}")
    logger(f"  DecoderNumber of layers: {cfg.MODEL_CONFIG['decoder_depth']}")
    logger(f"  WiFo MaskRatio: {cfg.MODEL_CONFIG['mask_ratio']}")
    logger(f"\nTraining Configuration:")
    logger(f"  Batch size: {cfg.TRAIN_CONFIG['batch_size']}")
    logger(f"  Epochs: {cfg.TRAIN_CONFIG['num_epochs']}")
    logger(f"  Learning Rate: {cfg.TRAIN_CONFIG['learning_rate']}")
    logger(f"  Weight decay: {cfg.TRAIN_CONFIG['weight_decay']}")
    logger(f"  Num workers: {cfg.TRAIN_CONFIG['num_workers']}")
    logger(f"  GPU IDs: {cfg.TRAIN_CONFIG['gpu_ids']}")
    logger(f"\nFreeze Configuration:")
    logger(f"  Freeze Encoder: {cfg.FREEZE_CONFIG['freeze_encoder']}")
    logger(f"  Freeze Decoder: {cfg.FREEZE_CONFIG['freeze_decoder']}")
    logger(f"  Freeze RGB Proj: {cfg.FREEZE_CONFIG['freeze_rgb_proj']}")
    logger("="*60 + "\n")

    logger("Load dataset...")
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

    logger(f"✓ Dataset: {len(dataset)} Sample, {len(dataloader)} batches")

    logger("\nCreate Model...")
    model = create_network(model_size=args.model_size)

    if Path(args.pretrained_path).exists():
        logger(f"Load pre-trained weights: {args.pretrained_path}")
        model, _ = pretrained_weights(
            model,
            args.pretrained_path,
            freeze_encoder=args.freeze_encoder,
            freeze_decoder_layers=args.freeze_decoder_layers,
            verbose=False
        )
    else:
        logger(f"⚠️  Pre-training weight not found, Use Random Initialization")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    gpu_ids = cfg.TRAIN_CONFIG.get('gpu_ids', [0])

    model = model.to(device)

    if torch.cuda.is_available() and len(gpu_ids) > 1:
        logger(f"✓ Used a lot GPUTraining: {gpu_ids}")
        model = GatedWiFoWrapper(model)
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logger(f"  Main GPU: cuda:{gpu_ids[0]}")
        logger(f"  Total GPUNumber: {len(gpu_ids)}")
        logger(f"  ✅ UsedwrapperCompatibleDataParallel")
    else:
        model = GatedWiFoWrapper(model)
        logger(f"✓ Model Ready (Equipment: {device})")

    real_model = model
    if isinstance(model, nn.DataParallel):
        real_model = model.module
    if isinstance(real_model, GatedWiFoWrapper):
        real_model = real_model.model

    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger(f"✓ Trainable Parameters: {trainable_count:,} / {total_params:,} ({100*trainable_count/total_params:.1f}%)")

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

        logger(f"\n✓ Group Learning Rate Configuration:")
        logger(f"   RGB Gate:      {args.lr * args.rgb_gate_lr_multiplier:.2e} (Foundationlr x {args.rgb_gate_lr_multiplier:.0f})")
        logger(f"   RGBProjection Layer:     {args.lr * args.rgb_proj_lr_multiplier:.2e} (Foundationlr x {args.rgb_proj_lr_multiplier:.0f})")
        logger(f"   Other Parameters:       {args.lr:.2e} (Foundationlr)")
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=cfg.TRAIN_CONFIG['weight_decay']
        )
        logger(f"\n✓ Unified learning rate: {args.lr:.2e}")

    logger(f"\n{'='*60}")
    logger(f"Training Configuration")
    logger(f"{'='*60}")
    if len(gpu_ids) > 1:
        logger(f"  GPUConfiguration: MultipleGPUParallel ({len(gpu_ids)}cards)")
        logger(f"    - GPU IDs: {gpu_ids}")
        logger(f"    - MainGPU: cuda:{gpu_ids[0]}")
        logger(f"    - LossAuto Average to PrimaryGPU")
    else:
        logger(f"  GPUConfiguration: SingleGPU (cuda:{gpu_ids[0]})")
    logger(f"  WiFo MaskRatio: {args.mask_ratio}")
    if args.mask_ratio == 0.0:
        logger(f"  Optimize goals: loss_visible (All valid frames)")
        logger(f"  ✅ Do not useWiFoMask, Maximize data utilization")
    else:
        logger(f"  Optimize goals: loss_masked (Masked section)")
        logger(f"  ⚠️  UseWiFoMask, {args.mask_ratio*100:.0f}%Masked")
    logger(f"  Learning Rate: {args.lr}")
    logger(f"  Weight decay: {cfg.TRAIN_CONFIG['weight_decay']}")
    logger(f"  Batch size: {args.batch_size} ({'Totalbatch: ' + str(args.batch_size * len(gpu_ids)) if len(gpu_ids) > 1 else ''})")
    logger(f"  Number of training rounds: {args.epochs}")
    logger(f"  Minimum Effective Frame: {cfg.WINDOW_CONFIG['min_valid_frames']}")

    start_epoch = args.start_epoch
    best_loss = float('inf')
    gate_history = []

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger(f"\n{'='*60}")
            logger(f"FromcheckpointContinue training")
            logger(f"{'='*60}")
            logger(f"Loadcheckpoint: {resume_path}")

            checkpoint = torch.load(resume_path, map_location=device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                real_model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger(f"✓ Optimizer status loaded")
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    logger(f"✓ Fromepoch {start_epoch}Continue")
                if 'best_loss' in checkpoint:
                    best_loss = checkpoint['best_loss']
                    logger(f"✓ Current bestloss: {best_loss:.4e}")
                if 'gate_history' in checkpoint:
                    gate_history = checkpoint['gate_history']
                    logger(f"✓ LoadedgateHistory ({len(gate_history)}Round) ")
                    logger(f"✓ Final RoundgateValue: {gate_history[-1]:.4f}")
            else:
                real_model.load_state_dict(checkpoint)
                logger(f"✓ Model weights loaded")
                logger(f"⚠️  Optimizer status not found, Use the new optimizer")

            logger(f"{'='*60}\n")
        else:
            logger(f"\n⚠️  CheckpointNot found: {resume_path}")
            logger(f"   Training from scratch\n")

    logger(f"\n{'='*60}")
    logger(f"Start training...")
    logger(f"{'='*60}")
    logger(f"Startepoch: {start_epoch}")
    logger(f"Endepoch: {args.epochs}")
    logger(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        avg_loss, avg_loss_masked, avg_loss_visible = train_one_epoch(
            model, dataloader, optimizer, device, epoch, args
        )

        epoch_time = time.time() - start_time

        logger(f"\nEpoch {epoch}/{args.epochs} Done:")
        logger(f"  Loss: {avg_loss:.4e}")
        logger(f"  Loss (Masked): {avg_loss_masked:.4e}")
        logger(f"  Loss (Visible): {avg_loss_visible:.4e}")
        logger(f"  Time: {epoch_time:.1f}s")

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
            logger(f"  ✅ Best model saved (loss: {best_loss:.4e}): {checkpoint_path}")

        logger("")

    logger(f"\n{'='*60}")
    logger(f"Training Complete!")
    logger(f"{'='*60}")
    logger(f"BestLoss: {best_loss:.4e}")

    final_gate = gate_history[-1]
    initial_gate = gate_history[0] if gate_history else 0.01
    logger(f"FinalRGB Gate: {final_gate:.4f} (Initial: {initial_gate:.4f})")

    if hasattr(cfg, 'RESUME_CONFIG'):
        gate_target = cfg.RESUME_CONFIG.get('gate_target', 0.25)
        gate_warning = cfg.RESUME_CONFIG.get('gate_warning_threshold', 0.35)
        gate_critical = cfg.RESUME_CONFIG.get('gate_critical_threshold', 0.50)

        if final_gate >= gate_critical:
            logger(f"⚠️  GateStatus: Critical - RGBExcessive dominance (>{gate_critical})")
            logger(f"   Suggestion: Decrease learning rate multiple or stop training")
        elif final_gate >= gate_warning:
            logger(f"⚠️  GateStatus: Warning - Approaching excessive dominance ({gate_warning}-{gate_critical})")
            logger(f"   Suggestion: Closely monitored, Consider lowering your learning rate")
        elif final_gate >= gate_target * 0.8:
            logger(f"✅ GateStatus: Ideal - Reach equilibrium (~{gate_target})")
            logger(f"   RGBProvide effective enhancements, But not dominantCSIFeatures")
        else:
            logger(f"📈 GateStatus: Somewhat low - There's still room for growth (<{gate_target*0.8:.2f})")
            logger(f"   Suggestion: Can continue training or improve learning rate")
    logger(f"Save Model: {checkpoint_dir}/best_model.pth")
    logger(f"Log Saved: {log_file}")
    logger(f"{'='*60}\n")

if __name__ == '__main__':
    main()
