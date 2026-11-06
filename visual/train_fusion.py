"""Training a three-view visual feature fusion model. """

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_fusion import *
from models.fusion import create_fusion_model
from data.fusion_set import create_dataloaders
from utils.common import (
    set_seed,
    get_view_angles,
    create_sample_list,
    split_train_val,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    compute_temporal_consistency_loss,
    compute_contrastive_loss,
    format_time
)

class FusionLoss(nn.Module):

    def __init__(self, weights: dict = None):
        super().__init__()
        self.weights = weights if weights else LOSS_WEIGHTS

    def forward(self, fused, original_views):

        recon_target = torch.stack(original_views, dim=0).mean(dim=0)
        recon_loss = F.mse_loss(fused, recon_target)

        consistency_loss = compute_temporal_consistency_loss(original_views)

        contrast_loss = compute_contrastive_loss(fused, original_views)

        total_loss = (
            self.weights['reconstruction'] * recon_loss +
            self.weights['consistency'] * consistency_loss +
            self.weights['contrast'] * contrast_loss
        )

        loss_dict = {
            'reconstruction': recon_loss.item(),
            'consistency': consistency_loss.item(),
            'contrast': contrast_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch
):

    model.train()

    loss_meter = AverageMeter()
    recon_meter = AverageMeter()
    consist_meter = AverageMeter()
    contrast_meter = AverageMeter()

    start_time = time.time()

    for batch_idx, (view_features, scene_names, rsf_names, scene_types) in enumerate(train_loader):
        view_features = [v.to(device) for v in view_features]

        view_angles = get_view_angles(scene_types[0])

        optimizer.zero_grad()
        fused = model(view_features, view_angles)

        loss, loss_dict = criterion(fused, view_features)

        loss.backward()

        if GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

        optimizer.step()

        batch_size = view_features[0].size(0)
        loss_meter.update(loss_dict['total'], batch_size)
        recon_meter.update(loss_dict['reconstruction'], batch_size)
        consist_meter.update(loss_dict['consistency'], batch_size)
        contrast_meter.update(loss_dict['contrast'], batch_size)

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(f"  [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"(R:{recon_meter.avg:.4f} C:{consist_meter.avg:.4f} Con:{contrast_meter.avg:.4f}) "
                  f"Time: {format_time(elapsed)}")

    return loss_meter.avg

@torch.no_grad()
def validate(
    model,
    val_loader,
    criterion,
    device,
    epoch
):

    model.eval()

    loss_meter = AverageMeter()
    recon_meter = AverageMeter()
    consist_meter = AverageMeter()
    contrast_meter = AverageMeter()

    for view_features, scene_names, rsf_names, scene_types in val_loader:
        view_features = [v.to(device) for v in view_features]

        view_angles = get_view_angles(scene_types[0])

        fused = model(view_features, view_angles)

        loss, loss_dict = criterion(fused, view_features)

        batch_size = view_features[0].size(0)
        loss_meter.update(loss_dict['total'], batch_size)
        recon_meter.update(loss_dict['reconstruction'], batch_size)
        consist_meter.update(loss_dict['consistency'], batch_size)
        contrast_meter.update(loss_dict['contrast'], batch_size)

    print(f"  Verify Loss: {loss_meter.avg:.4f} "
          f"(R:{recon_meter.avg:.4f} C:{consist_meter.avg:.4f} Con:{contrast_meter.avg:.4f})")

    return loss_meter.avg

def main():

    print("\n" + "="*70)
    print("Three view environmental feature fusion training")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n[Step1] Validate Configuration")
    print_config()
    if not validate_config():
        return

    set_seed(RANDOM_SEED)
    print(f"✅ Random Seeds: {RANDOM_SEED}")

    os.makedirs(FUSION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("\n[Step2] Prepare data")
    all_samples = create_sample_list(SCENES, TRAIN_FEATURE_DIR)
    print(f"  Total Samples: {len(all_samples)}")

    if USE_VALIDATION:
        train_samples, val_samples = split_train_val(all_samples, VAL_RATIO, RANDOM_SEED)
        print(f"  Training Samples: {len(train_samples)}")
        print(f"  Validate Sample: {len(val_samples)}")
    else:
        train_samples = all_samples
        val_samples = []
        print(f"  Training Samples: {len(train_samples)} (Don't use validation sets)")
        print(f"  The convergence will be judged by the training loss")

    print("\n[Step3] Create Data Loader")
    train_loader, val_loader = create_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        feature_dir=TRAIN_FEATURE_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print("\n[Step4] Create Model")
    model = create_fusion_model(MODEL_CONFIG)
    print(f"  Number of model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    device = torch.device(DEVICE)
    model = model.to(device)
    print(f"  Equipment: {device}")

    print("\n[Step5] Create Optimizer")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS
    )

    if LR_SCHEDULER == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
            eta_min=MIN_LR
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )

    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Scheduler: {LR_SCHEDULER}")

    criterion = FusionLoss(LOSS_WEIGHTS)

    print("\n[Step6] Start training")
    print("="*70)

    best_val_loss = float('inf')
    start_epoch = 0

    patience_counter = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 70)

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        if USE_VALIDATION and (epoch + 1) % VAL_INTERVAL == 0:
            val_loss = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch
            )
        else:
            val_loss = train_loss

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        is_best = val_loss < (best_val_loss - MIN_DELTA)
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0

            best_checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                'fusion_best.pth'
            )
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=best_checkpoint_path,
                is_best=True
            )
            print(f"  💾 Save optimal model: {best_checkpoint_path}")
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start

        if USE_VALIDATION:
            print(f"\nEpoch {epoch+1} Done - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"LR: {current_lr:.6f}, "
                  f"Time: {format_time(epoch_time)}")
            if is_best:
                print(f"🎉 New Optimal Validation Loss: {best_val_loss:.4f}")
        else:
            print(f"\nEpoch {epoch+1} Done - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"LR: {current_lr:.6f}, "
                  f"Time: {format_time(epoch_time)}")
            if is_best:
                print(f"🎉 New Optimal Training Loss: {best_val_loss:.4f}")

        if EARLY_STOPPING and patience_counter >= PATIENCE:
            print(f"\n⚠️  Early StoppingTrigger: Continuous{PATIENCE}PieceepochsNo improvement")
            print(f"   Optimal Loss: {best_val_loss:.4f} (Epoch {epoch+1-PATIENCE})")
            break

    if SAVE_LAST:
        last_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'fusion_last.pth')
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=train_loss,
            val_loss=val_loss,
            checkpoint_path=last_checkpoint_path,
            is_best=False
        )
        print(f"  💾 Save lastepoch: {last_checkpoint_path}\n")

    print("\n" + "="*70)
    print("✅ Training Complete! ")
    print(f"Optimal Validation Loss: {best_val_loss:.4f}")
    print(f"Model Save Path: {os.path.join(CHECKPOINT_DIR, 'fusion_best.pth')}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
