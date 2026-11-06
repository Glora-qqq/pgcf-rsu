"""From Existing checkpoint Continue fine-tuning. """

import subprocess
import sys
from pathlib import Path

import config as cfg

CHECKPOINT_PATH = cfg.RESUME_CONFIG['checkpoint_path']

START_EPOCH = cfg.RESUME_CONFIG['start_epoch']
END_EPOCH = cfg.RESUME_CONFIG['end_epoch']

BASE_LR = cfg.TRAIN_CONFIG['learning_rate']
USE_GROUPED_LR = cfg.RESUME_CONFIG['use_grouped_lr']
RGB_GATE_MULTIPLIER = cfg.RESUME_CONFIG['rgb_gate_lr_multiplier']
RGB_PROJ_MULTIPLIER = cfg.RESUME_CONFIG['rgb_proj_lr_multiplier']

BATCH_SIZE = cfg.TRAIN_CONFIG['batch_size']
MASK_RATIO = cfg.MODEL_CONFIG['mask_ratio']

checkpoint_path = Path(CHECKPOINT_PATH)
if not checkpoint_path.exists():
    print(f"❌ CheckpointNot found: {checkpoint_path}")
    print(f"   Please check that the path is correct")
    sys.exit(1)

print("=" * 60)
print("FromCheckpointContinue training")
print("=" * 60)
print(f"✅ All configurations from config.py Read")
print()
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"StartEpoch: {START_EPOCH}")
print(f"EndEpoch: {END_EPOCH}")
print(f"Extra training: {END_EPOCH - START_EPOCH + 1} Round")
print()
if USE_GROUPED_LR:
    print("Learning Rate Configuration (Group) : ")
    print(f"  RGB Gate:    {BASE_LR:.2e} x {RGB_GATE_MULTIPLIER:.0f} = {BASE_LR * RGB_GATE_MULTIPLIER:.2e}")
    print(f"  RGBProjection Layer:   {BASE_LR:.2e} x {RGB_PROJ_MULTIPLIER:.0f} = {BASE_LR * RGB_PROJ_MULTIPLIER:.2e}")
    print(f"  Other Parameters:    {BASE_LR:.2e}")
else:
    print(f"Learning Rate Configuration (Unified) : {BASE_LR:.2e}")
print()
print(f"Other Parameters: ")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Mask ratio: {MASK_RATIO}")
print()
print("Expected Effect (Use group learning rate) : ")
print(f"  - Epoch 40: Gate ≈ 0.05-0.08")
print(f"  - Epoch 50: Gate ≈ 0.12-0.18")
print(f"  - Epoch 60: Gate ≈ 0.20-0.28")
print("=" * 60)
print()

cmd = [
    "python", "train.py",
    "--resume", CHECKPOINT_PATH,
    "--start_epoch", str(START_EPOCH),
    "--epochs", str(END_EPOCH),
    "--lr", str(BASE_LR),
    "--batch_size", str(BATCH_SIZE),
    "--mask_ratio", str(MASK_RATIO),
]

if USE_GROUPED_LR:
    cmd.extend([
        "--use_grouped_lr",
        "--rgb_gate_lr_multiplier", str(RGB_GATE_MULTIPLIER),
        "--rgb_proj_lr_multiplier", str(RGB_PROJ_MULTIPLIER),
    ])

print("Start training...")
print(f"Command: {' '.join(cmd)}")
print()

try:
    result = subprocess.run(cmd, check=True)

    print()
    print("=" * 60)
    print("Training Complete! ")
    print("=" * 60)
    print()
    print("Inspection results: ")
    print("  - Model: checkpoints/best_model.pth")
    print()

except subprocess.CalledProcessError as e:
    print()
    print("=" * 60)
    print("❌ Error during training")
    print("=" * 60)
    print(f"Error code: {e.returncode}")
    print()
    print("Please check: ")
    print("  1. train.pyPresence")
    print("  2. Is the dataset path correct?")
    print("  3. GPUAvailability")
    sys.exit(1)

except KeyboardInterrupt:
    print()
    print("=" * 60)
    print("⚠️  Training interrupted by user")
    print("=" * 60)
    print()
    print("Can continue training with the same command")
    sys.exit(0)
