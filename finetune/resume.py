"""从已有 checkpoint 继续微调。"""

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
    print(f"❌ Checkpoint未找到: {checkpoint_path}")
    print(f"   请检查路径是否正确")
    sys.exit(1)

print("=" * 60)
print("从Checkpoint继续训练")
print("=" * 60)
print(f"✅ 所有配置从 config.py 读取")
print()
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"起始Epoch: {START_EPOCH}")
print(f"结束Epoch: {END_EPOCH}")
print(f"额外训练: {END_EPOCH - START_EPOCH + 1} 轮")
print()
if USE_GROUPED_LR:
    print("学习率配置（分组）：")
    print(f"  RGB Gate:    {BASE_LR:.2e} × {RGB_GATE_MULTIPLIER:.0f} = {BASE_LR * RGB_GATE_MULTIPLIER:.2e}")
    print(f"  RGB投影层:   {BASE_LR:.2e} × {RGB_PROJ_MULTIPLIER:.0f} = {BASE_LR * RGB_PROJ_MULTIPLIER:.2e}")
    print(f"  其他参数:    {BASE_LR:.2e}")
else:
    print(f"学习率配置（统一）：{BASE_LR:.2e}")
print()
print(f"其他参数：")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Mask ratio: {MASK_RATIO}")
print()
print("预期效果（使用分组学习率）：")
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

print("启动训练...")
print(f"命令: {' '.join(cmd)}")
print()

try:
    result = subprocess.run(cmd, check=True)

    print()
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)
    print()
    print("检查结果：")
    print("  - 模型: checkpoints/best_model.pth")
    print()

except subprocess.CalledProcessError as e:
    print()
    print("=" * 60)
    print("❌ 训练过程出错")
    print("=" * 60)
    print(f"错误代码: {e.returncode}")
    print()
    print("请检查：")
    print("  1. train.py是否存在")
    print("  2. 数据集路径是否正确")
    print("  3. GPU是否可用")
    sys.exit(1)

except KeyboardInterrupt:
    print()
    print("=" * 60)
    print("⚠️  训练被用户中断")
    print("=" * 60)
    print()
    print("可以使用相同命令继续训练")
    sys.exit(0)
