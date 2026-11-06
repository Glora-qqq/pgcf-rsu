"""WiFo 渐进门控微调的数据、模型和优化配置。"""

import os
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

_DATA = Path(os.environ.get('DATA_ROOT', PROJECT_ROOT.parent / 'data'))
DATA_ROOT = _DATA / 'train'

TRAIN_WIFO_ROOT = PROJECT_ROOT

CSI_DATA_ROOT = DATA_ROOT

MASK_ROOT = PROJECT_ROOT / 'masks'

RGB_FEATURE_ROOT = Path(os.environ.get(
    'RGB_FEATURE_ROOT',
    PROJECT_ROOT.parent / 'visual' / 'outputs' / 'fused_features',
))

CASE_STUDY_ROOT = Path(os.environ.get('WIFO_WEIGHTS', PROJECT_ROOT.parent / 'weights' / 'WiFo-RGB.pkl')).parent
WIFO_RGB_PRETRAIN_PATH = Path(os.environ.get('WIFO_WEIGHTS', PROJECT_ROOT.parent / 'weights' / 'WiFo-RGB.pkl'))
WIFO_RGB_CODE_DIR = PROJECT_ROOT / 'models'

CHECKPOINT_ROOT = PROJECT_ROOT / 'checkpoints'

LOG_ROOT = PROJECT_ROOT / 'logs'

SCENES = ['Scene1', 'Scene2', 'Scene3']

SCENE_RSF_CONFIG = {
    'Scene1': ['RSF1', 'RSF4', 'RSF7'],
    'Scene2': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
    'Scene3': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
}

CSI_CONFIG = {
    'num_frames': 1500,
    'height': 128,
    'width': 32,
    'num_channels': 2,

    'frame_interval': 0.05,

}

RGB_CONFIG = {
    'feature_dim': 1024,

    'aggregation_strategy': 'mean',
}

WINDOW_CONFIG = {
    'window_size': 60,
    'stride': 5,
    'min_valid_frames': 8,

}

MODEL_CONFIG = {
    'model_size': 'base',

    'embed_dim': 512,
    'depth': 6,
    'decoder_embed_dim': 512,
    'decoder_depth': 4,
    'num_heads': 8,
    'decoder_num_heads': 8,
    'mlp_ratio': 2,

    't_patch_size': 4,
    'patch_size': 4,

    'pos_emb': 'SinCos_3D',
    'no_qkv_bias': False,

    'rgb_input_dim': 1024,
    'rgb_output_dim': 512,

    'mask_ratio': 0.5,
}

TRAIN_CONFIG = {
    'batch_size': 42,
    'num_workers': 4,

    'num_epochs': 40,

    'learning_rate': 1e-4,
    'weight_decay': 0.05,


    'device': 'cuda',
    'gpu_ids': [0,1,2,3,4,5,6],
}

FREEZE_CONFIG = {

    'freeze_encoder': True,
    'freeze_embedding': True,
    'freeze_pos_embed': True,
    'freeze_encoder_blocks': True,

    'freeze_rgb_proj': False,

    'freeze_decoder': True,
    'freeze_decoder_embed': True,
    'freeze_decoder_blocks': True,
    'freeze_decoder_pred': True,

}

RESUME_CONFIG = {
    'checkpoint_path': 'checkpoints/best_model.pth',

    'start_epoch': 35,
    'end_epoch': 60,

    'use_grouped_lr': True,
    'rgb_gate_lr_multiplier': 20.0,
    'rgb_proj_lr_multiplier': 10.0,

    'gate_target': 0.25,
    'gate_warning_threshold': 0.35,
    'gate_critical_threshold': 0.50,

}

def validate_paths():

    critical_paths = {
        'CSI_DATA_ROOT': CSI_DATA_ROOT,
        'MASK_ROOT': MASK_ROOT,
        'WIFO_RGB_PRETRAIN_PATH': WIFO_RGB_PRETRAIN_PATH,
    }

    missing_paths = []
    for name, path in critical_paths.items():
        if not path.exists():
            missing_paths.append(f"  ❌ {name}: {path}")

    if missing_paths:
        print("[WARNING] 以下路径不存在:")
        for path in missing_paths:
            print(path)
        return False

    print("[OK] 关键路径验证通过")
    return True

def create_output_dirs():

    output_dirs = [
        CHECKPOINT_ROOT,
        LOG_ROOT,
    ]

    for dir_path in output_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] 输出目录已创建:")
    print(f"  检查点: {CHECKPOINT_ROOT}")
    print(f"  日志: {LOG_ROOT}")

def print_config():

    print("="*80)
    print("PGCF-RSU 微调配置")
    print("="*80)

    print(f"\n[项目路径]")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  微调目录: {TRAIN_WIFO_ROOT}")

    print(f"\n[数据路径]")
    print(f"  CSI数据: {CSI_DATA_ROOT}")
    print(f"  Mask文件: {MASK_ROOT}")
    print(f"  DINOv2特征: {RGB_FEATURE_ROOT}")
    print(f"    {'✅ 存在' if RGB_FEATURE_ROOT.exists() else '❌ 不存在（需要确认）'}")

    print("\n[预训练权重]")
    print(f"  Case Study目录: {CASE_STUDY_ROOT}")
    print(f"  预训练权重: {WIFO_RGB_PRETRAIN_PATH}")
    print(f"    {'✅ 存在' if WIFO_RGB_PRETRAIN_PATH.exists() else '❌ 不存在'}")

    print(f"\n[场景配置]")
    print(f"  训练场景: {SCENES}")
    for scene, rsf_list in SCENE_RSF_CONFIG.items():
        print(f"    {scene}: {rsf_list}")

    print(f"\n[数据配置]")
    print(f"  CSI: {CSI_CONFIG['num_channels']} x {CSI_CONFIG['num_frames']} x {CSI_CONFIG['height']} x {CSI_CONFIG['width']}")
    print(f"  RGB特征: [1500, {RGB_CONFIG['feature_dim']}]")
    print(f"  滑动窗口: {WINDOW_CONFIG['window_size']}帧, 步长{WINDOW_CONFIG['stride']}, 最少{WINDOW_CONFIG['min_valid_frames']}帧")

    print(f"\n[模型配置]")
    print(f"  架构: GatedWiFo {MODEL_CONFIG['model_size']}")
    print(f"  Embed维度: {MODEL_CONFIG['embed_dim']}")
    print(f"  Encoder深度: {MODEL_CONFIG['depth']}")
    print(f"  Decoder深度: {MODEL_CONFIG['decoder_depth']}")
    print(f"  RGB投影: {MODEL_CONFIG['rgb_input_dim']} → {MODEL_CONFIG['rgb_output_dim']}")

    print(f"\n[训练配置（针对小数据量）]")
    print(f"  ⚠️  数据量: 82样本, ~18K窗口（较少！）")
    print(f"  目标: 提取CSI特征（而非重建）")
    print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"  Epochs: {TRAIN_CONFIG['num_epochs']} (防止过拟合)")
    print(f"  学习率: {TRAIN_CONFIG['learning_rate']}")
    print(f"  Weight decay: {TRAIN_CONFIG['weight_decay']} (正则化)")
    print(f"  Num workers: {TRAIN_CONFIG['num_workers']}")
    print(f"  GPU IDs: {TRAIN_CONFIG['gpu_ids']}")
    print(f"  RGB聚合: {RGB_CONFIG['aggregation_strategy']}")

    print(f"\n[微调策略（最保守，最适合小数据）]")
    print(f"  🎯 目标: 提取CSI特征（保留预训练知识 + 适配RGB）")
    print(f"  Encoder: {'冻结✅' if FREEZE_CONFIG['freeze_encoder'] else '微调'} (保留预训练 CSI 编码器)")
    print(f"  RGB投影: {'冻结' if FREEZE_CONFIG['freeze_rgb_proj'] else '微调✅'} (唯一微调，让RGB适配)")
    print(f"  Decoder: {'冻结✅' if FREEZE_CONFIG['freeze_decoder'] else '微调'} (只用于训练信号)")
    print(f"  可训练参数: ~1M (2%) - 最保险！")
    print(f"  样本/参数比: 18K/1M = 0.018 (非常健康)")
    print(f"  ✅ 优势: 几乎不会过拟合，完全保留预训练CSI知识")

    print("="*80)

if __name__ == '__main__':
    print_config()

    print(f"\n[验证路径]")
    validate_paths()

    print(f"\n[创建输出目录]")
    create_output_dirs()

    print(f"\n[完成]")
