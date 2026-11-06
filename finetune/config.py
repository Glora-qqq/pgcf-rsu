"""WiFo Data for progressive gating fine-tuning, Models and Optimized Configurations. """

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
        print("[WARNING] The following paths do not exist:")
        for path in missing_paths:
            print(path)
        return False

    print("[OK] Critical path validation passed")
    return True

def create_output_dirs():

    output_dirs = [
        CHECKPOINT_ROOT,
        LOG_ROOT,
    ]

    for dir_path in output_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Output directory created:")
    print(f"  Checkpoint: {CHECKPOINT_ROOT}")
    print(f"  Log: {LOG_ROOT}")

def print_config():

    print("="*80)
    print("PGCF-RSU Fine-tune configuration")
    print("="*80)

    print(f"\n[Project Path]")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Fine Tune Table of Contents: {TRAIN_WIFO_ROOT}")

    print(f"\n[Data Path]")
    print(f"  CSIData: {CSI_DATA_ROOT}")
    print(f"  MaskFile: {MASK_ROOT}")
    print(f"  DINOv2Features: {RGB_FEATURE_ROOT}")
    print(f"    {'✅ Present' if RGB_FEATURE_ROOT.exists() else '❌ Absent (Confirmation required) '}")

    print("\n[Pre-trained weights]")
    print(f"  Case StudyContents: {CASE_STUDY_ROOT}")
    print(f"  Pre-trained weights: {WIFO_RGB_PRETRAIN_PATH}")
    print(f"    {'✅ Present' if WIFO_RGB_PRETRAIN_PATH.exists() else '❌ Absent'}")

    print(f"\n[Scenario Configuration]")
    print(f"  Training Scenarios: {SCENES}")
    for scene, rsf_list in SCENE_RSF_CONFIG.items():
        print(f"    {scene}: {rsf_list}")

    print(f"\n[Data Configuration]")
    print(f"  CSI: {CSI_CONFIG['num_channels']} x {CSI_CONFIG['num_frames']} x {CSI_CONFIG['height']} x {CSI_CONFIG['width']}")
    print(f"  RGBFeatures: [1500, {RGB_CONFIG['feature_dim']}]")
    print(f"  Slide Window: {WINDOW_CONFIG['window_size']}Frame, Step{WINDOW_CONFIG['stride']}, Min{WINDOW_CONFIG['min_valid_frames']}Frame")

    print(f"\n[Model Configuration]")
    print(f"  Architecture: GatedWiFo {MODEL_CONFIG['model_size']}")
    print(f"  EmbedDimension: {MODEL_CONFIG['embed_dim']}")
    print(f"  EncoderDepth: {MODEL_CONFIG['depth']}")
    print(f"  DecoderDepth: {MODEL_CONFIG['decoder_depth']}")
    print(f"  RGBProjection: {MODEL_CONFIG['rgb_input_dim']} → {MODEL_CONFIG['rgb_output_dim']}")

    print(f"\n[Training Configuration (For small data volumes) ]")
    print(f"  ⚠️  Amount of data: 82Sample, ~18KWindow (Less! ) ")
    print(f"  Objectives: ExtractCSIFeatures (Rather than rebuilding) ")
    print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"  Epochs: {TRAIN_CONFIG['num_epochs']} (Prevent overfitting)")
    print(f"  Learning Rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"  Weight decay: {TRAIN_CONFIG['weight_decay']} (Regularization)")
    print(f"  Num workers: {TRAIN_CONFIG['num_workers']}")
    print(f"  GPU IDs: {TRAIN_CONFIG['gpu_ids']}")
    print(f"  RGBAggregation: {RGB_CONFIG['aggregation_strategy']}")

    print(f"\n[Fine-tune strategy (Most conservative, Best for small data) ]")
    print(f"  🎯 Objectives: ExtractCSIFeatures (Retain pre-trained knowledge + FitRGB) ")
    print(f"  Encoder: {'Freeze✅' if FREEZE_CONFIG['freeze_encoder'] else 'Fine-tuning'} (Preserve pre-training CSI Encoder)")
    print(f"  RGBProjection: {'Freeze' if FREEZE_CONFIG['freeze_rgb_proj'] else 'Fine-tuning✅'} (Unique fine-tuning, LetRGBFit)")
    print(f"  Decoder: {'Freeze✅' if FREEZE_CONFIG['freeze_decoder'] else 'Fine-tuning'} (For training signals only)")
    print(f"  Trainable Parameters: ~1M (2%) - Most insured! ")
    print(f"  Sample/Parameter Ratio: 18K/1M = 0.018 (Very healthy)")
    print(f"  ✅ Strengths: Hardly overfit, Full retention pre-trainingCSIKnowledge")

    print("="*80)

if __name__ == '__main__':
    print_config()

    print(f"\n[Validation Path]")
    validate_paths()

    print(f"\n[Create Output Directory]")
    create_output_dirs()

    print(f"\n[Done]")
