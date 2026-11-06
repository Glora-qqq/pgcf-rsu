"""Three Perspectives Fusion Training and Reasoning Path and Scene Configuration. """

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get('DATA_ROOT', os.path.abspath(os.path.join(_HERE, '..', 'data')))
ENV_FEATURE_DIR = os.environ.get('RGB_WORK_DIR', os.path.join(_HERE, 'outputs'))

TRAIN_FEATURE_DIR = os.path.join(ENV_FEATURE_DIR, 'train_feature')
TEST_FEATURE_DIR = os.path.join(ENV_FEATURE_DIR, 'test_feature')

FUSION_OUTPUT_DIR = os.path.join(ENV_FEATURE_DIR, 'fused_features')
CHECKPOINT_DIR = os.path.join(ENV_FEATURE_DIR, 'fusion_checkpoints')
LOG_DIR = os.path.join(ENV_FEATURE_DIR, 'fusion_logs')

SCENES = {
    'Scene1': {
        'rsf_list': ['RSF1', 'RSF4', 'RSF7'],
        'scene_type': 'intersection',
        'num_rsf': 3
    },
    'Scene2': {
        'rsf_list': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
        'scene_type': 'street',
        'num_rsf': 4
    },
    'Scene3': {
        'rsf_list': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
        'scene_type': 'street',
        'num_rsf': 4
    }
}

VIEWS = ['left', 'middle', 'right']

VIEW_ANGLES = {
    'intersection': {
        'left': -0.9163,
        'middle': 0.0,
        'right': 0.9163
    },
    'street': {
        'left': -1.5708,
        'middle': 0.0,
        'right': 1.5708
    }
}

FEATURE_DIM = 1024
SEQUENCE_LENGTH = 750

MODEL_CONFIG = {
    'd_model': 1024,
    'num_heads': 16,
    'num_layers': 4,
    'dropout': 0.1
}

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BETAS = (0.9, 0.999)

BATCH_SIZE = 3
NUM_EPOCHS = 50
GRADIENT_CLIP = 1.0
NUM_WORKERS = 4

LR_SCHEDULER = 'cosine'
WARMUP_EPOCHS = 5
MIN_LR = 1e-5

USE_VALIDATION = False
VAL_RATIO = 0.15
VAL_INTERVAL = 1

EARLY_STOPPING = True
PATIENCE = 10
MIN_DELTA = 1e-4

LOSS_WEIGHTS = {
    'reconstruction': 1.0,
    'consistency': 0.2,
    'contrast': 0.1
}

GPU_IDS = [1]
DEVICE = 'cuda'

SAVE_INTERVAL = None
SAVE_BEST_ONLY = True
SAVE_LAST = True
METRIC_FOR_BEST = 'val_loss'

LOG_INTERVAL = 10

RANDOM_SEED = 42

def validate_config():

    errors = []

    if not os.path.exists(TRAIN_FEATURE_DIR):
        errors.append(f"Training feature directory does not exist: {TRAIN_FEATURE_DIR}")

    for scene_name, scene_config in SCENES.items():
        scene_dir = os.path.join(TRAIN_FEATURE_DIR, scene_name)
        if not os.path.exists(scene_dir):
            errors.append(f"Scene directory does not exist: {scene_dir}")
        else:
            for rsf in scene_config['rsf_list']:
                rsf_dir = os.path.join(scene_dir, rsf)
                if not os.path.exists(rsf_dir):
                    errors.append(f"RSFDirectory does not exist: {rsf_dir}")

    import torch
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        errors.append("CUDANot available")

    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("✅ Configuration validation passed")
    return True

def print_config():

    print("\n" + "="*70)
    print("Triangle Fusion Training Configuration")
    print("="*70)

    print("\n[Data Configuration] ")
    for scene_name, scene_config in SCENES.items():
        print(f"  {scene_name}: {scene_config['num_rsf']} RSF ({scene_config['scene_type']})")
    print(f"  Number of Viewing Angles: {len(VIEWS)}")
    print(f"  Feature Dimensions: {FEATURE_DIM}")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")

    print("\n[Model Configuration] ")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n[Training Configuration] ")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Validation ratio: {VAL_RATIO:.1%}")

    print("\n[Loss weight] ")
    for key, value in LOSS_WEIGHTS.items():
        print(f"  {key}: {value}")

    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    print_config()
    validate_config()
