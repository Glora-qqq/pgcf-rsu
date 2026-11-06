"""Completing the Image for Model Reasoning, Mask and Output Path. """

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..'))
DATA_ROOT = os.environ.get('DATA_ROOT', os.path.join(_REPO, 'data'))
WORK_DIR = os.environ.get('RGB_WORK_DIR', os.path.join(_HERE, 'outputs'))
DINOV2_ROOT = os.environ.get('DINOV2_ROOT', os.path.join(_REPO, 'third_party', 'dinov2'))
PROJECT_ROOT = DATA_ROOT

DINOV2_WEIGHTS_PATH = os.environ.get(
    'DINOV2_WEIGHTS',
    os.path.join(DINOV2_ROOT, 'weights', 'dinov2_vitl14_reg4_pretrain.pth'),
)

TRANSFORMER_MODEL_PATH = os.path.join(WORK_DIR, 'checkpoints', 'best_model_RGB.pth')

TRAIN_CONFIG = {
    'images_dir': os.environ.get('IMAGE_DIR_TRAIN', os.path.join(DATA_ROOT, 'images', 'train')),
    'mask_dir': os.path.join(WORK_DIR, 'mask-RGB-train'),
    'output_dir': os.path.join(WORK_DIR, 'train_feature'),
    'snapshot_start': 1,
    'snapshot_end': 750,
    'scene_prefix': 'Scene'
}

TEST_CONFIG = {
    'images_dir': os.environ.get('IMAGE_DIR_TEST', os.path.join(DATA_ROOT, 'images', 'test')),
    'mask_dir': os.path.join(WORK_DIR, 'mask-RGB-test'),
    'output_dir': os.path.join(WORK_DIR, 'test_feature'),
    'snapshot_start': 750,
    'snapshot_end': 1499,
    'scene_prefix': 'Scene',
    'image_scene_format': 'Scene_{scene_num}'
}

DINOV2_MODEL_NAME = 'dinov2_vitl14_reg'
DINOV2_FEATURE_DIM = 1024

TRANSFORMER_D_MODEL = 1024
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 8
TRANSFORMER_DIM_FEEDFORWARD = 4096
TRANSFORMER_DROPOUT = 0.1
MAX_SEQ_LENGTH = 750

VIEWS = ['left', 'middle', 'right']

GPU_IDS = [1, 3]

BATCH_SIZE_PER_GPU = 512

IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

LOG_DIR = os.path.join(WORK_DIR, 'logs')

SHOW_PROGRESS = True

SAVE_INTERMEDIATE = False

NUM_WORKERS = 8

PREFETCH_QUEUE_SIZE = 16

def validate_config():

    errors = []

    if not os.path.exists(DINOV2_WEIGHTS_PATH):
        errors.append(f"DINOv2Weight does not exist: {DINOV2_WEIGHTS_PATH}")

    if not os.path.exists(TRANSFORMER_MODEL_PATH):
        errors.append(f"TransformerModel does not exist: {TRANSFORMER_MODEL_PATH}")

    if not os.path.exists(TRAIN_CONFIG['images_dir']):
        errors.append(f"Training set image directory does not exist: {TRAIN_CONFIG['images_dir']}")
    if not os.path.exists(TRAIN_CONFIG['mask_dir']):
        errors.append(f"Training SetsmaskDirectory does not exist: {TRAIN_CONFIG['mask_dir']}")

    if not os.path.exists(TEST_CONFIG['images_dir']):
        errors.append(f"Test set image directory does not exist: {TEST_CONFIG['images_dir']}")
    if not os.path.exists(TEST_CONFIG['mask_dir']):
        errors.append(f"Test setmaskDirectory does not exist: {TEST_CONFIG['mask_dir']}")

    import torch
    if not torch.cuda.is_available():
        errors.append("CUDANot available")
    else:
        available_gpus = torch.cuda.device_count()
        print(f"  AvailableGPUQuantity: {available_gpus}")

    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("✅ Configuration validation passed")
    return True
