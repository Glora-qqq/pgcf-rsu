"""Export on Test Set CSI Paths and scenarios when blending features with vision. """

import os
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

_DATA = Path(os.environ.get('DATA_ROOT', PROJECT_ROOT.parent / 'data'))
TEST_DATA_ROOT = _DATA / 'test'

TEST_MASK_ROOT = PROJECT_ROOT / 'masks_test'

TEST_RGB_FEATURE_ROOT = PROJECT_ROOT / 'fusion-test-RGB'

TRAIN_WIFO_ROOT = PROJECT_ROOT

TEST_SCENES = ['Scene_1', 'Scene_2', 'Scene_3']

SCENE_NAME_MAPPING = {
    'Scene_1': 'Scene1',
    'Scene_2': 'Scene2',
    'Scene_3': 'Scene3',
}

TEST_RSF_CONFIG = {
    'Scene1': ['RSF1', 'RSF4', 'RSF7'],
    'Scene2': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
    'Scene3': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
}

TEST_CSI_CONFIG = {
    'num_frames_original': 750,
    'num_frames_padded': 1500,

    'height': 128,
    'width': 32,
    'num_channels': 2,

    'downsample_factor': 4,
    'num_subcarriers_after': 32,
}

TEST_MASK_CONFIG = {
    'mask_format': 'npy',

    'force_mask_start': 750,
    'force_mask_end': 1500,

    'missing_value': 1.0,
    'valid_value': 0.0,
}

TEST_RGB_CONFIG = {
    'feature_dim': 1024,

    'num_frames_original': 750,
    'num_frames_padded': 1500,

    'padding_strategy': 'mean',

    'file_format': 'npz',
    'feature_key': 'fused',
}

TEST_WINDOW_CONFIG = {
    'window_size': 60,
    'stride': 5,
    'min_valid_frames': 8,

    'rgb_strategy': 'mean',
}

TEST_CONFIG = {
    'batch_size': 24,
    'num_workers': 4,

    'device': 'cuda',
    'gpu_ids': [0,1,2,3],

    'mask_ratio': 0.5,
}

TEST_OUTPUT_ROOT = PROJECT_ROOT / 'test_outputs'

TEST_FEATURE_ROOT = PROJECT_ROOT / 'test_features'


WIFO_RGB_PRETRAIN_PATH = Path(os.environ.get('WIFO_WEIGHTS', PROJECT_ROOT.parent / 'weights' / 'WiFo-RGB.pkl'))

FINETUNED_CHECKPOINT = PROJECT_ROOT / 'checkpoints' / 'best_model.pth'

DEBUG_CONFIG = {
    'verbose': True,
    'save_samples': False,
}

def verify_paths():

    print("=" * 60)
    print("Test Configuration Path Validation")
    print("=" * 60)

    paths = {
        'TEST_DATA_ROOT': TEST_DATA_ROOT,
        'TEST_MASK_ROOT': TEST_MASK_ROOT,
        'TEST_RGB_FEATURE_ROOT': TEST_RGB_FEATURE_ROOT,
        'WIFO_RGB_PRETRAIN_PATH': WIFO_RGB_PRETRAIN_PATH,
    }

    all_exist = True
    for name, path in paths.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_exist = False

    print("=" * 60)
    if all_exist:
        print("✓ All paths exist")
    else:
        print("✗ Some paths do not exist, Please check")
    print("=" * 60)

    return all_exist
