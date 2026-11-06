"""Data Used for Feature Export, Mask, Weight and Output Path. """

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

_DATA = Path(os.environ.get('DATA_ROOT', PROJECT_ROOT.parent / 'data'))
TRAIN_DATA_ROOT = _DATA / 'train'
TEST_DATA_ROOT = _DATA / 'test'

TRAIN_MASK_ROOT = PROJECT_ROOT / 'masks'
TEST_MASK_ROOT = PROJECT_ROOT / 'masks_test'

TRAIN_RGB_ROOT = PROJECT_ROOT / 'fusion-train-RGB'
TEST_RGB_ROOT = PROJECT_ROOT / 'fusion-test-RGB'

FEATURES_OUTPUT_ROOT = PROJECT_ROOT / 'features'

CHECKPOINT_PATH = PROJECT_ROOT / 'checkpoints' / 'best_model.pth'

TRAIN_SCENES = ['Scene1', 'Scene2', 'Scene3']

TEST_SCENES = ['Scene_1', 'Scene_2', 'Scene_3']

SCENE_NAME_MAPPING = {
    'Scene_1': 'Scene1',
    'Scene_2': 'Scene2',
    'Scene_3': 'Scene3',
}

TRAIN_RSF_CONFIG = {
    'Scene1': ['RSF1', 'RSF4', 'RSF7'],
    'Scene2': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
    'Scene3': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
}

TEST_RSF_CONFIG = {
    'Scene_1': ['RSF1', 'RSF4', 'RSF7'],
    'Scene_2': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
    'Scene_3': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
}

CSI_CONFIG = {
    'num_frames': 1500,

    'height': 128,
    'width': 32,
    'num_channels': 2,

    't_patch_size': 4,
    'patch_size': 4,
}

MODEL_CONFIG = {
    'patch_size': 4,
    't_patch_size': 4,

    'embed_dim': 512,
    'depth': 6,
    'num_heads': 8,

    'decoder_embed_dim': 512,
    'decoder_depth': 4,
    'decoder_num_heads': 8,

    'mlp_ratio': 2.0,

    'rgb_input_dim': 1024,

    'mask_ratio': 0.0,
}

WINDOW_CONFIG = {
    'window_size': 60,
    'stride': 5,
    'min_valid_frames': 0,

    'rgb_strategy': 'mean',
}

EXTRACT_CONFIG = {
    'batch_size': 1,
    'num_workers': 4,

    'feature_source': 'decoder_features',

    'feature_dim': 512,

    'use_gps_alignment': True,
    'verify_alignment': True,

    'save_format': 'npy',
    'include_metadata': True,
    'include_gps': True,

    'verbose': True,
    'log_interval': 10,

    'num_gpus': 8,
    'gpu_ids': None,

    'device': 'cuda',
    'gpu_id': 0,
}

def get_output_paths():

    return {
        'features_root': FEATURES_OUTPUT_ROOT,
        'train_features': FEATURES_OUTPUT_ROOT / 'train',
        'test_features': FEATURES_OUTPUT_ROOT / 'test',
        'feature_index': FEATURES_OUTPUT_ROOT / 'feature_index.json',
        'feature_statistics': FEATURES_OUTPUT_ROOT / 'feature_statistics.json',
        'extraction_log': FEATURES_OUTPUT_ROOT / 'extraction.log',
    }

def get_feature_file_path(scene: str, rsf: str, vehicle: str, is_test: bool = False):

    split = 'test' if is_test else 'train'
    feature_dir = FEATURES_OUTPUT_ROOT / split / scene / vehicle
    rsf_num = rsf.replace('RSF', '')
    feature_file = feature_dir / f"{vehicle}_RSF{rsf_num}_wifo_features.npy"
    return feature_file

def verify_config():

    issues = []

    if not CHECKPOINT_PATH.exists():
        issues.append(f"Checkpoint does not exist: {CHECKPOINT_PATH}")

    if not TRAIN_DATA_ROOT.exists():
        issues.append(f"Training set data does not exist: {TRAIN_DATA_ROOT}")

    if not TEST_DATA_ROOT.exists():
        issues.append(f"Test set data does not exist: {TEST_DATA_ROOT}")

    if not TRAIN_MASK_ROOT.exists():
        issues.append(f"Training SetsMaskAbsent: {TRAIN_MASK_ROOT}")

    if not TEST_MASK_ROOT.exists():
        issues.append(f"Test setMaskAbsent: {TEST_MASK_ROOT}")

    if not TRAIN_RGB_ROOT.exists():
        issues.append(f"Training SetsRGBAbsent: {TRAIN_RGB_ROOT}")

    if not TEST_RGB_ROOT.exists():
        issues.append(f"Test setRGBAbsent: {TEST_RGB_ROOT}")

    if issues:
        print("⚠️  Configure validation to find issues: ")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ Configuration validation passed! ")
        return True

def print_config_summary():

    print("\n" + "=" * 70)
    print("Feature Extraction Configuration Summary")
    print("=" * 70)

    print("\n[Path Configuration] ")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Output directory: {FEATURES_OUTPUT_ROOT}")

    print("\n[Scenario Configuration] ")
    print(f"  Training Set Scenario: {', '.join(TRAIN_SCENES)}")
    print(f"  Test Set Scenario: {', '.join(TEST_SCENES)}")

    print("\n[Model Configuration] ")
    print(f"  CSIDimension: [{CSI_CONFIG['num_frames']}, {CSI_CONFIG['height']}, {CSI_CONFIG['width']}]")
    print(f"  Feature Dimensions: {EXTRACT_CONFIG['feature_dim']}")
    print(f"  MaskRatio: {MODEL_CONFIG['mask_ratio']} (Feature extraction unmasked)")

    print("\n[Slide Window] ")
    print(f"  Window size: {WINDOW_CONFIG['window_size']}")
    print(f"  Slide Step Size: {WINDOW_CONFIG['stride']}")

    print("\n[Extract Configuration] ")
    print(f"  Feature Source: {EXTRACT_CONFIG['feature_source']}")
    if EXTRACT_CONFIG['feature_source'] == 'encoder':
        print(f"    → PureCSIFeatures (Not subject toRGB gateImpact) ")
    elif EXTRACT_CONFIG['feature_source'] == 'decoder_features':
        print(f"    → Decoder BlocksOutput (CSI+RGBFused High-Level Semantic Features) ")
        print(f"    → Suitable for tasks that require scene semantics such as vehicle positioning")
    print(f"  GPSAlignment: {EXTRACT_CONFIG['use_gps_alignment']}")
    print(f"  Output format: {EXTRACT_CONFIG['save_format']}")
    print(f"  Equipment: {EXTRACT_CONFIG['device']}")

    print("\n" + "=" * 70)

if __name__ == '__main__':
    print_config_summary()
    print("\n")
    verify_config()

    print("\n[Sample Output Path] ")
    paths = get_output_paths()
    for key, path in paths.items():
        print(f"  {key}: {path}")

    print("\n[Sample Feature File Path] ")
    train_file = get_feature_file_path('Scene1', 'RSF1', 'Car1', is_test=False)
    test_file = get_feature_file_path('Scene_1', 'RSF1', 'Car1', is_test=True)
    print(f"  Training Sets: {train_file}")
    print(f"  Test set: {test_file}")
