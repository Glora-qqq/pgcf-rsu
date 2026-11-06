"""定位推理的测试车辆、滤波开关和轨迹输出路径。"""

import os
from pathlib import Path
from config import *

TEST_DATA_ROOT = SRC_ROOT / 'test'

TEST_WIFO_FEATURES_ROOT = Path(os.environ.get(
    'TEST_WIFO_FEATURES_ROOT',
    PROJECT_ROOT.parent / 'finetune' / 'features' / 'test',
))

TEST_OUTPUT_ROOT = PROJECT_ROOT / 'predictions'

SCENE_DIR_MAPPING = {
    'Scene1': 'Scene_1',
    'Scene2': 'Scene_2',
    'Scene3': 'Scene_3',
}

TEST_VEHICLES = {
    'Scene1': ['Car1', 'Car10', 'Car12', 'Car4', 'Car5', 'Car7'],
    'Scene2': ['Car14', 'Car18', 'Car19', 'Car2', 'Car7', 'Car8'],
    'Scene3': [
        'AT1',
        'AT2',
        'AT3',
        'AT4',
        'AT5',
        'AT6',
        'Car1',
        'Car10',
        'Car20',
        'Car4'
    ],
}

MODEL_CHECKPOINT_PATH = CHECKPOINT_ROOT / 'pose_best.pt'
ABLATION_MODALITY = 'full'

TEST_BATCH_SIZE = 1

TEST_DEVICE = 'cpu'

VERBOSE = True

USE_INPUT_KALMAN = True

INPUT_KALMAN_CONFIG = {
    'process_noise': 0.10,
    'measurement_noise': 1.6,
}

USE_AT_UKF = True
AT_UKF_FILTER_ORIENTATION = False
AT_UKF_CONFIG = {
    'process_noise': 0.1,
    'process_noise_ori': 0.01,
    'measurement_noise': 1.6,
    'measurement_noise_ori': 0.05,
    'dt': 0.05,
}

USE_POST_PROCESSING_FILTER = True

POST_CAR_PROCESS_NOISE = 0.10
POST_CAR_MEASUREMENT_NOISE = 1.6

OUTPUT_FILENAME_TEMPLATE = "{vehicle}.txt"

POSITION_DECIMAL_PLACES = 8

INCLUDE_HEADER = False

GPS_FILE_SUFFIX = '_train_cat_noise.txt'

WIFO_FEATURE_FILENAME_TEMPLATE = "{vehicle}_{rsf}_wifo_features.npy"

RSF_EXTRINSIC_FILENAME_TEMPLATE = "RSF_position_{scene_id}.txt"

if __name__ == '__main__':
    print("=" * 60)
    print("PGCF-RSU 定位推理配置")
    print("=" * 60)
    print(f"测试数据根目录: {TEST_DATA_ROOT}")
    print(f"WiFo特征根目录: {TEST_WIFO_FEATURES_ROOT}")
    print(f"模型检查点路径: {MODEL_CHECKPOINT_PATH}")
    print(f"测试输出目录: {TEST_OUTPUT_ROOT}")
    print(f"\n场景目录映射:")
    for scene, scene_dir in SCENE_DIR_MAPPING.items():
        print(f"  {scene} -> {scene_dir}")
    print(f"\n测试车辆统计:")
    for scene, vehicles in TEST_VEHICLES.items():
        print(f"  {scene}: {len(vehicles)} 辆车")
    print(f"  总计: {sum(len(v) for v in TEST_VEHICLES.values())} 辆车")
    print(f"\n场景RSF配置:")
    for scene, rsfs in SCENE_RSF_CONFIG.items():
        print(f"  {scene}: {rsfs}")
    print("=" * 60)
