"""场景、车辆、路侧单元和坐标系常量。"""

import numpy as np

SCENES = ['Scene1', 'Scene2', 'Scene3']

SCENE_RSF_CONFIG = {
    'Scene1': ['RSF1', 'RSF4', 'RSF7'],
    'Scene2': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
    'Scene3': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
}

SCENE_VEHICLES = {
    'Scene1': ['Car1', 'Car4', 'Car5', 'Car7', 'Car10', 'Car12'],
    'Scene2': ['Car2', 'Car7', 'Car8', 'Car14', 'Car18', 'Car19'],
    'Scene3': ['Car1', 'Car4', 'Car10', 'Car20', 'AT1', 'AT2', 'AT3', 'AT4', 'AT5', 'AT6'],
}

SCENE_DESCRIPTIONS = {
    'Scene1': '分叉路口高车流量密度场景',
    'Scene2': '超宽车道低车流量密度场景',
    'Scene3': '低空经济场景',
}

VEHICLE_HEIGHT_CORRECTION = {
    ('Scene1', 'Car1'): 2.05,
    ('Scene1', 'Car4'): 1.00,
    ('Scene1', 'Car5'): 0.95,
    ('Scene1', 'Car7'): 1.40,
    ('Scene1', 'Car10'): 0.80,
    ('Scene1', 'Car12'): 2.05,

    ('Scene2', 'Car2'): 0.52,
    ('Scene2', 'Car7'): 1.07,
    ('Scene2', 'Car8'): 0.52,
    ('Scene2', 'Car14'): 0.52,
    ('Scene2', 'Car18'): 0.40,
    ('Scene2', 'Car19'): 0.66,

    ('Scene3', 'Car1'): 1.25,
    ('Scene3', 'Car4'): 0.52,
    ('Scene3', 'Car10'): 0.52,
    ('Scene3', 'Car20'): 0.52,
    ('Scene3', 'AT1'): 0.0,
    ('Scene3', 'AT2'): 0.0,
    ('Scene3', 'AT3'): 0.0,
    ('Scene3', 'AT4'): 0.0,
    ('Scene3', 'AT5'): 0.0,
    ('Scene3', 'AT6'): 0.0,
}

def get_vehicle_height_correction(scene_name, vehicle_name):

    key = (scene_name, vehicle_name)
    if key not in VEHICLE_HEIGHT_CORRECTION:
        print(f"[WARNING] 未找到车辆高度修正值: {scene_name}/{vehicle_name}，使用0.0")
        return 0.0
    return VEHICLE_HEIGHT_CORRECTION[key]

def is_aerial_vehicle(vehicle_name):

    return vehicle_name.startswith('AT') or vehicle_name.startswith('UAV')

RSF_ANTENNA_ARRAY_POSITION = {
    'Scene1': np.array([0.0, 0.0, 0.5]),
    'Scene2': np.array([0.0, 0.0, 2.3]),
    'Scene3': np.array([0.0, 0.0, 2.3]),
}

def get_rsf_antenna_position(scene_name):

    return RSF_ANTENNA_ARRAY_POSITION.get(scene_name, np.array([0.0, 0.0, 0.5]))

RSF_CAMERA_CONFIG_SCENE1 = {
    'position': np.array([3.0, 0.0, -0.2]),
    'pitch': 15.0,
    'cameras': [
        {'name': 'left', 'yaw': -52.5},
        {'name': 'middle', 'yaw': 0.0},
        {'name': 'right', 'yaw': 52.5},
    ]
}

RSF_CAMERA_CONFIG_SCENE2_3 = {
    'position': np.array([4.9, 0.0, 0.1]),
    'pitch': 25.0,
    'cameras': [
        {'name': 'left', 'yaw': -90.0},
        {'name': 'middle', 'yaw': 0.0},
        {'name': 'right', 'yaw': 90.0},
    ]
}

def get_rsf_camera_config(scene_name):

    if scene_name == 'Scene1':
        return RSF_CAMERA_CONFIG_SCENE1
    else:
        return RSF_CAMERA_CONFIG_SCENE2_3

CAMERA_PARAMS = {
    'width': 1920,
    'height': 1080,
    'fov': 100.0,
    'frame_interval': 0.05,
}

def compute_camera_intrinsic():

    width = CAMERA_PARAMS['width']
    height = CAMERA_PARAMS['height']
    fov = CAMERA_PARAMS['fov']

    fx = (width / 2) / np.tan(fov * np.pi / 360)
    fy = fx

    cx = width / 2
    cy = height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=float)

    return K

CAMERA_INTRINSIC = compute_camera_intrinsic()

def print_camera_intrinsic():

    print("="*60)
    print("相机内参矩阵 K:")
    print("="*60)
    print(CAMERA_INTRINSIC)
    print(f"\nfx = fy = {CAMERA_INTRINSIC[0, 0]:.2f}")
    print(f"cx = {CAMERA_INTRINSIC[0, 2]:.2f}")
    print(f"cy = {CAMERA_INTRINSIC[1, 2]:.2f}")
    print("="*60)

CSI_PARAMS = {
    'format': '128x32',
    'tx_antennas': 32,
    'rx_antennas': 128,
    'carrier_freq': 28e9,
    'wavelength': 3e8 / 28e9,
    'antenna_spacing': 3e8 / 28e9 / 2,
    'total_frames': 1500,
}

TRAIN_FRAMES = 1500
TEST_FRAMES_INPUT = 750
TEST_FRAMES_PREDICT = 750

TIMESTAMP_CONFIG = {
    'total_frames': 1500,
    'train_range': (1, 1500),
    'test_input_range': (1, 750),
    'test_predict_range': (751, 1500),
    'frame_interval': 0.05,
    'total_duration': 1500 * 0.05,
}

WORLD_COORDINATE_SYSTEM = {
    'name': 'NED',
    'description': 'North-East-Down',
    'x_axis': 'North',
    'y_axis': 'East',
    'z_axis': 'Down',
}

VEHICLE_COORDINATE_SYSTEM = {
    'name': 'Vehicle',
    'x_axis': 'Forward',
    'y_axis': 'Right',
    'z_axis': 'Down',
    'origin': 'Vehicle center',
}

RSF_COORDINATE_SYSTEM = {
    'name': 'RSF',
    'x_axis': 'Forward',
    'y_axis': 'Right',
    'z_axis': 'Down',
    'origin': 'RSF position in world',
}

FILE_TEMPLATES = {
    'csi': '{rsf}_{vehicle}_high_fencha_CSI.mat',
    'csi_alt': '{rsf}_{vehicle}_CSI.mat',
    'gt': '{vehicle}_train_cat_gt.txt',
    'noise': '{vehicle}_train_cat_noise.txt',
    'rsf_position': 'RSF_position_{scene_num}.txt',
}

PATH_TEMPLATES = {
    'scene': '{data_root}/{scene}',
    'vehicle': '{data_root}/{scene}/{vehicle}',
    'rsf': '{data_root}/{scene}/{rsf}',
    'rsf_csi': '{data_root}/{scene}/{rsf}/CSI',
    'rsf_rgb': '{data_root}/{scene}/{rsf}/RGB',
}

DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi

RSF_CLASSIFICATION = {
    'Scene1': {
        'intersection': ['RSF1'],
        'mid_road': ['RSF4', 'RSF7'],
    },
    'Scene2': {
        'intersection': ['RSF1', 'RSF3'],
        'mid_road': ['RSF7', 'RSF11'],
    },
    'Scene3': {
        'intersection': ['RSF1', 'RSF3', 'RSF11'],
        'mid_road': ['RSF7'],
    },
}

def get_rsf_type(scene_name, rsf_name):

    classification = RSF_CLASSIFICATION.get(scene_name, {})

    if rsf_name in classification.get('intersection', []):
        return 'intersection'
    elif rsf_name in classification.get('mid_road', []):
        return 'mid_road'
    else:
        return 'intersection'

def is_intersection_rsf(scene_name, rsf_name):

    return get_rsf_type(scene_name, rsf_name) == 'intersection'

def is_midroad_rsf(scene_name, rsf_name):

    return get_rsf_type(scene_name, rsf_name) == 'mid_road'
