"""Read the location and external reference of the roadside unit. """

import os
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import DEG_TO_RAD
from utils.antenna import get_rsf_antenna_world_position

def load_rsf_positions(
    rsf_file_path: Path
) -> Optional[Dict[str, Dict[str, float]]]:

    try:
        with open(rsf_file_path, 'r', encoding='utf-8') as f:
            rsf_dict = json.load(f)

        for rsf_name, rsf_data in rsf_dict.items():
            required_keys = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
            for key in required_keys:
                if key not in rsf_data:
                    print(f"[WARNING] RSF {rsf_name} Missing field: {key}")

        return rsf_dict

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSONParsing failed: {rsf_file_path}")
        print(f"        Error: {e}")
        print("        Please check that the file format is correct (full JSON, including the opening and closing braces)")
        return None
    except Exception as e:
        print(f"[ERROR] LoadRSFLocation file failed: {rsf_file_path}")
        print(f"        Error: {e}")
        return None

def load_scene_rsf_positions(
    data_root: Path,
    scene_name: str
) -> Optional[Dict[str, Dict[str, float]]]:

    scene_num = scene_name.replace('Scene_', '').replace('Scene', '')

    rsf_file = data_root / scene_name / f"RSF_position_{scene_num}.txt"

    if not rsf_file.exists():
        print(f"[WARNING] Not foundRSFLocation File:")
        print(f"           Scenario={scene_name}")
        print(f"           Path={rsf_file}")
        return None

    return load_rsf_positions(rsf_file)

def rsf_dict_to_numpy(rsf_dict: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:

    rsf_numpy = {}

    for rsf_name, rsf_data in rsf_dict.items():
        x = float(rsf_data['X'])
        y = float(rsf_data['Y'])
        z = float(rsf_data['Z'])
        roll = float(rsf_data['Roll']) * DEG_TO_RAD
        pitch = float(rsf_data['Pitch']) * DEG_TO_RAD
        yaw = float(rsf_data['Yaw']) * DEG_TO_RAD

        rsf_numpy[rsf_name] = np.array([x, y, z, roll, pitch, yaw], dtype=float)

    return rsf_numpy

def get_rsf_pose(
    rsf_dict: Dict[str, Dict[str, float]],
    rsf_name: str
) -> Optional[tuple]:

    if rsf_name not in rsf_dict:
        print(f"[WARNING] RSF {rsf_name} Not in dictionary")
        return None

    rsf_data = rsf_dict[rsf_name]

    position = np.array([
        float(rsf_data['X']),
        float(rsf_data['Y']),
        float(rsf_data['Z'])
    ], dtype=float)

    orientation = np.array([
        float(rsf_data['Roll']) * DEG_TO_RAD,
        float(rsf_data['Pitch']) * DEG_TO_RAD,
        float(rsf_data['Yaw']) * DEG_TO_RAD
    ], dtype=float)

    return position, orientation

def get_rsf_with_antenna_positions(
    data_root: Path,
    scene_name: str,
    rsf_list: list
) -> tuple:

    rsf_dict = load_scene_rsf_positions(data_root, scene_name)

    if rsf_dict is None:
        raise ValueError(f"Unable to load scenes {scene_name} ofRSFExternal Ginseng")

    rsf_extrinsic_list = []
    rsf_antenna_list = []

    for rsf_name in rsf_list:
        position, orientation = get_rsf_pose(rsf_dict, rsf_name)
        if position is None:
            raise ValueError(f"RSFExternal parameter does not exist: {rsf_name}")

        rsf_extrinsic = np.concatenate([position, orientation])
        rsf_extrinsic_list.append(rsf_extrinsic)

        antenna_position = get_rsf_antenna_world_position(
            rsf_extrinsic,
            scene_name
        )
        rsf_antenna_list.append(antenna_position)

    return rsf_extrinsic_list, rsf_antenna_list

if __name__ == '__main__':
    data_root = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[3] / 'data')) / 'train'

    print("="*80)
    print("TestScene1 RSFLocation")
    print("="*80)
    rsf_dict = load_scene_rsf_positions(data_root, 'Scene1')

    if rsf_dict is not None:
        print(f"Loaded successfully, Contains {len(rsf_dict)} PieceRSF")

        for rsf_name, rsf_data in rsf_dict.items():
            print(f"\n{rsf_name}:")
            print(f"  Raw Data: {rsf_data}")

            position, orientation = get_rsf_pose(rsf_dict, rsf_name)
            if position is not None:
                print(f"  Location (m): {position}")
                print(f"  Posture (Radian): {orientation}")
                print(f"  Posture (degrees): {orientation * 180 / np.pi}")

        rsf_numpy = rsf_dict_to_numpy(rsf_dict)
        print(f"\nConvertednumpyFormat:")
        for rsf_name, rsf_array in rsf_numpy.items():
            print(f"  {rsf_name}: {rsf_array}")
