"""The position of the roadside antenna in the world coordinate system. """

import numpy as np
from .coordinates import rsf_to_world
from .constants import get_rsf_antenna_position, DEG_TO_RAD

def convert_rsf_extrinsic_to_radians(rsf_extrinsic_dict: dict) -> np.ndarray:

    x = float(rsf_extrinsic_dict['X'])
    y = float(rsf_extrinsic_dict['Y'])
    z = float(rsf_extrinsic_dict['Z'])
    roll = float(rsf_extrinsic_dict['Roll']) * DEG_TO_RAD
    pitch = float(rsf_extrinsic_dict['Pitch']) * DEG_TO_RAD
    yaw = float(rsf_extrinsic_dict['Yaw']) * DEG_TO_RAD

    return np.array([x, y, z, roll, pitch, yaw], dtype=float)

def convert_rsf_extrinsic_batch(rsf_dict: dict) -> dict:

    rsf_numpy_dict = {}

    for rsf_name, rsf_data in rsf_dict.items():
        rsf_numpy_dict[rsf_name] = convert_rsf_extrinsic_to_radians(rsf_data)

    return rsf_numpy_dict

def get_rsf_antenna_world_position(
    rsf_extrinsic: np.ndarray,
    scene_name: str
) -> np.ndarray:

    rsf_position_world = rsf_extrinsic[:3]
    rsf_roll = rsf_extrinsic[3]
    rsf_pitch = rsf_extrinsic[4]
    rsf_yaw = rsf_extrinsic[5]

    antenna_position_rsf = get_rsf_antenna_position(scene_name)

    antenna_position_world = rsf_to_world(
        antenna_position_rsf,
        rsf_position_world,
        rsf_roll,
        rsf_pitch,
        rsf_yaw
    )

    return antenna_position_world

def get_rsf_antenna_world_position_batch(
    rsf_extrinsic_list: list,
    scene_name: str
) -> list:

    antenna_position_world_list = []

    for rsf_extrinsic in rsf_extrinsic_list:
        antenna_position_world = get_rsf_antenna_world_position(
            rsf_extrinsic,
            scene_name
        )
        antenna_position_world_list.append(antenna_position_world)

    return antenna_position_world_list

def get_rsf_full_extrinsic(
    rsf_extrinsic: np.ndarray,
    scene_name: str
) -> np.ndarray:

    antenna_position_world = get_rsf_antenna_world_position(
        rsf_extrinsic,
        scene_name
    )

    full_extrinsic = np.concatenate([
        rsf_extrinsic,
        antenna_position_world
    ])

    return full_extrinsic

def get_rsf_extrinsic_and_antenna_features(
    rsf_extrinsic: np.ndarray,
    scene_name: str
) -> dict:

    rsf_origin = rsf_extrinsic.copy()

    antenna_position = get_rsf_antenna_world_position(rsf_extrinsic, scene_name)

    full_extrinsic = np.concatenate([rsf_origin, antenna_position])

    return {
        'rsf_origin': rsf_origin,
        'antenna_position': antenna_position,
        'full_extrinsic': full_extrinsic,
    }
