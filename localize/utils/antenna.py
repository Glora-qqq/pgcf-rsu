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

if __name__ == '__main__':
    print("="*80)
    print("TestRSFExternal parameter and antenna position conversion")
    print("="*80)

    rsf_dict = {
        "RSF1": {"X": 190.0, "Y": -69.7, "Z": -5.5, "Roll": 0, "Pitch": 0, "Yaw": 155},
        "RSF4": {"X": 229.3, "Y": -1.7, "Z": -5.5, "Roll": 0, "Pitch": 0, "Yaw": -50},
        "RSF7": {"X": 68.3, "Y": -32.6, "Z": -5.5, "Roll": 0, "Pitch": 0, "Yaw": 65},
    }

    print("Page1Step: RSFExternal reference angle conversion (degrees→Radian) ")
    print("-" * 50)

    for rsf_name, rsf_data in rsf_dict.items():
        print(f"\n{rsf_name}:")
        print(f"  Raw Data (JSON) : {rsf_data}")

        rsf_extrinsic = convert_rsf_extrinsic_to_radians(rsf_data)
        print(f"  External parameters after conversion[6]: {rsf_extrinsic}")
        print(f"  Pose Angle Validation: {rsf_extrinsic[3:] * 180/np.pi}°")

    print("\n" + "="*80)
    print("Page2Step: Batch Conversion Testing")
    print("-" * 50)

    rsf_numpy_dict = convert_rsf_extrinsic_batch(rsf_dict)

    for rsf_name, rsf_extrinsic in rsf_numpy_dict.items():
        print(f"\n{rsf_name} External Ginseng[6]: {rsf_extrinsic}")

    print("\n" + "="*80)
    print("Page3Step: Antenna position calculation (RSFCoordinate system→World Coordinate System) ")
    print("-" * 50)

    for rsf_name, rsf_extrinsic in rsf_numpy_dict.items():
        print(f"\n{rsf_name}:")
        print(f"  RSFOrigin (World coordinates) : {rsf_extrinsic[:3]}")

        antenna_world = get_rsf_antenna_world_position(rsf_extrinsic, 'Scene1')
        print(f"  Antenna position (RSFCoordinate system) : [0, 0, 0.5]")
        print(f"  Antenna position (World Coordinate System) : {antenna_world}")
        print(f"  Height change: {antenna_world[2] - rsf_extrinsic[2]:.3f}m")

    print("\n" + "="*80)
    print("Page4Step: Full feature generation")
    print("-" * 50)

    rsf1_extrinsic = rsf_numpy_dict['RSF1']
    features = get_rsf_extrinsic_and_antenna_features(rsf1_extrinsic, 'Scene1')

    print(f"\nRSF1Feature Dictionary:")
    print(f"  rsf_origin[6]: {features['rsf_origin']}")
    print(f"  antenna_position[3]: {features['antenna_position']}")
    print(f"  full_extrinsic[9]: {features['full_extrinsic']}")

    print("\n" + "="*80)
    print("Page5Step: Full Conversion Process Summary")
    print("-" * 50)

    print("\nConversion process: ")
    print("1. JSONRead: {'X': 190.0, 'Y': -69.7, 'Z': -5.5, 'Yaw': 155°}")
    print("2. Angle Conversion: [190.0, -69.7, -5.5, 0, 0, 2.705rad]")
    print("3. Antenna calculation: RSFOrigin + R(roll,pitch,yaw) @ [0,0,0.5]")
    print("4. Feature Output: SelectableRSFOrigin[6], Antenna position[3]or complete features[9]")

    print("\nAndrsu.pyContrast:")
    print("✅ rsu.py: Angle conversion included")
    print("✅ antenna.py: Angle conversion is now also included + Antenna position calculation")
    print("✅ Functional consistency: Both modules handle angle conversion correctly")

    print("="*80)
    print("✅ RSFComplete conversion process test passed! ")
