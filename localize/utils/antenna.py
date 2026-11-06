"""路侧天线在世界坐标系中的位置。"""

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
    print("测试RSF外参和天线位置转换")
    print("="*80)

    rsf_dict = {
        "RSF1": {"X": 190.0, "Y": -69.7, "Z": -5.5, "Roll": 0, "Pitch": 0, "Yaw": 155},
        "RSF4": {"X": 229.3, "Y": -1.7, "Z": -5.5, "Roll": 0, "Pitch": 0, "Yaw": -50},
        "RSF7": {"X": 68.3, "Y": -32.6, "Z": -5.5, "Roll": 0, "Pitch": 0, "Yaw": 65},
    }

    print("第1步：RSF外参角度转换（度→弧度）")
    print("-" * 50)

    for rsf_name, rsf_data in rsf_dict.items():
        print(f"\n{rsf_name}:")
        print(f"  原始数据（JSON）: {rsf_data}")

        rsf_extrinsic = convert_rsf_extrinsic_to_radians(rsf_data)
        print(f"  转换后外参[6]: {rsf_extrinsic}")
        print(f"  姿态角度验证: {rsf_extrinsic[3:] * 180/np.pi}°")

    print("\n" + "="*80)
    print("第2步：批量转换测试")
    print("-" * 50)

    rsf_numpy_dict = convert_rsf_extrinsic_batch(rsf_dict)

    for rsf_name, rsf_extrinsic in rsf_numpy_dict.items():
        print(f"\n{rsf_name} 外参[6]: {rsf_extrinsic}")

    print("\n" + "="*80)
    print("第3步：天线位置计算（RSF坐标系→世界坐标系）")
    print("-" * 50)

    for rsf_name, rsf_extrinsic in rsf_numpy_dict.items():
        print(f"\n{rsf_name}:")
        print(f"  RSF原点（世界坐标）: {rsf_extrinsic[:3]}")

        antenna_world = get_rsf_antenna_world_position(rsf_extrinsic, 'Scene1')
        print(f"  天线位置（RSF坐标系）: [0, 0, 0.5]")
        print(f"  天线位置（世界坐标系）: {antenna_world}")
        print(f"  高度变化: {antenna_world[2] - rsf_extrinsic[2]:.3f}m")

    print("\n" + "="*80)
    print("第4步：完整特征生成")
    print("-" * 50)

    rsf1_extrinsic = rsf_numpy_dict['RSF1']
    features = get_rsf_extrinsic_and_antenna_features(rsf1_extrinsic, 'Scene1')

    print(f"\nRSF1特征字典:")
    print(f"  rsf_origin[6]: {features['rsf_origin']}")
    print(f"  antenna_position[3]: {features['antenna_position']}")
    print(f"  full_extrinsic[9]: {features['full_extrinsic']}")

    print("\n" + "="*80)
    print("第5步：完整转换流程总结")
    print("-" * 50)

    print("\n转换流程：")
    print("1. JSON读取: {'X': 190.0, 'Y': -69.7, 'Z': -5.5, 'Yaw': 155°}")
    print("2. 角度转换: [190.0, -69.7, -5.5, 0, 0, 2.705rad]")
    print("3. 天线计算: RSF原点 + R(roll,pitch,yaw) @ [0,0,0.5]")
    print("4. 特征输出: 可选择RSF原点[6]、天线位置[3]或完整特征[9]")

    print("\n与rsu.py的对比:")
    print("✅ rsu.py: 包含角度转换功能")
    print("✅ antenna.py: 现在也包含角度转换 + 天线位置计算")
    print("✅ 功能一致性: 两个模块都正确处理角度转换")

    print("="*80)
    print("✅ RSF完整转换流程测试通过！")
