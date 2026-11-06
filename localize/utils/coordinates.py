"""车体高度修正、角度换算和路侧坐标到世界坐标。"""

import numpy as np
from .constants import get_vehicle_height_correction, is_aerial_vehicle, DEG_TO_RAD

def apply_height_correction(position, roll, pitch, yaw, scene_name, vehicle_name):

    position = np.array(position, dtype=float)

    height_correction = get_vehicle_height_correction(scene_name, vehicle_name)

    offset_vehicle = np.array([0.0, 0.0, -height_correction])

    R = rotation_matrix_from_euler(roll, pitch, yaw)
    offset_world = R @ offset_vehicle

    corrected_position = position + offset_world

    return corrected_position

def convert_angles_deg_to_rad(roll_deg, pitch_deg, yaw_deg, vehicle_name):

    if is_aerial_vehicle(vehicle_name):
        roll_rad = roll_deg * DEG_TO_RAD
        pitch_rad = pitch_deg * DEG_TO_RAD
        yaw_rad = yaw_deg * DEG_TO_RAD
    else:
        roll_rad = roll_deg
        pitch_rad = pitch_deg
        yaw_rad = yaw_deg

    return roll_rad, pitch_rad, yaw_rad

def rotation_matrix_from_euler(roll, pitch, yaw):

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    return R

def rsf_to_world(rsf_points, rsf_origin_world, rsf_roll, rsf_pitch, rsf_yaw):

    rsf_points = np.array(rsf_points, dtype=float)
    rsf_origin_world = np.array(rsf_origin_world, dtype=float)

    R = rotation_matrix_from_euler(rsf_roll, rsf_pitch, rsf_yaw)

    if rsf_points.ndim == 1:
        world_points = rsf_origin_world + R @ rsf_points
    else:
        world_points = rsf_origin_world + (R @ rsf_points.T).T

    return world_points

if __name__ == '__main__':
    print("="*80)
    print("测试坐标转换功能")
    print("="*80)

    print("1. 测试普通车辆高度修正 (Scene1/Car1):")
    position = np.array([190.5, -69.3, 0.0])
    roll, pitch, yaw = 0.01, -0.02, 1.57

    corrected_pos = apply_height_correction(
        position, roll, pitch, yaw, 'Scene1', 'Car1'
    )
    print(f"  原始位置: {position}")
    print(f"  修正后位置: {corrected_pos}")
    print(f"  高度变化: {corrected_pos[2] - position[2]:.3f}m")

    print("\n2. 测试AT类型角度转换:")
    roll_deg, pitch_deg, yaw_deg = 5.0, -2.0, 90.0

    roll_rad, pitch_rad, yaw_rad = convert_angles_deg_to_rad(
        roll_deg, pitch_deg, yaw_deg, 'AT1'
    )
    print(f"  原始角度（度）: {roll_deg}, {pitch_deg}, {yaw_deg}")
    print(f"  转换后（弧度）: {roll_rad:.4f}, {pitch_rad:.4f}, {yaw_rad:.4f}")
    print(f"  验证（度）: {roll_rad*180/np.pi:.1f}, {pitch_rad*180/np.pi:.1f}, {yaw_rad*180/np.pi:.1f}")

    print("\n3. 测试普通车辆角度（保持弧度制）:")
    roll_orig, pitch_orig, yaw_orig = 0.087, -0.035, 1.571

    roll_out, pitch_out, yaw_out = convert_angles_deg_to_rad(
        roll_orig, pitch_orig, yaw_orig, 'Car1'
    )
    print(f"  输入（弧度）: {roll_orig}, {pitch_orig}, {yaw_orig}")
    print(f"  输出（弧度）: {roll_out}, {pitch_out}, {yaw_out}")
    print(f"  是否相同: {np.allclose([roll_orig, pitch_orig, yaw_orig], [roll_out, pitch_out, yaw_out])}")

    print("\n4. 测试旋转矩阵:")
    R = rotation_matrix_from_euler(0, 0, np.pi/2)
    print(f"  绕Z轴旋转90度的旋转矩阵:")
    print(f"  {R}")

    det_R = np.linalg.det(R)
    is_orthogonal = np.allclose(R @ R.T, np.eye(3))
    print(f"  行列式: {det_R:.6f} (应该≈1)")
    print(f"  正交性: {is_orthogonal} (应该为True)")

    print("="*80)
    print("✅ 测试通过！")
