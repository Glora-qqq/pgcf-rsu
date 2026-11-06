"""Body height correction, Angle conversion and roadside coordinates to world coordinates. """

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
