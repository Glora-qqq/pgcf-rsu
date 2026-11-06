"""Read fine-tuning phase export 512 Dimensions CSI Features. """

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def load_vehicle_wifo_features(
    features_root: Path,
    scene_name: str,
    vehicle_name: str,
    rsf_list: List[str],
    load_gps: bool = True,
    load_frame_info: bool = True,
) -> Dict:

    features_root = Path(features_root)
    vehicle_dir = features_root / scene_name / vehicle_name

    if not vehicle_dir.exists():
        raise ValueError(f"Vehicle directory does not exist: {vehicle_dir}")

    wifo_features = []
    for rsf_name in rsf_list:
        feature_file = vehicle_dir / f"{vehicle_name}_{rsf_name}_wifo_features.npy"
        if not feature_file.exists():
            raise ValueError(f"Feature file does not exist: {feature_file}")

        features = np.load(str(feature_file))
        wifo_features.append(features)

    T_list = [feat.shape[0] for feat in wifo_features]
    if len(set(T_list)) > 1:
        raise ValueError(f"DifferentRSFInconsistent frames: {T_list}")

    result = {
        'wifo_features': wifo_features,
        'scene': scene_name,
        'vehicle': vehicle_name,
        'rsf_list': rsf_list,
        'num_frames': T_list[0],
    }

    if load_gps:
        gps_file = vehicle_dir / f"{vehicle_name}_gps_coordinates.npy"
        if gps_file.exists():
            gps_coordinates = np.load(str(gps_file))
            result['gps_coordinates'] = gps_coordinates
        else:
            print(f"[WARNING] GPSFile does not exist: {gps_file}")
            result['gps_coordinates'] = None

    if load_frame_info:
        info_file = vehicle_dir / f"{vehicle_name}_frame_info.json"
        if info_file.exists():
            with open(str(info_file), 'r') as f:
                frame_info = json.load(f)
            result['frame_info'] = frame_info
        else:
            print(f"[WARNING] Frame information file does not exist: {info_file}")
            result['frame_info'] = None

    return result

def load_vehicle_wifo_features_scene1(
    features_root: Path,
    vehicle_name: str,
    load_gps: bool = True,
    load_frame_info: bool = True,
) -> Dict:

    rsf_list = ['RSF1', 'RSF4', 'RSF7']
    return load_vehicle_wifo_features(
        features_root=features_root,
        scene_name='Scene1',
        vehicle_name=vehicle_name,
        rsf_list=rsf_list,
        load_gps=load_gps,
        load_frame_info=load_frame_info,
    )

def load_vehicle_wifo_features_scene2_3(
    features_root: Path,
    scene_name: str,
    vehicle_name: str,
    load_gps: bool = True,
    load_frame_info: bool = True,
) -> Dict:

    if scene_name not in ['Scene2', 'Scene3']:
        raise ValueError(f"Scene name must be Scene2 or Scene3, Actual: {scene_name}")

    rsf_list = ['RSF1', 'RSF3', 'RSF7', 'RSF11']
    return load_vehicle_wifo_features(
        features_root=features_root,
        scene_name=scene_name,
        vehicle_name=vehicle_name,
        rsf_list=rsf_list,
        load_gps=load_gps,
        load_frame_info=load_frame_info,
    )

def load_scene_all_vehicles_wifo_features(
    features_root: Path,
    scene_name: str,
    rsf_list: Optional[List[str]] = None,
    load_gps: bool = True,
    load_frame_info: bool = True,
) -> Dict[str, Dict]:

    features_root = Path(features_root)
    scene_dir = features_root / scene_name

    if not scene_dir.exists():
        raise ValueError(f"Scene directory does not exist: {scene_dir}")

    if rsf_list is None:
        if scene_name == 'Scene1':
            rsf_list = ['RSF1', 'RSF4', 'RSF7']
        elif scene_name in ['Scene2', 'Scene3']:
            rsf_list = ['RSF1', 'RSF3', 'RSF7', 'RSF11']
        else:
            raise ValueError(f"Unknown Scenario: {scene_name}")

    all_vehicles = {}
    for vehicle_dir in scene_dir.iterdir():
        if not vehicle_dir.is_dir():
            continue

        vehicle_name = vehicle_dir.name

        try:
            vehicle_data = load_vehicle_wifo_features(
                features_root=features_root,
                scene_name=scene_name,
                vehicle_name=vehicle_name,
                rsf_list=rsf_list,
                load_gps=load_gps,
                load_frame_info=load_frame_info,
            )
            all_vehicles[vehicle_name] = vehicle_data
        except Exception as e:
            print(f"[WARNING] Load {scene_name}/{vehicle_name} Failed: {e}")

    return all_vehicles

def get_wifo_feature_by_frame_index(
    wifo_features: np.ndarray,
    frame_idx: int,
) -> np.ndarray:

    if frame_idx < 0 or frame_idx >= wifo_features.shape[0]:
        raise IndexError(f"Frame index out of range: {frame_idx}, Total frames: {wifo_features.shape[0]}")

    return wifo_features[frame_idx]

def get_wifo_features_by_frame_range(
    wifo_features: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:

    if start_idx < 0 or end_idx >= wifo_features.shape[0] or start_idx > end_idx:
        raise IndexError(f"Invalid frame range: [{start_idx}, {end_idx}], Total frames: {wifo_features.shape[0]}")

    return wifo_features[start_idx:end_idx+1]

def align_wifo_features_with_gt(
    wifo_features: np.ndarray,
    gt_frame_range: Tuple[int, int],
    vehicle_frame_range: Tuple[int, int],
) -> np.ndarray:

    vehicle_start, vehicle_end = vehicle_frame_range
    gt_start, gt_end = gt_frame_range

    relative_start = gt_start - vehicle_start
    relative_end = gt_end - vehicle_start

    if relative_start < 0 or relative_end >= wifo_features.shape[0]:
        raise ValueError(
            f"GTFrame range out of vehicle range: "
            f"GT=[{gt_start}, {gt_end}], "
            f"Vehicle=[{vehicle_start}, {vehicle_end}], "
            f"Relative index=[{relative_start}, {relative_end}], "
            f"Number of feature frames={wifo_features.shape[0]}"
        )

    return wifo_features[relative_start:relative_end+1]

if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import WIFO_FEATURES_ROOT

    print("="*80)
    print("WiFoFeature loader test")
    print("="*80)
    print(f"Feature Root (Fromconfig.py) : {WIFO_FEATURES_ROOT}\n")

    features_root = WIFO_FEATURES_ROOT

    print("\n[Test1] Scene1/Car1 Load (3PieceRSF) ")
    print("-" * 50)

    result = load_vehicle_wifo_features_scene1(
        features_root=features_root,
        vehicle_name='Car1',
    )

    print(f"Scenario: {result['scene']}")
    print(f"Vehicle: {result['vehicle']}")
    print(f"RSFList: {result['rsf_list']}")
    print(f"Total frames: {result['num_frames']}")
    print(f"\nWiFoNumber of features: {len(result['wifo_features'])}")
    for i, feat in enumerate(result['wifo_features']):
        print(f"  {result['rsf_list'][i]}: {feat.shape}")

    if result['gps_coordinates'] is not None:
        print(f"\nGPSCoordinates: {result['gps_coordinates'].shape}")

    if result['frame_info'] is not None:
        info = result['frame_info']
        print(f"\nFrame Information:")
        print(f"  Start Frame: {info['start_frame']}")
        print(f"  End Frame: {info['end_frame']}")
        print(f"  Total frames: {info['total_frames']}")

    print("\n" + "="*80)
    print("[Test2] Scene2/Car14 Load (4PieceRSF) ")
    print("-" * 50)

    result2 = load_vehicle_wifo_features_scene2_3(
        features_root=features_root,
        scene_name='Scene2',
        vehicle_name='Car14',
    )

    print(f"Scenario: {result2['scene']}")
    print(f"Vehicle: {result2['vehicle']}")
    print(f"RSFList: {result2['rsf_list']}")
    print(f"Total frames: {result2['num_frames']}")
    print(f"\nWiFoNumber of features: {len(result2['wifo_features'])}")
    for i, feat in enumerate(result2['wifo_features']):
        print(f"  {result2['rsf_list'][i]}: {feat.shape}")

    print("\n" + "="*80)
    print("[Test3] Scene3/AT1 Load (4PieceRSF) ")
    print("-" * 50)

    result3 = load_vehicle_wifo_features_scene2_3(
        features_root=features_root,
        scene_name='Scene3',
        vehicle_name='AT1',
    )

    print(f"Scenario: {result3['scene']}")
    print(f"Vehicle: {result3['vehicle']}")
    print(f"RSFList: {result3['rsf_list']}")
    print(f"Total frames: {result3['num_frames']}")

    print("\n" + "="*80)
    print("[Test4] Frame Index Access")
    print("-" * 50)

    wifo_feat = result['wifo_features'][0]
    print(f"Feature Shapes: {wifo_feat.shape}")

    frame_0 = get_wifo_feature_by_frame_index(wifo_feat, 0)
    print(f"Page0Frame characteristics: {frame_0.shape}")

    frames_0_9 = get_wifo_features_by_frame_range(wifo_feat, 0, 9)
    print(f"Page0-9Frame characteristics: {frames_0_9.shape}")

    print("\n" + "="*80)
    print("[Test5] WiFoFeatures vs.GTAlignment")
    print("-" * 50)

    info = result['frame_info']
    vehicle_frame_range = (info['start_frame'] - 1, info['end_frame'] - 1)
    gt_frame_range = vehicle_frame_range

    print(f"Vehicle Frame Range: {vehicle_frame_range} (0-based)")
    print(f"GTFrame Range: {gt_frame_range} (0-based)")

    aligned = align_wifo_features_with_gt(
        wifo_features=wifo_feat,
        gt_frame_range=gt_frame_range,
        vehicle_frame_range=vehicle_frame_range,
    )
    print(f"Post-alignment features: {aligned.shape}")
    print(f"Is same: {np.array_equal(aligned, wifo_feat)}")

    print("\n" + "="*80)
    print("[Test6] LoadScene1All vehicles")
    print("-" * 50)

    all_vehicles = load_scene_all_vehicles_wifo_features(
        features_root=features_root,
        scene_name='Scene1',
        load_gps=False,
        load_frame_info=True,
    )

    print(f"Scene1Number of vehicles: {len(all_vehicles)}")
    for vehicle_name, vehicle_data in all_vehicles.items():
        print(f"  {vehicle_name}: {vehicle_data['num_frames']}Frame, {len(vehicle_data['rsf_list'])}PieceRSF")

    print("\n" + "="*80)
    print("✅ All tests passed! ")
    print("="*80)
