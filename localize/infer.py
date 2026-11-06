"""Load positioning weights, Output vehicle trajectory for three scenarios. """

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from scipy.signal import savgol_filter

sys.path.append(str(Path(__file__).parent))

from config_test import *
from config import TrainConfig
from data.test_set import MultiRSFPose6DTestDataset, collate_test_batch
from train import PoseFusionModel

USE_POST_AT_UKF = False
POST_AT_UKF_ENABLED_VEHICLES = ['AT1', 'AT6']
POST_AT_UKF_FILTER_ORIENTATION = False
POST_AT_UKF_CONFIG = {
    'process_noise': 0.1,
    'process_noise_ori': 0.01,
    'measurement_noise': 0.8,
    'measurement_noise_ori': 0.05,
    'dt': 0.05,
}

ENABLE_OUTPUT_SG_SMOOTH = True
OUTPUT_SG_WINDOW_CAR = 11
OUTPUT_SG_POLYORDER_CAR = 3
OUTPUT_SG_WINDOW_AT = 51
OUTPUT_SG_POLYORDER_AT = 3

try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False

def apply_post_car_kalman(positions, config):

    from utils.kalman import KalmanFilter6D

    if not KALMAN_AVAILABLE:
        print(f"  [Post-processing] SkipCarFilter (filterpyNot Installed) ")
        positions[:, 2] = 0.0
        return positions

    process_noise = getattr(config, 'POST_CAR_PROCESS_NOISE', 0.10)
    measurement_noise = getattr(config, 'POST_CAR_MEASUREMENT_NOISE', 1.6)

    T = len(positions)
    poses_6d = np.zeros((T, 6))
    poses_6d[:, :3] = positions

    kf = KalmanFilter6D(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        smooth_orientation=False
    )
    smoothed_6d = kf.smooth(poses_6d)

    smoothed_positions = smoothed_6d[:, :3].copy()
    smoothed_positions[:, 2] = 0.0

    print(f"  [Post-processing] Car 2DKalman filtering complete (XYNoise Reduction, Z=0) ")
    print(f"    - Parameter: Q={process_noise}, R={measurement_noise}")

    return smoothed_positions

def apply_post_at_ukf(positions, config):

    try:
        from utils.ukf import UKF6DFilter
    except ImportError:
        print(f"  [Post-processing] SkipAT UKFFilter (UKFModule not installed) ")
        return positions

    process_noise = getattr(config, 'POST_AT_UKF_CONFIG', {}).get('process_noise', 0.1)
    measurement_noise = getattr(config, 'POST_AT_UKF_CONFIG', {}).get('measurement_noise', 1.6)
    dt = getattr(config, 'POST_AT_UKF_CONFIG', {}).get('dt', 0.05)

    T = len(positions)
    poses_6d = np.zeros((T, 6))
    poses_6d[:, :3] = positions

    ukf = UKF6DFilter(
        process_noise_pos=process_noise,
        process_noise_ori=0.01,
        measurement_noise_pos=measurement_noise,
        measurement_noise_ori=0.05,
        dt=dt
    )
    smoothed_6d = ukf.filter_trajectory(poses_6d)

    smoothed_positions = smoothed_6d[:, :3].copy()

    print(f"  [Post-processing] AT 3D UKFFilter complete (XYZNoise Reduction) ")
    print(f"    - Parameter: Q={process_noise}, R={measurement_noise}, dt={dt}")

    return smoothed_positions

def _sg_smooth(data, window, poly):
    data = np.asarray(data, dtype=float)
    if window % 2 == 0 or window <= poly or len(data) < window:
        return data
    try:
        return savgol_filter(data, window_length=window, polyorder=poly)
    except Exception:
        return data

def smooth_positions_xy(positions, window, poly):

    out = positions.copy()
    out[:, 0] = _sg_smooth(out[:, 0], window, poly)
    out[:, 1] = _sg_smooth(out[:, 1], window, poly)
    return out

def smooth_positions_xyz(positions, window, poly):

    out = positions.copy()
    for i in range(3):
        out[:, i] = _sg_smooth(out[:, i], window, poly)
    return out

def load_model(checkpoint_path, config, device):

    print(f"\n[Load Model] {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    ckpt_cfg = checkpoint.get('config') if isinstance(checkpoint, dict) else None
    if isinstance(ckpt_cfg, dict):
        modality = str(ckpt_cfg.get('ablation_modality', getattr(config, 'ablation_modality', 'full')) or 'full').lower()
        if modality != str(getattr(config, 'ablation_modality', 'full')).lower():
            print(f"  [Sync] checkpoint.ablation_modality={modality} → Override Test config")
            config.ablation_modality = modality
        if 'geo_proj_hidden' in ckpt_cfg:
            config.geo_proj_hidden = ckpt_cfg['geo_proj_hidden']

    model = PoseFusionModel(config).to(device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model weights loaded successfully")

        if 'epoch' in checkpoint:
            print(f"  - Training rounds: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"  - Best Validation Loss: {checkpoint['best_val_loss']:.4f}")
        print(f"  - ablation_modality: {getattr(config, 'ablation_modality', 'full')}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ Model weights loaded successfully (Direct loading) ")

    model.eval()

    return model

def test_single_vehicle(model, test_loader, device, scene_name, vehicle_name, output_dir, config):

    print(f"\n{'='*60}")
    print(f"Test Vehicle: {scene_name}/{vehicle_name}")
    print(f"{'='*60}")

    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"{scene_name}/{vehicle_name}"):
            rsf_data_dict = batch['rsf_data_dict']
            gps_poses_6d = batch['gps_poses_6d'].to(device)
            scene_name_batch = batch['scene_name']
            vehicle_names = batch['vehicle_names']
            frame_ids = batch['frame_ids']

            for rsf_name in rsf_data_dict:
                rsf_data_dict[rsf_name]['csi_features'] = rsf_data_dict[rsf_name]['csi_features'].to(device)
                rsf_data_dict[rsf_name]['spatial_features'] = rsf_data_dict[rsf_name]['spatial_features'].to(device)

            outputs = model(
                rsf_data_dict=rsf_data_dict,
                gps_poses_6d=gps_poses_6d,
                scene_name=scene_name_batch,
                vehicle_names=vehicle_names
            )

            predicted_positions = outputs['pose_dict']['positions'].cpu().numpy()

            all_predictions = predicted_positions.squeeze(0)
            all_frame_ids = frame_ids.numpy()

            break

    is_car = not vehicle_name.startswith('AT')

    if USE_POST_PROCESSING_FILTER and is_car:
        print(f"\n  [Post-processing] CarVehicle: Apply2DKalman filtering...")
        all_predictions = apply_post_car_kalman(
            all_predictions,
            config=type('Config', (), {
                'POST_CAR_PROCESS_NOISE': POST_CAR_PROCESS_NOISE,
                'POST_CAR_MEASUREMENT_NOISE': POST_CAR_MEASUREMENT_NOISE,
            })
        )
    elif is_car:
        all_predictions[:, 2] = 0.0
        print(f"  [Post-processing] SkipCarFilter, ZSet to0")
    else:
        if USE_POST_AT_UKF and vehicle_name in POST_AT_UKF_ENABLED_VEHICLES:
            print(f"\n  [Post-processing] ATVehicle {vehicle_name}: Apply3D UKFFilter...")
            all_predictions = apply_post_at_ukf(
                all_predictions,
                config=type('Config', (), {
                    'POST_AT_UKF_CONFIG': POST_AT_UKF_CONFIG,
                })
            )
        else:
            print(f"  [Post-processing] ATVehicle {vehicle_name}: Predict output directly using the model")

    if ENABLE_OUTPUT_SG_SMOOTH:
        if is_car:
            all_predictions = smooth_positions_xy(all_predictions, OUTPUT_SG_WINDOW_CAR, OUTPUT_SG_POLYORDER_CAR)
            print(f"  [Smoothing] Car SGSmooth Done (window={OUTPUT_SG_WINDOW_CAR}, poly={OUTPUT_SG_POLYORDER_CAR})")
        else:
            all_predictions = smooth_positions_xyz(all_predictions, OUTPUT_SG_WINDOW_AT, OUTPUT_SG_POLYORDER_AT)
            print(f"  [Smoothing] AT SGSmooth Done (window={OUTPUT_SG_WINDOW_AT}, poly={OUTPUT_SG_POLYORDER_AT})")

    scene_dir = output_dir / SCENE_DIR_MAPPING[scene_name]
    scene_dir.mkdir(parents=True, exist_ok=True)

    output_file = scene_dir / OUTPUT_FILENAME_TEMPLATE.format(vehicle=vehicle_name)

    with open(output_file, 'w') as f:
        for i in range(len(all_predictions)):
            x, y, z = all_predictions[i]
            frame_id = int(all_frame_ids[i])

            f.write(f"{x:.{POSITION_DECIMAL_PLACES}f} "
                   f"{y:.{POSITION_DECIMAL_PLACES}f} "
                   f"{z:.{POSITION_DECIMAL_PLACES}f} "
                   f"{frame_id}\n")

    print(f"✓ Forecast Complete: {len(all_predictions)} Frame")
    print(f"✓ Save to: {output_file}")

    return output_file

def test_all_vehicles(model, config, device):

    print(f"\n{'='*60}")
    print("Start testing all vehicles")
    print(f"{'='*60}")

    results = {
        'total_vehicles': 0,
        'success_vehicles': 0,
        'failed_vehicles': [],
        'output_files': []
    }

    test_vehicles = TEST_VEHICLES

    total_vehicles = sum(len(vehicles) for vehicles in test_vehicles.values())
    results['total_vehicles'] = total_vehicles

    print(f"\nTotal: {total_vehicles} cars")
    for scene, vehicles in test_vehicles.items():
        print(f"  {scene}: {len(vehicles)} cars")

    for scene_name, vehicles in test_vehicles.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scene_name}")
        print(f"{'='*60}")

        for vehicle_name in vehicles:
            try:
                test_dataset = MultiRSFPose6DTestDataset(
                    test_data_root=TEST_DATA_ROOT,
                    wifo_features_root=TEST_WIFO_FEATURES_ROOT,
                    scene_name=scene_name,
                    vehicle_name=vehicle_name,
                    rsf_list=SCENE_RSF_CONFIG[scene_name],
                    use_input_kalman=USE_INPUT_KALMAN,
                    input_kalman_config=INPUT_KALMAN_CONFIG,
                    config=config,
                )

                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=TEST_BATCH_SIZE,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_test_batch
                )

                output_file = test_single_vehicle(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    scene_name=scene_name,
                    vehicle_name=vehicle_name,
                    output_dir=TEST_OUTPUT_ROOT,
                    config=config
                )

                results['success_vehicles'] += 1
                results['output_files'].append(str(output_file))

            except Exception as e:
                print(f"✗ Test failed: {scene_name}/{vehicle_name}")
                print(f"  Error: {e}")
                results['failed_vehicles'].append({
                    'scene': scene_name,
                    'vehicle': vehicle_name,
                    'error': str(e)
                })

    return results

def save_test_summary(results, config):

    summary_file = TEST_OUTPUT_ROOT / 'test_summary.json'

    summary = {
        'config': {
            'checkpoint': str(MODEL_CHECKPOINT_PATH),
            'test_data_root': str(TEST_DATA_ROOT),
            'wifo_features_root': str(TEST_WIFO_FEATURES_ROOT),
        },
        'results': {
            'total_vehicles': results['total_vehicles'],
            'success_vehicles': results['success_vehicles'],
            'failed_vehicles': results['failed_vehicles'],
        },
        'output_files': results['output_files']
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Test Summary Saved: {summary_file}")

def main():

    print("=" * 80)
    print("PGCF-RSU Positioning inference")
    print("=" * 80)

    print(f"\n[Configuration Information]")
    print(f"  Test Data Root: {TEST_DATA_ROOT}")
    print(f"  WiFoFeature Root: {TEST_WIFO_FEATURES_ROOT}")
    print(f"  Model checkpoint: {MODEL_CHECKPOINT_PATH}")
    print(f"  Output directory: {TEST_OUTPUT_ROOT}")
    print(f"  Equipment: {TEST_DEVICE}")
    print(f"\n[Filter Configuration]")
    print(f"  InputCar: {'2D KalmanFilter' if USE_INPUT_KALMAN else 'No filtering'}")
    print(f"  InputAT: {'3D UKFFilter' if USE_AT_UKF else 'No filtering'}")
    print(f"  Output end post-processing:")
    if USE_POST_PROCESSING_FILTER:
        print(f"    ✅ Enable (Car + ATVehicle) ")
        print(f"       - Car: 2D KalmanPost-processing (Q={POST_CAR_PROCESS_NOISE}, R={POST_CAR_MEASUREMENT_NOISE})")
        if USE_POST_AT_UKF:
            print(f"       - AT: 3D UKFPost-processing (Q={POST_AT_UKF_CONFIG.get('process_noise', 0.1)}, R={POST_AT_UKF_CONFIG.get('measurement_noise', 1.6)}, dt={POST_AT_UKF_CONFIG.get('dt', 0.05)})")
        else:
            print(f"       - AT: Predict output directly using the model")
    else:
        print(f"    ❌ Close")
    print(f"\n✓ All vehicles in all scenarios will be tested (of22Vehicles)")

    if not TEST_DATA_ROOT.exists():
        raise FileNotFoundError(f"Test data root directory does not exist: {TEST_DATA_ROOT}")
    if not TEST_WIFO_FEATURES_ROOT.exists():
        raise FileNotFoundError(f"WiFoFeature root directory does not exist: {TEST_WIFO_FEATURES_ROOT}")
    if not MODEL_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {MODEL_CHECKPOINT_PATH}")

    config = TrainConfig(
        use_adaptive_kalman=True,
        use_input_kalman=USE_INPUT_KALMAN,
        input_kalman_process_noise=INPUT_KALMAN_CONFIG['process_noise'],
        input_kalman_measurement_noise=INPUT_KALMAN_CONFIG['measurement_noise'],

        use_at_ukf=USE_AT_UKF,
        at_ukf_orientation=AT_UKF_FILTER_ORIENTATION,
        at_ukf_process_noise=AT_UKF_CONFIG['process_noise'],
        at_ukf_process_noise_ori=AT_UKF_CONFIG['process_noise_ori'],
        at_ukf_measurement_noise=AT_UKF_CONFIG['measurement_noise'],
        at_ukf_measurement_noise_ori=AT_UKF_CONFIG['measurement_noise_ori'],
        at_ukf_dt=AT_UKF_CONFIG['dt'],
        ablation_modality=ABLATION_MODALITY,
    )

    print(f"✓ Configuration loaded successfully")
    print(f"  - CarVehicle: {'2D KalmanFilter' if USE_INPUT_KALMAN else 'No filtering'}")
    if USE_AT_UKF:
        print(f"  - ATVehicle: 3D UKFFilter (Q={AT_UKF_CONFIG['process_noise']}, R={AT_UKF_CONFIG['measurement_noise']}) ✅")
    else:
        print(f"  - ATVehicle: No filtering")

    device = torch.device(TEST_DEVICE)
    model = load_model(MODEL_CHECKPOINT_PATH, config, device)

    results = test_all_vehicles(model, config, device)

    print(f"\n{'='*80}")
    print("Test Complete")
    print(f"{'='*80}")
    print(f"Total: {results['total_vehicles']} cars")
    print(f"Success: {results['success_vehicles']} cars")
    print(f"Failed: {len(results['failed_vehicles'])} cars")

    if results['failed_vehicles']:
        print(f"\nFailed Vehicle:")
        for failed in results['failed_vehicles']:
            print(f"  - {failed['scene']}/{failed['vehicle']}: {failed['error']}")

    save_test_summary(results, config)

    print(f"\n{'='*80}")
    print("All tests completed! ")
    print(f"Forecast results saved in: {TEST_OUTPUT_ROOT}")
    print(f"  - Scene_1/: {len([v for v in TEST_VEHICLES['Scene1']])} files")
    print(f"  - Scene_2/: {len([v for v in TEST_VEHICLES['Scene2']])} files")
    print(f"  - Scene_3/: {len([v for v in TEST_VEHICLES['Scene3']])} files")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
