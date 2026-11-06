"""Multiple GPU Version Fusion Feature Export. """

import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import time
from typing import Dict, List, Tuple
import sys

from config_infer import *

from models.network import GatedWiFo

from data.full_sequence import (
    load_vehicle_data_for_extraction,
    SCENE_CSI_SUFFIX,
    VEHICLE_NAME_MAPPING,
)

from utils.aggregate import (
    patch_to_frame_features,
    aggregate_window_features_to_frames,
    crop_features_by_gps,
)
from data.gps import get_vehicle_frame_info

from infer import FeatureExtractor

def collect_all_tasks(is_test=False):

    tasks = []

    if is_test:
        scenes = TEST_SCENES
        rsf_config = TEST_RSF_CONFIG
        data_root = TEST_DATA_ROOT
    else:
        scenes = TRAIN_SCENES
        rsf_config = TRAIN_RSF_CONFIG
        data_root = TRAIN_DATA_ROOT

    for scene in scenes:
        rsf_list = rsf_config.get(scene, [])

        for rsf in rsf_list:
            rsf_dir = data_root / scene / rsf / 'CSI'

            if not rsf_dir.exists():
                print(f"[WARNING] RSFDirectory does not exist: {rsf_dir}")
                continue

            csi_files = list(rsf_dir.glob('*.mat'))

            if not csi_files:
                print(f"[WARNING] {scene}/{rsf} NoneCSIFile")
                continue

            vehicles = set()
            for f in csi_files:
                parts = f.stem.split('_')
                if len(parts) >= 2:
                    vehicle = parts[1]
                    vehicles.add(vehicle)

            for vehicle in sorted(vehicles):
                tasks.append((scene, rsf, vehicle))

    return tasks

def process_task_list(
    gpu_id: int,
    task_list: List[Tuple[str, str, str]],
    checkpoint_path: Path,
    is_test: bool,
    progress_dict: dict = None
):

    device = f'cuda:{gpu_id}'
    print(f"\n[GPU {gpu_id}] Start Processing {len(task_list)} vehicles")
    print(f"[GPU {gpu_id}] Equipment: {device}")

    try:
        extractor = FeatureExtractor(
            checkpoint_path=checkpoint_path,
            config=EXTRACT_CONFIG,
            device=device
        )
    except Exception as e:
        print(f"[GPU {gpu_id}] [ERROR] Initialization failed: {e}")
        return

    success = 0
    failed = []

    for idx, (scene, rsf, vehicle) in enumerate(task_list):
        try:
            print(f"\n[GPU {gpu_id}] [{idx+1}/{len(task_list)}] {scene}/{rsf}/{vehicle}")

            if is_test:
                data_root = TEST_DATA_ROOT
                mask_root = TEST_MASK_ROOT
                rgb_root = TEST_RGB_ROOT
            else:
                data_root = TRAIN_DATA_ROOT
                mask_root = TRAIN_MASK_ROOT
                rgb_root = TRAIN_RGB_ROOT

            extractor.extract_vehicle_features(
                scene=scene,
                rsf=rsf,
                vehicle=vehicle,
                data_root=data_root,
                mask_root=mask_root,
                rgb_root=rgb_root,
                is_test=is_test
            )

            success += 1

            if progress_dict is not None:
                progress_dict[gpu_id] = (idx + 1, len(task_list), success)

        except Exception as e:
            print(f"[GPU {gpu_id}] [ERROR] {scene}/{rsf}/{vehicle} Failed: {e}")
            failed.append(f"{scene}/{rsf}/{vehicle}")

            import traceback
            traceback.print_exc()

    print(f"\n[GPU {gpu_id}] Done! ")
    print(f"[GPU {gpu_id}] Success: {success}/{len(task_list)}")
    print(f"[GPU {gpu_id}] Failed: {len(failed)}")

    if failed:
        print(f"[GPU {gpu_id}] Failure List:")
        for v in failed:
            print(f"  - {v}")

    return success, failed

def split_tasks_to_gpus(tasks: List, num_gpus: int) -> List[List]:

    task_lists = [[] for _ in range(num_gpus)]

    for idx, task in enumerate(tasks):
        gpu_id = idx % num_gpus
        task_lists[gpu_id].append(task)

    return task_lists

def main():

    parser = argparse.ArgumentParser(description='PGCF-RSU feature export')
    parser.add_argument('--num_gpus', type=int, default=8, help='UsedGPUQuantity')
    parser.add_argument('--gpu_ids', type=str, default=None, help='AssignGPU ID, Example "0,1,2,3"')
    parser.add_argument('--test_only', action='store_true', help='Only test sets are extracted')
    parser.add_argument('--train_only', action='store_true', help='Only extract training sets')
    args = parser.parse_args()

    print("=" * 70)
    print("PGCF-RSU Feature Export")
    print("=" * 70)

    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        num_gpus = len(gpu_ids)
    else:
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
        gpu_ids = list(range(num_gpus))

    print(f"\nUseGPU: {gpu_ids} ({num_gpus}Zhang)")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    for gpu_id in gpu_ids:
        if gpu_id >= torch.cuda.device_count():
            print(f"[ERROR] GPU {gpu_id} Absent! ")
            return

        device_name = torch.cuda.get_device_name(gpu_id)
        print(f"  GPU {gpu_id}: {device_name}")

    print_config_summary()

    print("\nValidate Configuration...")
    if not verify_config():
        print("\n⚠️  Configuration validation failed, Please check the path configuration! ")
        return

    if not args.test_only:
        print("\n" + "="*70)
        print("Collect Training Set Tasks...")
        print("="*70)

        train_tasks = collect_all_tasks(is_test=False)
        print(f"Total Tasks: {len(train_tasks)}")

        if train_tasks:
            train_task_lists = split_tasks_to_gpus(train_tasks, num_gpus)

            print("\nTask Assignment:")
            for gpu_id, task_list in zip(gpu_ids, train_task_lists):
                print(f"  GPU {gpu_id}: {len(task_list)} vehicles")

            print("\n" + "="*70)
            print("Start parallel extraction of training set features...")
            print("="*70)

            start_time = time.time()

            mp.set_start_method('spawn', force=True)

            processes = []
            for gpu_id, task_list in zip(gpu_ids, train_task_lists):
                if not task_list:
                    continue

                p = mp.Process(
                    target=process_task_list,
                    args=(gpu_id, task_list, CHECKPOINT_PATH, False)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            elapsed = time.time() - start_time
            print(f"\nTraining Set Extraction Complete! Time consuming: {elapsed/60:.1f} Minutes")

    if not args.train_only:
        print("\n" + "="*70)
        print("Collect Test Set Tasks...")
        print("="*70)

        test_tasks = collect_all_tasks(is_test=True)
        print(f"Total Tasks: {len(test_tasks)}")

        if test_tasks:
            test_task_lists = split_tasks_to_gpus(test_tasks, num_gpus)

            print("\nTask Assignment:")
            for gpu_id, task_list in zip(gpu_ids, test_task_lists):
                print(f"  GPU {gpu_id}: {len(task_list)} vehicles")

            print("\n" + "="*70)
            print("Start parallel extraction of test set features...")
            print("="*70)

            start_time = time.time()

            mp.set_start_method('spawn', force=True)

            processes = []
            for gpu_id, task_list in zip(gpu_ids, test_task_lists):
                if not task_list:
                    continue

                p = mp.Process(
                    target=process_task_list,
                    args=(gpu_id, task_list, CHECKPOINT_PATH, True)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            elapsed = time.time() - start_time
            print(f"\nTest Set Extraction Complete! Time consuming: {elapsed/60:.1f} Minutes")

    print("\n" + "="*70)
    print("All feature extraction completed! ")
    print("="*70)

    print("\nGenerate Extraction Summary...")
    try:
        extractor_temp = FeatureExtractor(
            checkpoint_path=CHECKPOINT_PATH,
            config=EXTRACT_CONFIG,
            device='cuda:0'
        )
        extractor_temp.save_extraction_summary()
        print("✓ Summary saved")
    except Exception as e:
        print(f"⚠️  Abstract generation failed: {e}")

if __name__ == '__main__':
    main()
