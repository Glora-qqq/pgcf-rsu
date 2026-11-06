"""with a trained fusion model, Combine the visual features of the three perspectives into one for each roadside unit 1024 Dimension Sequence. """

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_fusion import *
from models.fusion import create_fusion_model
from utils.common import get_view_angles, create_sample_list

def load_trained_model(checkpoint_path, device='cuda'):

    print(f"\nLoad Model: {checkpoint_path}")

    model = create_fusion_model(MODEL_CONFIG)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"  ✅ Model loaded successfully")
    print(f"  Training rounds: {checkpoint['epoch']}")
    print(f"  Training Losses: {checkpoint['train_loss']:.4f}")

    if 'val_loss' in checkpoint:
        print(f"  Validate loss: {checkpoint['val_loss']:.4f}")

    model = model.to(device)
    model.eval()

    return model

def load_three_views(feature_dir, scene_name, rsf_name):

    views = {}

    for view in ['left', 'middle', 'right']:
        filename = f"{scene_name}_{rsf_name}_{view}_env_feature.npy"
        filepath = os.path.join(feature_dir, scene_name, rsf_name, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature file does not exist: {filepath}")

        views[view] = np.load(filepath)

    return views

@torch.no_grad()
def fuse_single_sample(model, views_dict, scene_type, device):

    view_tensors = []
    for view_name in ['left', 'middle', 'right']:
        feat = torch.from_numpy(views_dict[view_name]).float()
        feat = feat.unsqueeze(0).to(device)
        view_tensors.append(feat)

    view_angles = get_view_angles(scene_type)

    fused = model(view_tensors, view_angles)

    fused_np = fused.squeeze(0).cpu().numpy()

    return fused_np

def infer_fusion_set(
    model,
    feature_dir,
    output_dir,
    scenes_config,
    device='cuda',
    save_views=False
):

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Batch Fusion Features")
    print(f"{'='*70}")
    print(f"Enter directory: {feature_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Equipment: {device}")

    samples = create_sample_list(scenes_config, feature_dir)
    print(f"Number of samples: {len(samples)}")

    success_count = 0
    failed_samples = []

    for scene_name, rsf_name, scene_type in tqdm(samples, desc="Convergence Progress"):
        try:
            views = load_three_views(feature_dir, scene_name, rsf_name)

            fused = fuse_single_sample(model, views, scene_type, device)

            scene_output_dir = os.path.join(output_dir, scene_name, rsf_name)
            os.makedirs(scene_output_dir, exist_ok=True)

            fused_filename = f"{scene_name}_{rsf_name}_fused_env_feature.npy"
            fused_path = os.path.join(scene_output_dir, fused_filename)
            np.save(fused_path, fused)

            if save_views:
                for view_name, view_feat in views.items():
                    view_filename = f"{scene_name}_{rsf_name}_{view_name}_env_feature.npy"
                    view_path = os.path.join(scene_output_dir, view_filename)
                    np.save(view_path, view_feat)

            success_count += 1

        except Exception as e:
            failed_samples.append((scene_name, rsf_name, str(e)))
            print(f"\n❌ Failed: {scene_name}/{rsf_name} - {e}")

    print(f"\n{'='*70}")
    print(f"✅ Fusion complete")
    print(f"  Success: {success_count}/{len(samples)}")

    if failed_samples:
        print(f"  Failed: {len(failed_samples)}")
        for scene, rsf, error in failed_samples:
            print(f"    - {scene}/{rsf}: {error}")

    print(f"{'='*70}\n")

    return success_count, failed_samples

def main():
    parser = argparse.ArgumentParser(description='Batch fusion of three perspectives features')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Trained model checkpoint path')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Enter Feature Directory (Default: train_featureandtest_feature) ')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (Default: fused_features) ')
    parser.add_argument('--dataset', type=str, default='both',
                       choices=['train', 'test', 'both'],
                       help='Which dataset to process')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Equipment (cuda or cpu)')
    parser.add_argument('--gpu_id', type=int, default=1,
                       help='GPUNo.')
    parser.add_argument('--save_views', action='store_true',
                       help='while preserving the original triangular features')

    args = parser.parse_args()

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        device = torch.device('cuda')
        print(f"UseGPU: {args.gpu_id}")
    else:
        device = torch.device('cpu')
        print("UseCPU")

    model = load_trained_model(args.checkpoint, device)

    datasets_to_process = []

    if args.dataset in ['train', 'both']:
        input_dir = args.input_dir if args.input_dir else TRAIN_FEATURE_DIR
        output_dir = args.output_dir if args.output_dir else os.path.join(FUSION_OUTPUT_DIR, 'train')
        datasets_to_process.append(('Training Sets', input_dir, output_dir))

    if args.dataset in ['test', 'both']:
        input_dir = args.input_dir if args.input_dir else TEST_FEATURE_DIR
        output_dir = args.output_dir if args.output_dir else os.path.join(FUSION_OUTPUT_DIR, 'test')
        datasets_to_process.append(('Test set', input_dir, output_dir))

    total_success = 0
    total_failed = []

    for dataset_name, input_dir, output_dir in datasets_to_process:
        print(f"\n{'='*70}")
        print(f"Process {dataset_name}")
        print(f"{'='*70}")

        success, failed = infer_fusion_set(
            model=model,
            feature_dir=input_dir,
            output_dir=output_dir,
            scenes_config=SCENES,
            device=device,
            save_views=args.save_views
        )

        total_success += success
        total_failed.extend(failed)

    print(f"\n{'='*70}")
    print(f"🎉 All done! ")
    print(f"  Total Success: {total_success}")
    print(f"  Total Failures: {len(total_failed)}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
