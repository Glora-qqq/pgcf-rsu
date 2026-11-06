"""用训练好的融合模型，把三个视角的视觉特征合成每个路侧单元一条 1024 维序列。"""

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

    print(f"\n加载模型: {checkpoint_path}")

    model = create_fusion_model(MODEL_CONFIG)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"  ✅ 模型加载成功")
    print(f"  训练轮次: {checkpoint['epoch']}")
    print(f"  训练损失: {checkpoint['train_loss']:.4f}")

    if 'val_loss' in checkpoint:
        print(f"  验证损失: {checkpoint['val_loss']:.4f}")

    model = model.to(device)
    model.eval()

    return model

def load_three_views(feature_dir, scene_name, rsf_name):

    views = {}

    for view in ['left', 'middle', 'right']:
        filename = f"{scene_name}_{rsf_name}_{view}_env_feature.npy"
        filepath = os.path.join(feature_dir, scene_name, rsf_name, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"特征文件不存在: {filepath}")

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
    print(f"批量融合特征")
    print(f"{'='*70}")
    print(f"输入目录: {feature_dir}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")

    samples = create_sample_list(scenes_config, feature_dir)
    print(f"样本数量: {len(samples)}")

    success_count = 0
    failed_samples = []

    for scene_name, rsf_name, scene_type in tqdm(samples, desc="融合进度"):
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
            print(f"\n❌ 失败: {scene_name}/{rsf_name} - {e}")

    print(f"\n{'='*70}")
    print(f"✅ 融合完成")
    print(f"  成功: {success_count}/{len(samples)}")

    if failed_samples:
        print(f"  失败: {len(failed_samples)}")
        for scene, rsf, error in failed_samples:
            print(f"    - {scene}/{rsf}: {error}")

    print(f"{'='*70}\n")

    return success_count, failed_samples

def main():
    parser = argparse.ArgumentParser(description='批量融合三视角特征')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='训练好的模型检查点路径')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='输入特征目录（默认：train_feature和test_feature）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认：fused_features）')
    parser.add_argument('--dataset', type=str, default='both',
                       choices=['train', 'test', 'both'],
                       help='处理哪个数据集')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda or cpu)')
    parser.add_argument('--gpu_id', type=int, default=1,
                       help='GPU编号')
    parser.add_argument('--save_views', action='store_true',
                       help='同时保存原始三视角特征')

    args = parser.parse_args()

    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        device = torch.device('cuda')
        print(f"使用GPU: {args.gpu_id}")
    else:
        device = torch.device('cpu')
        print("使用CPU")

    model = load_trained_model(args.checkpoint, device)

    datasets_to_process = []

    if args.dataset in ['train', 'both']:
        input_dir = args.input_dir if args.input_dir else TRAIN_FEATURE_DIR
        output_dir = args.output_dir if args.output_dir else os.path.join(FUSION_OUTPUT_DIR, 'train')
        datasets_to_process.append(('训练集', input_dir, output_dir))

    if args.dataset in ['test', 'both']:
        input_dir = args.input_dir if args.input_dir else TEST_FEATURE_DIR
        output_dir = args.output_dir if args.output_dir else os.path.join(FUSION_OUTPUT_DIR, 'test')
        datasets_to_process.append(('测试集', input_dir, output_dir))

    total_success = 0
    total_failed = []

    for dataset_name, input_dir, output_dir in datasets_to_process:
        print(f"\n{'='*70}")
        print(f"处理 {dataset_name}")
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
    print(f"🎉 全部完成！")
    print(f"  总成功: {total_success}")
    print(f"  总失败: {len(total_failed)}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
