"""按时间顺序读取 RGB 帧，供补全模型训练。"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class RGBSequenceDataset(Dataset):

    def __init__(self, image_dir, mask_dir, scene, rsf, view, transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.scene = scene
        self.rsf = rsf
        self.view = view

        mask_path = get_mask_path(mask_dir, scene, rsf)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask文件不存在: {mask_path}")

        self.mask = np.load(mask_path)

        assert len(self.mask) == TOTAL_FRAMES,\
            f"Mask长度错误: {len(self.mask)} != {TOTAL_FRAMES}"

        self.frame_paths = []
        for frame_id in range(1, TOTAL_FRAMES + 1):
            img_path = get_image_path(image_dir, scene, rsf, frame_id, view)
            self.frame_paths.append(img_path)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.transform = transform

        self.num_missing = self.mask.sum()
        self.num_valid = TOTAL_FRAMES - self.num_missing

    def __len__(self):

        return TOTAL_FRAMES

    def __getitem__(self, idx):

        frame_id = idx + 1
        is_missing = self.mask[idx]

        if is_missing:
            return None, frame_id, True
        else:
            img_path = self.frame_paths[idx]

            if not os.path.exists(img_path):
                print(f"⚠️ 警告: 图像文件不存在 {img_path}")
                return None, frame_id, True

            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                return image, frame_id, False

            except Exception as e:
                print(f"⚠️ 警告: 图像加载失败 {img_path}: {e}")
                return None, frame_id, True

    def get_sample_info(self):

        missing_frames = np.where(self.mask)[0] + 1

        missing_ranges = []
        if len(missing_frames) > 0:
            diff = np.diff(missing_frames)
            breaks = np.where(diff > 1)[0]

            start_idx = 0
            for break_idx in breaks:
                missing_ranges.append((
                    int(missing_frames[start_idx]),
                    int(missing_frames[break_idx])
                ))
                start_idx = break_idx + 1
            missing_ranges.append((
                int(missing_frames[start_idx]),
                int(missing_frames[-1])
            ))

        return {
            'scene': self.scene,
            'rsf': self.rsf,
            'view': self.view,
            'total_frames': TOTAL_FRAMES,
            'valid_frames': int(self.num_valid),
            'missing_frames': int(self.num_missing),
            'missing_rate': float(self.num_missing / TOTAL_FRAMES),
            'missing_ranges': missing_ranges
        }

    def __repr__(self):
        info = self.get_sample_info()
        return (f"RGBSequenceDataset({info['scene']}_{info['rsf']}_{info['view']}, "
                f"{info['valid_frames']}/{info['total_frames']} frames, "
                f"missing={info['missing_rate']:.1%})")

def create_sample_info(image_dir, mask_dir):

    samples = get_all_samples(image_dir, mask_dir)

    sample_infos = []
    for scene, rsf, view in samples:
        try:
            dataset = RGBSequenceDataset(image_dir, mask_dir, scene, rsf, view)
            info = dataset.get_sample_info()
            sample_infos.append(info)
        except Exception as e:
            print(f"⚠️ 警告: 无法创建样本 {scene}_{rsf}_{view}: {e}")

    return sample_infos

def test_dataset():

    print("=" * 70)
    print("测试RGBSequenceDataset")
    print("=" * 70)

    samples = get_all_samples(IMAGE_DIR_TRAIN, MASK_DIR_TRAIN)

    if len(samples) == 0:
        print("❌ 错误: 没有找到任何样本")
        print(f"请检查mask目录: {MASK_DIR_TRAIN}")
        return

    print(f"\n找到 {len(samples)} 个样本")
    print(f"第一个样本: {samples[0]}")

    scene, rsf, view = samples[0]
    dataset = RGBSequenceDataset(IMAGE_DIR_TRAIN, MASK_DIR_TRAIN, scene, rsf, view)

    print(f"\n{dataset}")

    info = dataset.get_sample_info()
    print("\n样本详细信息:")
    print(f"  场景: {info['scene']}")
    print(f"  RSF: {info['rsf']}")
    print(f"  视角: {info['view']}")
    print(f"  总帧数: {info['total_frames']}")
    print(f"  有效帧数: {info['valid_frames']}")
    print(f"  缺失帧数: {info['missing_frames']}")
    print(f"  缺失率: {info['missing_rate']:.2%}")
    print(f"  缺失区间: {info['missing_ranges']}")

    print("\n测试读取前10帧:")
    for i in range(10):
        image, frame_id, is_missing = dataset[i]

        if is_missing:
            print(f"  帧 {frame_id}: ❌ 缺失")
        else:
            print(f"  帧 {frame_id}: ✅ 有效, shape={image.shape}")

    if len(info['missing_ranges']) > 0:
        first_missing_start, first_missing_end = info['missing_ranges'][0]
        print(f"\n测试第一个缺失区间 [{first_missing_start}, {first_missing_end}]:")

        image, frame_id, is_missing = dataset[first_missing_start - 2]
        print(f"  帧 {frame_id} (缺失前): {'❌ 缺失' if is_missing else '✅ 有效'}")

        image, frame_id, is_missing = dataset[first_missing_start - 1]
        print(f"  帧 {frame_id} (缺失开始): {'❌ 缺失' if is_missing else '✅ 有效'}")

        image, frame_id, is_missing = dataset[first_missing_end - 1]
        print(f"  帧 {frame_id} (缺失结束): {'❌ 缺失' if is_missing else '✅ 有效'}")

        if first_missing_end < TOTAL_FRAMES:
            image, frame_id, is_missing = dataset[first_missing_end]
            print(f"  帧 {frame_id} (缺失后): {'❌ 缺失' if is_missing else '✅ 有效'}")

    print("\n" + "=" * 70)
    print("统计所有样本")
    print("=" * 70)

    all_infos = create_sample_info(IMAGE_DIR_TRAIN, MASK_DIR_TRAIN)

    total_frames = sum(info['total_frames'] for info in all_infos)
    total_valid = sum(info['valid_frames'] for info in all_infos)
    total_missing = sum(info['missing_frames'] for info in all_infos)

    print(f"\n样本数: {len(all_infos)}")
    print(f"总帧数: {total_frames}")
    print(f"有效帧数: {total_valid}")
    print(f"缺失帧数: {total_missing}")
    print(f"总缺失率: {total_missing/total_frames:.2%}")

    print("\n按场景统计:")
    for scene in SCENES:
        scene_infos = [info for info in all_infos if info['scene'] == scene]
        if len(scene_infos) > 0:
            scene_total = sum(info['total_frames'] for info in scene_infos)
            scene_missing = sum(info['missing_frames'] for info in scene_infos)
            print(f"  {scene}: {len(scene_infos)} 样本, "
                  f"{scene_missing}/{scene_total} 缺失 "
                  f"({scene_missing/scene_total:.1%})")

    print("\n✅ 数据集测试完成！")

if __name__ == '__main__':
    test_dataset()
