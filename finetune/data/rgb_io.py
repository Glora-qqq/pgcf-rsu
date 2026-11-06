"""读取测试集视觉融合 npz，并补齐到模型长度。"""

import numpy as np
from pathlib import Path
from typing import Optional

def load_test_rgb_features(
    rgb_root: Path,
    scene: str,
    rsf: str,
    num_frames_original: int = 750,
    num_frames_padded: int = 1500,
    padding_strategy: str = 'mean',
    feature_key: str = 'fused',
) -> Optional[np.ndarray]:

    rgb_file = rgb_root / f'{scene}_{rsf}_features.npz'

    if not rgb_file.exists():
        print(f"[WARNING] RGB特征文件不存在: {rgb_file}")
        return None

    try:
        npz_data = np.load(rgb_file)

        if feature_key not in npz_data:
            print(f"[WARNING] Key '{feature_key}' 不存在，可用keys: {list(npz_data.keys())}")
            return None

        rgb_750 = npz_data[feature_key].astype(np.float32)

        if rgb_750.shape[0] != num_frames_original:
            print(f"[WARNING] RGB帧数不匹配: 期望{num_frames_original}, 实际{rgb_750.shape[0]}")
            if rgb_750.shape[0] > num_frames_original:
                rgb_750 = rgb_750[:num_frames_original]
            else:
                pad_len = num_frames_original - rgb_750.shape[0]
                rgb_750 = np.pad(
                    rgb_750,
                    ((0, pad_len), (0, 0)),
                    mode='edge'
                )

        if padding_strategy == 'mean':
            rgb_mean = rgb_750.mean(axis=0, keepdims=True)
            rgb_padding = np.repeat(rgb_mean, num_frames_padded - num_frames_original, axis=0)
        elif padding_strategy == 'repeat':
            last_frame = rgb_750[-1:, :]
            rgb_padding = np.repeat(last_frame, num_frames_padded - num_frames_original, axis=0)
        else:
            raise ValueError(f"未知的padding策略: {padding_strategy}")

        rgb_1500 = np.concatenate([rgb_750, rgb_padding], axis=0)

        return rgb_1500.astype(np.float32)

    except Exception as e:
        print(f"[ERROR] RGB特征加载失败 {rgb_file}: {e}")
        return None

def verify_test_rgb_loading():

    from config_test import TEST_RGB_FEATURE_ROOT

    print("=" * 60)
    print("验证测试RGB特征加载")
    print("=" * 60)

    test_scene = 'Scene1'
    test_rsf = 'RSF1'

    print("\n[1] 测试 padding_strategy='mean'")
    rgb_mean = load_test_rgb_features(
        rgb_root=TEST_RGB_FEATURE_ROOT,
        scene=test_scene,
        rsf=test_rsf,
        padding_strategy='mean',
    )

    if rgb_mean is not None:
        print(f"  ✓ RGB加载成功")
        print(f"    形状: {rgb_mean.shape}")
        print(f"    数据类型: {rgb_mean.dtype}")
        print(f"    前750帧范围: [{rgb_mean[:750].min():.4f}, {rgb_mean[:750].max():.4f}]")
        print(f"    后750帧范围: [{rgb_mean[750:].min():.4f}, {rgb_mean[750:].max():.4f}]")
        print(f"    后750帧是否全相同: {np.all(rgb_mean[750:] == rgb_mean[750, :])}")
    else:
        print("  ✗ RGB加载失败")

    print("\n[2] 测试 padding_strategy='repeat'")
    rgb_repeat = load_test_rgb_features(
        rgb_root=TEST_RGB_FEATURE_ROOT,
        scene=test_scene,
        rsf=test_rsf,
        padding_strategy='repeat',
    )

    if rgb_repeat is not None:
        print(f"  ✓ RGB加载成功")
        print(f"    形状: {rgb_repeat.shape}")
        print(f"    后750帧是否全等于最后一帧: {np.all(rgb_repeat[750:] == rgb_repeat[749, :])}")
    else:
        print("  ✗ RGB加载失败")

    print("=" * 60)

if __name__ == '__main__':
    verify_test_rgb_loading()
