"""用训练好的 Transformer 补全缺失帧的视觉特征。"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_infer import *
from data.infer_set import ExtractSample, create_extract_samples
from models.completion import TransformerFeatureCompletion

class DINOv2FeatureExtractor:

    def __init__(self, weights_path: str, device: torch.device):

        self.device = device

        print(f"🔧 加载DINOv2模型: {DINOV2_MODEL_NAME}")

        dinov2_path = Path(os.environ.get('DINOV2_ROOT', Path(weights_path).parent.parent))
        if str(dinov2_path) not in sys.path:
            sys.path.insert(0, str(dinov2_path))

        try:
            import dinov2.models.vision_transformer as vits
        except ImportError:
            raise ImportError(
                f"无法导入dinov2模块，请确认路径: {dinov2_path}"
            )

        if DINOV2_MODEL_NAME == 'dinov2_vitl14_reg':
            self.model = vits.__dict__['vit_large'](
                patch_size=14,
                img_size=526,
                init_values=1.0,
                num_register_tokens=4,
                block_chunks=0
            )
        else:
            raise ValueError(f"不支持的模型: {DINOV2_MODEL_NAME}")

        if os.path.exists(weights_path):
            print(f"📦 加载权重: {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu')

            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=True)
        else:
            print(f"⚠️  权重文件不存在: {weights_path}")

        self.model = self.model.to(device)
        self.model.eval()

        print(f"✅ DINOv2模型加载完成")

    @torch.no_grad()
    def infer(self, images: torch.Tensor, batch_size: int = 32) -> torch.Tensor:

        n_images = images.shape[0]
        features_list = []

        for i in range(0, n_images, batch_size):
            batch = images[i:i+batch_size].to(self.device)

            batch_features = self.model.forward_features(batch)

            if isinstance(batch_features, dict):
                batch_features = batch_features['x_norm_clstoken']
            else:
                batch_features = batch_features[:, 0]

            features_list.append(batch_features.cpu())

        features = torch.cat(features_list, dim=0)

        return features

def load_transformer_model(model_path: str, device: torch.device) -> nn.Module:

    print(f"🔧 加载Transformer模型: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    config = checkpoint.get('config', {})
    print(f"📋 模型配置: {config}")

    model = TransformerFeatureCompletion(
        d_model=config.get('d_model', TRANSFORMER_D_MODEL),
        nhead=config.get('nhead', TRANSFORMER_NHEAD),
        num_layers=config.get('num_layers', TRANSFORMER_NUM_LAYERS),
        dim_feedforward=config.get('dim_feedforward', TRANSFORMER_DIM_FEEDFORWARD),
        dropout=config.get('dropout', TRANSFORMER_DROPOUT),
        max_seq_length=config.get('max_seq_length', MAX_SEQ_LENGTH)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✅ Transformer模型加载完成")
    print(f"   训练轮次: {checkpoint.get('epoch', 'N/A')}")
    print(f"   验证损失: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return model

def extract_sample_features(
    sample: ExtractSample,
    backbone: DINOv2FeatureExtractor,
    transformer_model: nn.Module,
    device: torch.device
) -> np.ndarray:

    images, valid_mask = sample.load_images()

    features = backbone.infer(images, batch_size=BATCH_SIZE_PER_GPU)

    features_tensor = features.unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(sample.mask).bool().unsqueeze(0).to(device)

    with torch.no_grad():
        completed_features = transformer_model(
            features_tensor,
            src_key_padding_mask=mask_tensor
        )

    completed_features = completed_features.squeeze(0).cpu().numpy()

    return completed_features

def save_features(features: np.ndarray, sample: ExtractSample, output_dir: str):

    scene_dir = os.path.join(output_dir, sample.scene)
    rsf_dir = os.path.join(scene_dir, sample.rsf)
    os.makedirs(rsf_dir, exist_ok=True)

    filename = f"{sample.scene}_{sample.rsf}_{sample.view}_env_feature.npy"
    filepath = os.path.join(rsf_dir, filename)

    np.save(filepath, features)

def infer_set(
    dataset_name: str,
    dataset_config: dict,
    backbone: DINOv2FeatureExtractor,
    transformer_model: nn.Module,
    device: torch.device
):

    print("\n" + "=" * 80)
    print(f"🚀 开始提取 {dataset_name.upper()} 数据集特征")
    print("=" * 80)

    samples = create_extract_samples(dataset_config, VIEWS)

    if not samples:
        print(f"⚠️  未找到任何样本，跳过")
        return

    print(f"📊 共 {len(samples)} 个样本待处理")

    output_dir = dataset_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")

    success_count = 0
    fail_count = 0

    for sample in tqdm(samples, desc=f"提取{dataset_name}特征", ncols=100):
        try:
            features = extract_sample_features(
                sample,
                backbone,
                transformer_model,
                device
            )

            save_features(features, sample, output_dir)

            success_count += 1

        except Exception as e:
            print(f"\n❌ 处理失败: {sample}, 错误: {e}")
            fail_count += 1
            continue

    print("\n" + "=" * 80)
    print(f"✅ {dataset_name.upper()} 数据集提取完成")
    print(f"   成功: {success_count}/{len(samples)}")
    print(f"   失败: {fail_count}/{len(samples)}")
    print("=" * 80)

def main():

    print("=" * 80)
    print("🎯 环境特征提取程序")
    print("=" * 80)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n📋 验证配置...")
    if not validate_config():
        print("❌ 配置验证失败，退出")
        return

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("❌ 没有可用的GPU")
        return

    device = torch.device("cuda:0")
    print(f"\n🎮 使用GPU: cuda:0 (可见GPU数: {available_gpus})")

    print("\n" + "=" * 80)
    print("🔧 加载模型")
    print("=" * 80)

    backbone = DINOv2FeatureExtractor(DINOV2_WEIGHTS_PATH, device)

    transformer_model = load_transformer_model(TRANSFORMER_MODEL_PATH, device)

    infer_set(
        'train',
        TRAIN_CONFIG,
        backbone,
        transformer_model,
        device
    )

    infer_set(
        'test',
        TEST_CONFIG,
        backbone,
        transformer_model,
        device
    )

    print("\n" + "=" * 80)
    print("🎉 所有特征提取完成！")
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
