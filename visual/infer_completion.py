"""Trained with Transformer Completion of visual features of missing frames. """

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

        print(f"🔧 LoadDINOv2Model: {DINOV2_MODEL_NAME}")

        dinov2_path = Path(os.environ.get('DINOV2_ROOT', Path(weights_path).parent.parent))
        if str(dinov2_path) not in sys.path:
            sys.path.insert(0, str(dinov2_path))

        try:
            import dinov2.models.vision_transformer as vits
        except ImportError:
            raise ImportError(
                f"Unable to importdinov2Module, Please confirm the path: {dinov2_path}"
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
            raise ValueError(f"Unsupported model: {DINOV2_MODEL_NAME}")

        if os.path.exists(weights_path):
            print(f"📦 Loading weights: {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu')

            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=True)
        else:
            print(f"⚠️  Weight file does not exist: {weights_path}")

        self.model = self.model.to(device)
        self.model.eval()

        print(f"✅ DINOv2Model loading complete")

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

    print(f"🔧 LoadTransformerModel: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    config = checkpoint.get('config', {})
    print(f"📋 Model Configuration: {config}")

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

    print(f"✅ TransformerModel loading complete")
    print(f"   Training rounds: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Validate loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

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
    print(f"🚀 Start fetching {dataset_name.upper()} Dataset characteristics")
    print("=" * 80)

    samples = create_extract_samples(dataset_config, VIEWS)

    if not samples:
        print(f"⚠️  No samples found, Skip")
        return

    print(f"📊 of {len(samples)} samples pending")

    output_dir = dataset_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    success_count = 0
    fail_count = 0

    for sample in tqdm(samples, desc=f"Extract{dataset_name}Features", ncols=100):
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
            print(f"\n❌ Processing failed: {sample}, Error: {e}")
            fail_count += 1
            continue

    print("\n" + "=" * 80)
    print(f"✅ {dataset_name.upper()} Dataset extraction complete")
    print(f"   Success: {success_count}/{len(samples)}")
    print(f"   Failed: {fail_count}/{len(samples)}")
    print("=" * 80)

def main():

    print("=" * 80)
    print("🎯 Environmental Characteristics Extraction Procedure")
    print("=" * 80)
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n📋 Validate Configuration...")
    if not validate_config():
        print("❌ Configuration validation failed, Exit")
        return

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("❌ Not availableGPU")
        return

    device = torch.device("cuda:0")
    print(f"\n🎮 UseGPU: cuda:0 (VisibleGPUNumber: {available_gpus})")

    print("\n" + "=" * 80)
    print("🔧 Load Model")
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
    print("🎉 All feature extraction completed! ")
    print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
