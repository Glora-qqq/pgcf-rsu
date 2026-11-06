"""Load DINOv2, Put the RGB Image Sequencing 1024 Dimension Features. """

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get('DINOV2_ROOT', os.path.join(_REPO, 'third_party', 'dinov2')))

try:
    from dinov2.models import vision_transformer as vits
    from dinov2.hub.backbones import dinov2_vitl14_reg, dinov2_vitl14, dinov2_vitb14
except ImportError as e:
    print(f"❌ Unable to importDINOv2: {e}")
    print("Please set DINOV2_ROOT, or clone the official repository to third_party/dinov2")
    raise

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class DINOv2FeatureExtractor:

    def __init__(
        self,
        weights_path=None,
        model_type='vitl14_reg4',
        device='cuda',
        batch_size=32
    ):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.model_type = model_type

        self.feature_dim = {
            'vitb14': 768,
            'vitl14': 1024,
            'vitl14_reg4': 1024,
            'vitg14': 1536,
        }[model_type]

        print(f"InitializationDINOv2Feature Extractor")
        print(f"  Model Type: {model_type}")
        print(f"  Feature Dimensions: {self.feature_dim}")
        print(f"  Equipment: {self.device}")
        print(f"  Batch size: {batch_size}")

        self.model = self._load_model(weights_path)
        self.model.eval()

        print(f"✅ DINOv2Model loaded successfully")

    def _load_model(self, weights_path):

        if self.model_type == 'vitl14_reg4':
            print(f"Loading weights: {weights_path}")
            model = dinov2_vitl14_reg(pretrained=False)

            if weights_path and os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)

        elif self.model_type == 'vitl14':
            print(f"Loading weights: {weights_path}")
            model = dinov2_vitl14(pretrained=False)

            if weights_path and os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)

        elif self.model_type == 'vitb14':
            print(f"Loading weights: {weights_path}")
            model = dinov2_vitb14(pretrained=False)

            if weights_path and os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        model = model.to(self.device)
        return model

    @torch.no_grad()
    def extract_frame(self, image):

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        feature = self.model(image)

        return feature.cpu()

    @torch.no_grad()
    def extract_sequence(self, dataset, show_progress=True):

        total_frames = len(dataset)
        features = []
        mask = dataset.mask.copy()

        batch_images = []
        batch_indices = []

        iterator = range(total_frames)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Extract features {dataset.scene}_{dataset.rsf}_{dataset.view}")

        for i in iterator:
            image, frame_id, is_missing = dataset[i]

            if is_missing or image is None:
                features.append(np.zeros(self.feature_dim, dtype=np.float32))
            else:
                batch_images.append(image)
                batch_indices.append(i)

                if len(batch_images) == self.batch_size or i == total_frames - 1:
                    batch_tensor = torch.stack(batch_images)
                    batch_features = self.extract_frame(batch_tensor)

                    for feat in batch_features:
                        features.append(feat.numpy())

                    batch_images = []
                    batch_indices = []

        if len(batch_images) > 0:
            batch_tensor = torch.stack(batch_images)
            batch_features = self.extract_frame(batch_tensor)
            for feat in batch_features:
                features.append(feat.numpy())

        features = np.array(features, dtype=np.float32)

        return features, mask

    def extract_and_save(self, dataset, output_path, show_progress=True):

        features, mask = self.extract_sequence(dataset, show_progress)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        np.save(output_path, {
            'features': features,
            'mask': mask,
            'scene': dataset.scene,
            'rsf': dataset.rsf,
            'view': dataset.view,
            'feature_dim': self.feature_dim,
            'model_type': self.model_type,
            'valid_frames': (~mask).sum(),
            'missing_frames': mask.sum()
        })

        info = {
            'output_path': output_path,
            'shape': features.shape,
            'valid_frames': (~mask).sum(),
            'missing_frames': mask.sum(),
            'file_size_mb': os.path.getsize(output_path) / 1024 / 1024
        }

        return info

def test_extractor():

    print("=" * 70)
    print("TestDINOv2Feature Extractor")
    print("=" * 70)

    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    from sequences import RGBSequenceDataset

    extractor = DINOv2FeatureExtractor(
        weights_path=DINOV2_DEFAULT_WEIGHTS,
        model_type=DINOV2_MODEL_TYPE,
        device=DEVICE,
        batch_size=BATCH_SIZE
    )

    samples = get_all_samples(IMAGE_DIR_TRAIN, MASK_DIR_TRAIN)
    if len(samples) == 0:
        print("❌ No samples found")
        return

    scene, rsf, view = samples[0]
    print(f"\nTest Sample: {scene}_{rsf}_{view}")

    dataset = RGBSequenceDataset(IMAGE_DIR_TRAIN, MASK_DIR_TRAIN, scene, rsf, view)

    print("\nStart extracting features...")
    features, mask = extractor.extract_sequence(dataset, show_progress=True)

    print(f"\nExtract Results:")
    print(f"  Featuresshape: {features.shape}")
    print(f"  Featuresdtype: {features.dtype}")
    print(f"  Effective Frames: {(~mask).sum()}")
    print(f"  Missing frames: {mask.sum()}")

    valid_feat = features[~mask]
    missing_feat = features[mask]

    print(f"\nEffective Frame Characteristics:")
    print(f"  Quantity: {len(valid_feat)}")
    print(f"  Mean: {valid_feat.mean():.4f}")
    print(f"  Standard deviation: {valid_feat.std():.4f}")
    print(f"  Scope: [{valid_feat.min():.4f}, {valid_feat.max():.4f}]")

    print(f"\nMissing frame characteristics:")
    print(f"  Quantity: {len(missing_feat)}")
    print(f"  Is all zero: {np.allclose(missing_feat, 0)}")

    output_path = os.path.join(ENV_FEATURE_DIR_TRAIN, scene, rsf, f"{scene}_{rsf}_{view}_features.npy")
    print(f"\nSave features to: {output_path}")

    info = extractor.extract_and_save(dataset, output_path, show_progress=False)
    print(f"✅ Saved successfully")
    print(f"  File Size: {info['file_size_mb']:.2f} MB")

    print(f"\nVerify loading...")
    loaded_data = np.load(output_path, allow_pickle=True).item()
    print(f"  Featuresshape: {loaded_data['features'].shape}")
    print(f"  Scenario: {loaded_data['scene']}")
    print(f"  RSF: {loaded_data['rsf']}")
    print(f"  Angle of View: {loaded_data['view']}")
    print(f"  Valid Frames: {loaded_data['valid_frames']}")

    print("\n✅ Test Complete! ")

if __name__ == '__main__':
    test_extractor()
