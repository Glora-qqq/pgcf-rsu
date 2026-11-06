"""加载微调权重，按车辆和路侧单元导出 512 维特征。"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
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

class FeatureExtractor:

    def __init__(
        self,
        checkpoint_path: Path,
        config: Dict,
        device: str = 'cuda'
    ):

        self.checkpoint_path = checkpoint_path
        self.config = config
        self.device = device

        print(f"\n{'='*70}")
        print("加载模型...")
        print(f"{'='*70}")
        self.model = self._load_model()
        self.model.eval()
        print(f"✓ 模型加载成功！")

        self.output_root = FEATURES_OUTPUT_ROOT
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.extraction_stats = {
            'total_vehicles': 0,
            'success_count': 0,
            'failed_vehicles': [],
            'feature_shapes': {},
        }

    def _load_model(self) -> nn.Module:

        model = GatedWiFo(
            embed_dim=MODEL_CONFIG['embed_dim'],
            decoder_embed_dim=MODEL_CONFIG['decoder_embed_dim'],
            depth=MODEL_CONFIG['depth'],
            decoder_depth=MODEL_CONFIG['decoder_depth'],
            num_heads=MODEL_CONFIG['num_heads'],
            decoder_num_heads=MODEL_CONFIG['decoder_num_heads'],
            mlp_ratio=MODEL_CONFIG['mlp_ratio'],
            t_patch_size=MODEL_CONFIG['t_patch_size'],
            patch_size=MODEL_CONFIG['patch_size'],
            pos_emb='SinCos_3D',
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)

        return model

    def extract_scene_features(
        self,
        scene: str,
        rsf_list: List[str],
        is_test: bool = False
    ):

        data_type = "测试集" if is_test else "训练集"
        print(f"\n{'='*70}")
        print(f"提取 {data_type} {scene} 特征")
        print(f"{'='*70}")

        data_root = TEST_DATA_ROOT if is_test else TRAIN_DATA_ROOT
        mask_root = TEST_MASK_ROOT if is_test else TRAIN_MASK_ROOT
        rgb_root = TEST_RGB_ROOT if is_test else TRAIN_RGB_ROOT

        for rsf in rsf_list:
            print(f"\n[{scene}/{rsf}] 开始提取...")

            vehicles = self._get_vehicles_for_rsf(scene, rsf, data_root, is_test)

            if not vehicles:
                print(f"[WARNING] {scene}/{rsf} 无可用车辆")
                continue

            print(f"  找到 {len(vehicles)} 个车辆: {', '.join(vehicles)}")

            for vehicle in vehicles:
                try:
                    self.extract_vehicle_features(
                        scene=scene,
                        rsf=rsf,
                        vehicle=vehicle,
                        data_root=data_root,
                        mask_root=mask_root,
                        rgb_root=rgb_root,
                        is_test=is_test
                    )
                    self.extraction_stats['success_count'] += 1

                except Exception as e:
                    print(f"[ERROR] {scene}/{rsf}/{vehicle} 提取失败: {e}")
                    self.extraction_stats['failed_vehicles'].append(
                        f"{scene}/{rsf}/{vehicle}"
                    )

    def extract_vehicle_features(
        self,
        scene: str,
        rsf: str,
        vehicle: str,
        data_root: Path,
        mask_root: Path,
        rgb_root: Path,
        is_test: bool = False
    ):

        self.extraction_stats['total_vehicles'] += 1

        print(f"\n  [{vehicle}] 开始提取特征...")

        load_scene_for_mask = SCENE_NAME_MAPPING.get(scene, scene.replace('_', '')) if is_test else scene

        csi, mask, rgb_feature, meta = self._load_vehicle_data(
            load_scene_for_mask, rsf, vehicle, data_root, mask_root, rgb_root, is_test
        )

        gps_vehicle = VEHICLE_NAME_MAPPING.get(vehicle, vehicle)
        gps_info = get_vehicle_frame_info(
            scene=scene,
            vehicle=gps_vehicle,
            data_root=data_root,
            is_test=is_test
        )

        print(f"    GPS范围: 帧{gps_info['start_frame']}-{gps_info['end_frame']} (共{gps_info['total_frames']}帧)")

        window_features_list, window_start_indices = self._extract_window_features(csi, mask, rgb_feature)

        if len(window_features_list) == 0:
            print(f"    ⚠️  跳过: 没有高质量窗口")
            self.extraction_stats['failed_vehicles'].append(f"{scene}/{rsf}/{vehicle}")
            return

        print(f"    聚合 {len(window_features_list)} 个窗口特征...")
        window_size = WINDOW_CONFIG['window_size']

        features_full = aggregate_window_features_to_frames(
            window_features_list=window_features_list,
            window_start_indices=window_start_indices,
            window_size=window_size,
            total_frames=CSI_CONFIG['num_frames'],
            feature_dim=EXTRACT_CONFIG['feature_dim']
        )

        print(f"    完整特征: {features_full.shape}")

        if EXTRACT_CONFIG['use_gps_alignment']:
            features_vehicle = crop_features_by_gps(
                features_full=features_full,
                gps_info=gps_info,
                verify=EXTRACT_CONFIG['verify_alignment']
            )
            print(f"    GPS截取后: {features_vehicle.shape}")
        else:
            features_vehicle = features_full

        output_vehicle = VEHICLE_NAME_MAPPING.get(vehicle, vehicle)
        output_file = get_feature_file_path(scene, rsf, output_vehicle, is_test)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if EXTRACT_CONFIG['save_format'] == 'npy':
            np.save(output_file, features_vehicle)
        elif EXTRACT_CONFIG['save_format'] == 'npz':
            save_dict = {
                'features': features_vehicle,
                'gps_coordinates': gps_info['gps_coordinates'],
                'frame_ids': gps_info['frame_ids'],
                'scene': scene,
                'rsf': rsf,
                'vehicle': vehicle,
            }
            np.savez(output_file.with_suffix('.npz'), **save_dict)

        self.extraction_stats['feature_shapes'][f"{scene}/{rsf}/{vehicle}"] = features_vehicle.shape

        print(f"    ✓ 特征已保存: {output_file.name}")

    def _load_vehicle_data(
        self,
        scene: str,
        rsf: str,
        vehicle: str,
        data_root: Path,
        mask_root: Path,
        rgb_root: Path,
        is_test: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:

        csi, mask, rgb_feature, meta = load_vehicle_data_for_extraction(
            scene=scene,
            rsf=rsf,
            vehicle=vehicle,
            data_root=data_root,
            mask_root=mask_root,
            rgb_root=rgb_root,
            is_test=is_test,
            num_frames=CSI_CONFIG['num_frames'],
            rgb_feature_dim=MODEL_CONFIG['rgb_input_dim'],
        )

        return csi, mask, rgb_feature, meta

    def _extract_window_features(
        self,
        csi: torch.Tensor,
        mask: torch.Tensor,
        rgb_feature: torch.Tensor
    ) -> Tuple[List[np.ndarray], List[int]]:

        window_size = WINDOW_CONFIG['window_size']
        stride = WINDOW_CONFIG['stride']
        min_valid_frames = WINDOW_CONFIG['min_valid_frames']
        T = csi.shape[1]

        total_windows = (T - window_size) // stride + 1

        valid_windows = []
        for i in range(total_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size

            window_mask = mask[start_idx:end_idx]
            num_valid = int((window_mask == 0).sum())

            if num_valid >= min_valid_frames:
                valid_windows.append((start_idx, end_idx, num_valid))

        print(f"    总窗口: {total_windows}, 高质量窗口: {len(valid_windows)} (有效帧>={min_valid_frames})")

        if len(valid_windows) == 0:
            print(f"    ⚠️  警告: 没有高质量窗口！")
            return [], []

        window_features_list = []
        window_start_indices = []

        with torch.no_grad():
            for start_idx, end_idx, num_valid in tqdm(valid_windows, desc="    ", leave=False):
                csi_window = csi[:, start_idx:end_idx, :, :]

                csi_window = csi_window.unsqueeze(0).to(self.device)
                rgb_feature_batch = rgb_feature.unsqueeze(0).to(self.device)

                features = self._forward_infer(csi_window, rgb_feature_batch)

                features = features.squeeze(0).cpu().numpy()

                window_features_list.append(features)
                window_start_indices.append(start_idx)

        return window_features_list, window_start_indices

    def _forward_infer(
        self,
        csi: torch.Tensor,
        rgb_feature: torch.Tensor
    ) -> torch.Tensor:

        feature_source = self.config.get('feature_source', 'decoder_features')

        latent, mask, ids_restore, input_size = self.model.forward_encoder(
            csi, mask_ratio=0.0
        )

        B, L, D = latent.shape
        T = input_size[0] * self.model.t_patch_size
        H = input_size[1] * self.model.patch_size
        W = input_size[2] * self.model.patch_size

        if feature_source == 'encoder':
            patch_features = latent

        elif feature_source == 'decoder_features':
            decoder_features = self.model.forward_decoder(
                latent,
                ids_restore,
                input_size,
                rgb_feature=rgb_feature,
                return_features=True
            )
            patch_features = decoder_features[:, 1:, :]

        else:
            raise ValueError(f"Unknown feature_source: {feature_source}")

        frame_features = patch_to_frame_features(
            patch_features=patch_features,
            T=T,
            H=H,
            W=W,
            t_patch_size=self.model.t_patch_size,
            patch_size=self.model.patch_size
        )

        return frame_features

    def _get_vehicles_for_rsf(
        self,
        scene: str,
        rsf: str,
        data_root: Path,
        is_test: bool = False
    ) -> List[str]:

        csi_dir = data_root / scene / rsf / 'CSI'

        if not csi_dir.exists():
            return []

        csi_suffix = SCENE_CSI_SUFFIX.get(scene, 'high_fencha')

        vehicles = set()
        for mat_file in csi_dir.glob(f'{rsf}_*_{csi_suffix}_CSI.mat'):
            filename = mat_file.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                vehicle_csi = parts[1]
                vehicle = VEHICLE_NAME_MAPPING.get(vehicle_csi, vehicle_csi)
                if vehicle.startswith(('Car', 'AT', 'UAV')):
                    vehicles.add(vehicle)

        return sorted(list(vehicles))

    def save_extraction_summary(self):

        summary_file = FEATURES_OUTPUT_ROOT / 'extraction_summary.json'

        summary = {
            'total_vehicles': self.extraction_stats['total_vehicles'],
            'success_count': self.extraction_stats['success_count'],
            'failed_count': len(self.extraction_stats['failed_vehicles']),
            'failed_vehicles': self.extraction_stats['failed_vehicles'],
            'config': {
                'checkpoint': str(self.checkpoint_path),
                'feature_dim': EXTRACT_CONFIG['feature_dim'],
                'window_size': WINDOW_CONFIG['window_size'],
                'stride': WINDOW_CONFIG['stride'],
                'gps_alignment': EXTRACT_CONFIG['use_gps_alignment'],
            }
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 提取摘要已保存: {summary_file}")

def main():

    print("=" * 70)
    print("PGCF-RSU 特征导出")
    print("=" * 70)

    print_config_summary()

    print("\n验证配置...")
    if not verify_config():
        print("\n⚠️  配置验证失败，请检查路径配置！")
        return

    try:
        extractor = FeatureExtractor(
            checkpoint_path=CHECKPOINT_PATH,
            config=EXTRACT_CONFIG,
            device=EXTRACT_CONFIG['device']
        )
    except Exception as e:
        print(f"\n[ERROR] 初始化失败: {e}")
        return

    print("\n" + "="*70)
    print("开始提取训练集特征")
    print(f"min_valid_frames: {WINDOW_CONFIG['min_valid_frames']} 帧")
    print("="*70)

    for scene in TRAIN_SCENES:
        rsf_list = TRAIN_RSF_CONFIG.get(scene, [])
        extractor.extract_scene_features(
            scene=scene,
            rsf_list=rsf_list,
            is_test=False
        )

    print("\n" + "="*70)
    print("开始提取测试集特征")
    print(f"min_valid_frames: {WINDOW_CONFIG['min_valid_frames']} 帧")
    print("="*70)

    for scene in TEST_SCENES:
        rsf_list = TEST_RSF_CONFIG.get(scene, [])
        extractor.extract_scene_features(
            scene=scene,
            rsf_list=rsf_list,
            is_test=True
        )

    extractor.save_extraction_summary()

    print("\n" + "="*70)
    print("特征提取完成！")
    print("="*70)
    print(f"\n统计信息：")
    print(f"  总车辆数: {extractor.extraction_stats['total_vehicles']}")
    print(f"  成功: {extractor.extraction_stats['success_count']}")
    print(f"  失败: {len(extractor.extraction_stats['failed_vehicles'])}")

    if extractor.extraction_stats['failed_vehicles']:
        print(f"\n失败列表：")
        for v in extractor.extraction_stats['failed_vehicles']:
            print(f"  - {v}")

if __name__ == '__main__':
    main()
