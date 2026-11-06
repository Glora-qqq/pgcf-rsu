"""提取 DINOv2 特征并训练缺失帧补全模型。"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data.sequences import RGBSequenceDataset
from models.backbone import DINOv2FeatureExtractor
from models.completion import create_model, count_parameters

class TransformerTrainDataset(Dataset):

    def __init__(self, image_dir, mask_dir, backbone, feature_cache_dir=None, use_cached=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.backbone = backbone
        self.feature_cache_dir = feature_cache_dir
        self.use_cached = use_cached and feature_cache_dir is not None

        self.samples = []

        if os.path.exists(mask_dir):
            for scene in sorted(os.listdir(mask_dir)):
                scene_dir = os.path.join(mask_dir, scene)
                if not os.path.isdir(scene_dir):
                    continue

                for filename in sorted(os.listdir(scene_dir)):
                    if not filename.endswith('_RGBmask.npy'):
                        continue

                    name_without_suffix = filename.replace('_RGBmask.npy', '')
                    parts = name_without_suffix.split('_')

                    if len(parts) >= 2:
                        scene_from_file = parts[0]
                        rsf = parts[1]

                        if scene_from_file == scene:
                            for view in VIEWS:
                                self.samples.append((scene, rsf, view))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        scene, rsf, view = self.samples[idx]

        if self.use_cached:
            cache_path = os.path.join(
                self.feature_cache_dir, scene, rsf, f"{scene}_{rsf}_{view}_features.npy"
            )
            if os.path.exists(cache_path):
                data = np.load(cache_path, allow_pickle=True).item()
                features = torch.from_numpy(data['features']).float()
                mask = torch.from_numpy(data['mask']).bool()
                return features, mask, (scene, rsf, view)

        rgb_dataset = RGBSequenceDataset(
            self.image_dir, self.mask_dir, scene, rsf, view
        )

        features, mask = self.backbone.extract_sequence(
            rgb_dataset,
            show_progress=False
        )

        features = torch.from_numpy(features).float()
        mask = torch.from_numpy(mask).bool()

        return features, mask, (scene, rsf, view)

class DynamicFeatureDataset(Dataset):

    def __init__(self, image_dir, mask_dir, backbone):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.backbone = backbone

        self.samples = get_all_samples(image_dir, mask_dir)
        print(f"找到 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        scene, rsf, view = self.samples[idx]

        rgb_dataset = RGBSequenceDataset(
            self.image_dir, self.mask_dir, scene, rsf, view
        )

        features, mask = self.backbone.extract_sequence(
            rgb_dataset,
            show_progress=False
        )

        features = torch.from_numpy(features).float()
        mask = torch.from_numpy(mask).bool()

        return features, mask, (scene, rsf, view)

def create_interpolated_target(features, mask):

    batch_size, seq_len, feat_dim = features.shape
    target = features.clone()

    for b in range(batch_size):
        seq_mask = mask[b]
        seq_feat = features[b]

        if seq_mask.sum() == 0:
            continue

        valid_indices = (~seq_mask).nonzero(as_tuple=True)[0].cpu().numpy()

        if len(valid_indices) < 2:
            continue

        for d in range(feat_dim):
            valid_values = seq_feat[valid_indices, d].cpu().numpy()

            interp_values = np.interp(
                np.arange(seq_len),
                valid_indices,
                valid_values
            )

            target[b, :, d] = torch.from_numpy(interp_values).to(features.device)

    return target

def compute_loss(output, target, mask, loss_type='mse', mask_weight=1.0, valid_weight=0.1):

    if loss_type == 'mse':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.SmoothL1Loss(reduction='none')

    element_loss = criterion(output, target)

    frame_loss = element_loss.mean(dim=-1)

    mask_loss = (frame_loss * mask.float()).sum() / (mask.sum() + 1e-8)

    valid_loss = (frame_loss * (~mask).float()).sum() / ((~mask).sum() + 1e-8)

    total_loss = mask_weight * mask_loss + valid_weight * valid_loss

    return total_loss, {
        'total': total_loss.item(),
        'mask': mask_loss.item(),
        'valid': valid_loss.item()
    }

def train_epoch(model, dataloader, optimizer, device):

    import time
    model.train()
    total_loss = 0
    total_mask_loss = 0
    total_valid_loss = 0

    pbar = tqdm(dataloader, desc="🔥 训练中", unit="batch", ncols=120, leave=False)

    batch_start = time.time()
    for batch_idx, (features, mask, _) in enumerate(pbar):
        data_time = time.time() - batch_start
        features = features.to(device)
        mask = mask.to(device)

        target = create_interpolated_target(features, mask)

        optimizer.zero_grad()
        output = model(features, src_key_padding_mask=mask)

        loss, loss_dict = compute_loss(
            output, target, mask,
            loss_type=LOSS_TYPE,
            mask_weight=MASK_LOSS_WEIGHT,
            valid_weight=VALID_LOSS_WEIGHT
        )

        loss.backward()
        optimizer.step()

        total_loss += loss_dict['total']
        total_mask_loss += loss_dict['mask']
        total_valid_loss += loss_dict['valid']

        train_time = time.time() - batch_start - data_time

        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'mask_l': f"{loss_dict['mask']:.4f}",
            'valid_l': f"{loss_dict['valid']:.4f}",
            'data': f"{data_time:.2f}s",
            'train': f"{train_time:.2f}s"
        })

        batch_start = time.time()

    num_batches = len(dataloader)
    return total_loss / num_batches, total_mask_loss / num_batches, total_valid_loss / num_batches

@torch.no_grad()
def validate_epoch(model, dataloader, device):

    model.eval()
    total_loss = 0
    total_mask_loss = 0
    total_valid_loss = 0

    pbar = tqdm(dataloader, desc="✅ 验证中", unit="batch", ncols=120, leave=False)
    for features, mask, _ in pbar:
        features = features.to(device)
        mask = mask.to(device)

        target = create_interpolated_target(features, mask)
        output = model(features, src_key_padding_mask=mask)

        loss, loss_dict = compute_loss(
            output, target, mask,
            loss_type=LOSS_TYPE,
            mask_weight=MASK_LOSS_WEIGHT,
            valid_weight=VALID_LOSS_WEIGHT
        )

        total_loss += loss_dict['total']
        total_mask_loss += loss_dict['mask']
        total_valid_loss += loss_dict['valid']

    num_batches = len(dataloader)
    return total_loss / num_batches, total_mask_loss / num_batches, total_valid_loss / num_batches

def stage1_infer(image_dir, mask_dir, output_dir, gpu_ids=None):

    print("\n" + "="*80)
    print("🔥 阶段1: DINOv2特征提取（多GPU并行）")
    print("="*80)

    if gpu_ids is None:
        gpu_ids = GPU_IDS

    print(f"📌 使用GPU: {gpu_ids}")
    print(f"📌 批处理大小: {BATCH_SIZE_PER_GPU} 帧/GPU")

    samples = []

    if os.path.exists(mask_dir):
        for scene in sorted(os.listdir(mask_dir)):
            scene_dir = os.path.join(mask_dir, scene)
            if not os.path.isdir(scene_dir):
                continue

            for filename in sorted(os.listdir(scene_dir)):
                if not filename.endswith('_RGBmask.npy'):
                    continue

                name_without_suffix = filename.replace('_RGBmask.npy', '')
                parts = name_without_suffix.split('_')

                if len(parts) >= 2:
                    scene_from_file = parts[0]
                    rsf = parts[1]

                    if scene_from_file == scene:
                        for view in VIEWS:
                            samples.append((scene, rsf, view))

    print(f"📊 找到 {len(samples)} 个样本")

    scenes_count = {}
    for scene, rsf, view in samples:
        if scene not in scenes_count:
            scenes_count[scene] = {'rsfs': set(), 'views': set()}
        scenes_count[scene]['rsfs'].add(rsf)
        scenes_count[scene]['views'].add(view)

    print(f"📋 样本分布:")
    for scene in sorted(scenes_count.keys()):
        rsfs = len(scenes_count[scene]['rsfs'])
        views = len(scenes_count[scene]['views'])
        total = rsfs * views
        print(f"  {scene}: {rsfs} RSFs × {views} views = {total} samples")

    completed = 0
    pending_samples = []
    for scene, rsf, view in samples:
        output_path = os.path.join(output_dir, scene, rsf, f"{scene}_{rsf}_{view}_features.npy")
        if os.path.exists(output_path):
            completed += 1
        else:
            pending_samples.append((scene, rsf, view))

    if completed > 0:
        print(f"✅ 已完成: {completed}/{len(samples)} 样本")

    if len(pending_samples) == 0:
        print(f"🎉 所有特征已提取完成！")
        return

    print(f"⏳ 待处理: {len(pending_samples)} 个样本")
    print(f"📦 输出目录: {output_dir}")

    if len(gpu_ids) > 1 and ENABLE_MULTI_GPU:
        print(f"\n⚡ 启用多GPU并行加速 ({len(gpu_ids)} GPUs)")
        infer_multi_gpu(pending_samples, image_dir, mask_dir, output_dir, gpu_ids)
    else:
        print(f"\n🔧 单GPU模式")
        infer_single_gpu(pending_samples, image_dir, mask_dir, output_dir, gpu_ids[0])

    print(f"\n✅ 阶段1完成！特征已保存到: {output_dir}")

def infer_single_gpu(samples, image_dir, mask_dir, output_dir, gpu_id):

    import time
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue

    print(f"[GPU{gpu_id}] 🚀 进程启动")
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[GPU{gpu_id}] 📦 加载DINOv2模型...")
    start_time = time.time()

    from models.backbone import DINOv2FeatureExtractor
    extractor = DINOv2FeatureExtractor(
        model_type=DINOV2_MODEL_TYPE,
        device=device,
        batch_size=BATCH_SIZE_PER_GPU,
        weights_path=os.path.join(DINOV2_WEIGHTS_DIR, f'{DINOV2_MODEL_TYPE}_pretrain.pth')
    )

    load_time = time.time() - start_time
    print(f"[GPU{gpu_id}] ✅ 模型加载完成 ({load_time:.1f}s)")
    print(f"[GPU{gpu_id}] 🔥 开始提取 {len(samples)} 个样本")
    print(f"[GPU{gpu_id}] 💡 预加载优化: 6线程并行，3样本预加载队列")

    def load_dataset(scene, rsf, view):

        return RGBSequenceDataset(image_dir, mask_dir, scene, rsf, view)

    PREFETCH_SIZE = 3
    NUM_WORKERS = 6

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        prefetch_queue = Queue(maxsize=PREFETCH_SIZE)

        for i in range(min(PREFETCH_SIZE, len(samples))):
            future = executor.submit(load_dataset, *samples[i])
            prefetch_queue.put((i, future))

        next_to_load = PREFETCH_SIZE

        for idx, (scene, rsf, view) in enumerate(samples, 1):
            sample_start = time.time()
            output_path = os.path.join(output_dir, scene, rsf, f"{scene}_{rsf}_{view}_features.npy")

            print(f"[GPU{gpu_id}] [{idx}/{len(samples)}] 📂 {scene}_{rsf}_{view} ", end='', flush=True)
            load_start = time.time()

            _, current_future = prefetch_queue.get()
            rgb_dataset = current_future.result()
            load_time = time.time() - load_start
            print(f"加载:{load_time:.2f}s ", end='', flush=True)

            if next_to_load < len(samples):
                future = executor.submit(load_dataset, *samples[next_to_load])
                prefetch_queue.put((next_to_load, future))
                next_to_load += 1

            extract_start = time.time()
            features, mask = extractor.extract_sequence(rgb_dataset, show_progress=False)
            extract_time = time.time() - extract_start
            print(f"提取:{extract_time:.1f}s ", end='', flush=True)

            save_start = time.time()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, {
                'features': features,
                'mask': mask,
                'scene': scene,
                'rsf': rsf,
                'view': view
            })
            save_time = time.time() - save_start

            sample_time = time.time() - sample_start

            gpu_util = (extract_time / sample_time) * 100
            print(f"保存:{save_time:.1f}s | 总:{sample_time:.1f}s | GPU利用:{gpu_util:.0f}% ✅")

    print(f"[GPU{gpu_id}] 🎉 全部完成！")

def infer_multi_gpu(samples, image_dir, mask_dir, output_dir, gpu_ids):

    import torch.multiprocessing as mp

    num_gpus = len(gpu_ids)
    samples_per_gpu = [[] for _ in range(num_gpus)]
    for idx, sample in enumerate(samples):
        gpu_idx = idx % num_gpus
        samples_per_gpu[gpu_idx].append(sample)

    print(f"📊 样本分配:")
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        print(f"  GPU {gpu_id}: {len(samples_per_gpu[gpu_idx])} 样本")

    print(f"\n⚙️ 启动 {num_gpus} 个GPU进程...")
    print(f"⏳ 每个进程需要加载DINOv2模型（约10-15秒），请耐心等待...\n")

    mp.set_start_method('spawn', force=True)
    processes = []

    for gpu_idx, gpu_id in enumerate(gpu_ids):
        if len(samples_per_gpu[gpu_idx]) == 0:
            continue

        p = mp.Process(
            target=infer_single_gpu,
            args=(samples_per_gpu[gpu_idx], image_dir, mask_dir, output_dir, gpu_id)
        )
        p.start()
        print(f"✓ GPU{gpu_id} 进程已启动")
        processes.append(p)

    print(f"\n{'='*80}")
    print(f"💡 提示: 初次启动时，每个GPU需要独立加载模型，会有延迟")
    print(f"{'='*80}\n")

    for p in processes:
        p.join()

def stage2_train_transformer(feature_cache_dir, gpu_ids=None):

    print("\n" + "="*80)
    print("🚀 阶段2: Transformer训练")
    print("="*80)

    if gpu_ids is None:
        gpu_ids = GPU_IDS

    device = torch.device(f"cuda:{gpu_ids[0]}")

    use_multi_gpu = len(gpu_ids) > 1

    if use_multi_gpu:
        print(f"📌 使用多GPU: {gpu_ids} (DataParallel模式)")
        print(f"📌 主GPU: {gpu_ids[0]}")
    else:
        print(f"📌 使用单GPU: {gpu_ids[0]}")

    print(f"📌 批处理大小: {TRANSFORMER_BATCH_SIZE} 序列/batch")

    print("\n📦 加载预提取特征...")
    dataset = TransformerTrainDataset(
        IMAGE_DIR_TRAIN,
        MASK_DIR_TRAIN,
        None,
        feature_cache_dir=feature_cache_dir,
        use_cached=True
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"📊 数据集统计:")
    print(f"  总样本: {len(dataset)}")
    print(f"  训练集: {train_size} 样本")
    print(f"  验证集: {val_size} 样本")

    train_loader = DataLoader(
        train_subset,
        batch_size=TRANSFORMER_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=TRANSFORMER_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("\n🧠 创建Transformer模型...")
    model = create_model()
    model = model.to(device)

    if use_multi_gpu:
        print(f"🔗 启用DataParallel: {len(gpu_ids)}个GPU并行训练")
        model = nn.DataParallel(model, device_ids=gpu_ids)

    total_params, trainable_params, param_size = count_parameters(
        model.module if use_multi_gpu else model
    )
    print(f"📊 模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练: {trainable_params:,}")
    print(f"  大小: {param_size:.2f} MB")
    if use_multi_gpu:
        print(f"  每GPU显存: ~{param_size / len(gpu_ids):.1f} MB (模型) + 动态batch数据")

    optimizer = optim.Adam(
        model.parameters(),
        lr=TRANSFORMER_LEARNING_RATE
    )

    import time
    print("\n" + "=" * 80)
    print("🎯 开始训练")
    print("=" * 80)
    print(f"📊 训练集: {train_size} | 验证集: {val_size} | Batch: {TRANSFORMER_BATCH_SIZE}")
    print(f"💾 模型: {total_params:,}参数 ({param_size:.1f}MB)")
    print(f"🏋️ Epochs: {TRANSFORMER_EPOCHS} | 学习率: {TRANSFORMER_LEARNING_RATE}")
    print("=" * 80)

    best_val_loss = float('inf')
    os.makedirs(TRANSFORMER_CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, TRANSFORMER_EPOCHS + 1):
        epoch_start = time.time()

        print(f"\n{'='*80}")
        print(f"📍 Epoch {epoch}/{TRANSFORMER_EPOCHS}")
        print(f"{'='*80}")

        train_start = time.time()
        train_loss, train_mask_loss, train_valid_loss = train_epoch(
            model, train_loader, optimizer, device
        )
        train_time = time.time() - train_start

        val_start = time.time()
        val_loss, val_mask_loss, val_valid_loss = validate_epoch(
            model, val_loader, device
        )
        val_time = time.time() - val_start

        epoch_time = time.time() - epoch_start

        gpu_mem = ""
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            gpu_mem = f" | 🎮 GPU: {mem_allocated:.2f}GB/{mem_reserved:.2f}GB"

        print(f"\n📈 训练: Loss={train_loss:.4f} (mask={train_mask_loss:.4f}, valid={train_valid_loss:.4f}) | ⏱️ {train_time:.1f}s")
        print(f"📉 验证: Loss={val_loss:.4f} (mask={val_mask_loss:.4f}, valid={val_valid_loss:.4f}) | ⏱️ {val_time:.1f}s")
        print(f"⚡ Epoch总耗时: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}{gpu_mem}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            model_state = model.module.state_dict() if use_multi_gpu else model.state_dict()

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'd_model': TRANSFORMER_D_MODEL,
                    'nhead': TRANSFORMER_NHEAD,
                    'num_layers': TRANSFORMER_NUM_LAYERS,
                    'dim_feedforward': TRANSFORMER_DIM_FEEDFORWARD,
                    'dropout': TRANSFORMER_DROPOUT,
                    'max_seq_length': MAX_SEQ_LENGTH,
                }
            }
            torch.save(checkpoint, TRANSFORMER_BEST_MODEL)
            print(f"💾 保存最佳模型 → {TRANSFORMER_BEST_MODEL} (val_loss: {val_loss:.4f} ⬇️)")

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                TRANSFORMER_CHECKPOINT_DIR,
                f'checkpoint_epoch{epoch}.pth'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 保存checkpoint: epoch{epoch}")

    print("\n" + "=" * 80)
    print(f"✅ 阶段2完成！最佳验证损失: {best_val_loss:.4f}")
    print(f"💾 模型已保存: {TRANSFORMER_BEST_MODEL}")
    print("=" * 80)

def train_mode(backbone):

    print("\n" + "=" * 80)
    print("🚀 两阶段训练流程")
    print("=" * 80)
    print("📌 阶段1: DINOv2特征提取（多GPU并行）")
    print("📌 阶段2: Transformer训练（单GPU）")
    print("=" * 80)

    feature_cache_dir = os.path.join(BASE_DIR, 'feature_cache')

    stage1_infer(
        IMAGE_DIR_TRAIN,
        MASK_DIR_TRAIN,
        feature_cache_dir,
        gpu_ids=GPU_IDS
    )

    stage2_train_transformer(
        feature_cache_dir,
        gpu_ids=GPU_IDS
    )

    print("\n" + "=" * 80)
    print("🎉 两阶段训练全部完成！")
    print("=" * 80)

def extract_mode(backbone, dataset='train'):

    print("=" * 70)
    print("模式: 端到端提取环境特征")
    print("=" * 70)

    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    if dataset == 'train':
        image_dir = IMAGE_DIR_TRAIN
        mask_dir = MASK_DIR_TRAIN
        output_dir = ENV_FEATURE_DIR_TRAIN
    else:
        image_dir = IMAGE_DIR_TEST
        mask_dir = MASK_DIR_TEST
        output_dir = ENV_FEATURE_DIR_TEST

    print(f"\n加载Transformer模型: {TRANSFORMER_BEST_MODEL}")

    if not os.path.exists(TRANSFORMER_BEST_MODEL):
        print("⚠️  Transformer模型不存在，将只提取DINOv2特征（不补全）")
        transformer = None
    else:
        checkpoint = torch.load(TRANSFORMER_BEST_MODEL, map_location=device)
        config = checkpoint.get('config', {})

        transformer = create_model(**config)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        transformer = transformer.to(device)
        transformer.eval()

        print(f"✅ 模型加载成功 (epoch {checkpoint.get('epoch', 'N/A')})")

    print("\n获取样本列表...")
    samples = get_all_samples(image_dir, mask_dir)
    print(f"找到 {len(samples)} 个样本")

    print("\n开始提取...")
    print("=" * 70)

    start_time = time.time()
    success_count = 0
    failed_samples = []
    total_missing = 0

    for scene, rsf, view in tqdm(samples, desc="提取进度"):
        try:
            rgb_dataset = RGBSequenceDataset(image_dir, mask_dir, scene, rsf, view)

            rsf_dir = os.path.join(output_dir, scene, rsf)
            os.makedirs(rsf_dir, exist_ok=True)

            output_filename = f"{scene}_{rsf}_{view}_features.npy"
            output_path = os.path.join(rsf_dir, output_filename)

            if os.path.exists(output_path):
                success_count += 1
                continue

            dinov2_features = backbone.extract_sequence(
                rgb_dataset,
                show_progress=False
            )

            mask = rgb_dataset.mask

            if transformer is not None:
                features_tensor = torch.from_numpy(dinov2_features).float().unsqueeze(0)
                mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0)

                features_tensor = features_tensor.to(device)
                mask_tensor = mask_tensor.to(device)

                with torch.no_grad():
                    completed_tensor = transformer(
                        features_tensor,
                        src_key_padding_mask=mask_tensor
                    )

                final_features = completed_tensor.cpu().numpy()[0]
                completed = True
            else:
                final_features = dinov2_features
                completed = False

            np.save(output_path, {
                'features': final_features,
                'mask': mask,
                'scene': scene,
                'rsf': rsf,
                'view': view,
                'total_frames': TOTAL_FRAMES,
                'missing_frames': int(mask.sum()),
                'valid_frames': int((~mask).sum()),
                'completed': completed
            })

            success_count += 1
            total_missing += mask.sum()

        except Exception as e:
            print(f"\n❌ {scene}_{rsf}_{view}: {e}")
            failed_samples.append(f"{scene}_{rsf}_{view}")

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("提取完成!")
    print("=" * 70)
    print(f"\n统计:")
    print(f"  成功: {success_count}/{len(samples)}")
    print(f"  失败: {len(failed_samples)}/{len(samples)}")
    print(f"  总缺失帧: {total_missing}")
    if success_count > 0:
        print(f"  平均缺失: {total_missing/success_count:.1f} 帧/样本")
    print(f"  用时: {elapsed_time/60:.1f} 分钟")
    print(f"\n输出目录: {output_dir}")
    print("=" * 70)

def verify_mode(dataset='train'):

    if dataset == 'train':
        output_dir = ENV_FEATURE_DIR_TRAIN
    else:
        output_dir = ENV_FEATURE_DIR_TEST

    print("=" * 70)
    print("验证输出结构")
    print("=" * 70)
    print(f"\n输出目录: {output_dir}\n")

    if not os.path.exists(output_dir):
        print("❌ 输出目录不存在")
        return

    total_files = 0
    total_size = 0

    for scene in SCENES:
        scene_dir = os.path.join(output_dir, scene)
        if not os.path.exists(scene_dir):
            continue

        print(f"{scene}/")

        rsf_list = SCENE_RSF_MAP.get(scene, [])
        for rsf in rsf_list:
            rsf_dir = os.path.join(scene_dir, rsf)
            if not os.path.exists(rsf_dir):
                continue

            print(f"  {rsf}/")

            for view in VIEWS:
                filename = f"{scene}_{rsf}_{view}_features.npy"
                filepath = os.path.join(rsf_dir, filename)

                if os.path.exists(filepath):
                    size = os.path.getsize(filepath) / 1024 / 1024

                    try:
                        data = np.load(filepath, allow_pickle=True).item()
                        features = data['features']
                        completed = data.get('completed', False)

                        norms = np.linalg.norm(features, axis=1)
                        zero_frames = (norms < 1e-6).sum()

                        status = "补全" if completed else "DINOv2"
                        if zero_frames > 0:
                            print(f"    ⚠️  {filename} ({size:.2f} MB) - {status} - 有{zero_frames}零帧")
                        else:
                            print(f"    ✅ {filename} ({size:.2f} MB) - {status}")

                        total_files += 1
                        total_size += size
                    except:
                        print(f"    ❌ {filename} - 加载失败")
                else:
                    print(f"    ❌ {filename} - 不存在")

    print(f"\n总计: {total_files} 文件, {total_size:.2f} MB")
    print("=" * 70)

def main():

    import argparse

    parser = argparse.ArgumentParser(description='环境特征提取统一脚本')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'extract', 'verify'],
                       help='运行模式: train=训练Transformer, extract=提取特征, verify=验证输出')
    parser.add_argument('--dataset', type=str, default='train',
                       choices=['train', 'test'],
                       help='数据集: train或test')
    parser.add_argument('--device', type=str, default=None,
                       help='设备（默认使用config.py中的DEVICE）')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批处理大小（默认使用config.py）')

    args = parser.parse_args()

    device = args.device if args.device else DEVICE
    if not torch.cuda.is_available():
        device = 'cpu'

    batch_size = args.batch_size if args.batch_size else BATCH_SIZE

    if args.mode == 'verify':
        verify_mode(args.dataset)
        return

    print("\n加载DINOv2模型（预训练，用于特征提取）...")
    backbone = DINOv2FeatureExtractor(
        weights_path=DINOV2_DEFAULT_WEIGHTS,
        device=device,
        batch_size=batch_size
    )
    print("✅ DINOv2加载完成\n")

    if args.mode == 'train':
        train_mode(backbone)
    elif args.mode == 'extract':
        extract_mode(backbone, args.dataset)

if __name__ == '__main__':
    main()
