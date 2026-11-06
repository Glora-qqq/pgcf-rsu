"""Extract DINOv2 Feature and Train Missing Frame Completion Model. """

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
        print(f"Found {len(self.samples)} samples")

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

    pbar = tqdm(dataloader, desc="🔥 In Training", unit="batch", ncols=120, leave=False)

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

    pbar = tqdm(dataloader, desc="✅ Verifying", unit="batch", ncols=120, leave=False)
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
    print("🔥 Phase1: DINOv2Feature extraction (MultipleGPUParallel) ")
    print("="*80)

    if gpu_ids is None:
        gpu_ids = GPU_IDS

    print(f"📌 UseGPU: {gpu_ids}")
    print(f"📌 Batch size: {BATCH_SIZE_PER_GPU} Frame/GPU")

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

    print(f"📊 Found {len(samples)} samples")

    scenes_count = {}
    for scene, rsf, view in samples:
        if scene not in scenes_count:
            scenes_count[scene] = {'rsfs': set(), 'views': set()}
        scenes_count[scene]['rsfs'].add(rsf)
        scenes_count[scene]['views'].add(view)

    print(f"📋 Sample Distribution:")
    for scene in sorted(scenes_count.keys()):
        rsfs = len(scenes_count[scene]['rsfs'])
        views = len(scenes_count[scene]['views'])
        total = rsfs * views
        print(f"  {scene}: {rsfs} RSFs x {views} views = {total} samples")

    completed = 0
    pending_samples = []
    for scene, rsf, view in samples:
        output_path = os.path.join(output_dir, scene, rsf, f"{scene}_{rsf}_{view}_features.npy")
        if os.path.exists(output_path):
            completed += 1
        else:
            pending_samples.append((scene, rsf, view))

    if completed > 0:
        print(f"✅ Completed: {completed}/{len(samples)} Sample")

    if len(pending_samples) == 0:
        print(f"🎉 All features have been extracted! ")
        return

    print(f"⏳ Pending: {len(pending_samples)} samples")
    print(f"📦 Output directory: {output_dir}")

    if len(gpu_ids) > 1 and ENABLE_MULTI_GPU:
        print(f"\n⚡ Enable MultipleGPUParallel acceleration ({len(gpu_ids)} GPUs)")
        infer_multi_gpu(pending_samples, image_dir, mask_dir, output_dir, gpu_ids)
    else:
        print(f"\n🔧 SingleGPUMode")
        infer_single_gpu(pending_samples, image_dir, mask_dir, output_dir, gpu_ids[0])

    print(f"\n✅ Phase1Done! Feature saved to: {output_dir}")

def infer_single_gpu(samples, image_dir, mask_dir, output_dir, gpu_id):

    import time
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue

    print(f"[GPU{gpu_id}] 🚀 Process startup")
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[GPU{gpu_id}] 📦 LoadDINOv2Model...")
    start_time = time.time()

    from models.backbone import DINOv2FeatureExtractor
    extractor = DINOv2FeatureExtractor(
        model_type=DINOV2_MODEL_TYPE,
        device=device,
        batch_size=BATCH_SIZE_PER_GPU,
        weights_path=os.path.join(DINOV2_WEIGHTS_DIR, f'{DINOV2_MODEL_TYPE}_pretrain.pth')
    )

    load_time = time.time() - start_time
    print(f"[GPU{gpu_id}] ✅ Model loading complete ({load_time:.1f}s)")
    print(f"[GPU{gpu_id}] 🔥 Start fetching {len(samples)} samples")
    print(f"[GPU{gpu_id}] 💡 Preload optimizations: 6Thread Parallelism, 3Sample Preload Queue")

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
            print(f"Load:{load_time:.2f}s ", end='', flush=True)

            if next_to_load < len(samples):
                future = executor.submit(load_dataset, *samples[next_to_load])
                prefetch_queue.put((next_to_load, future))
                next_to_load += 1

            extract_start = time.time()
            features, mask = extractor.extract_sequence(rgb_dataset, show_progress=False)
            extract_time = time.time() - extract_start
            print(f"Extract:{extract_time:.1f}s ", end='', flush=True)

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
            print(f"Save:{save_time:.1f}s | Total:{sample_time:.1f}s | GPUUse:{gpu_util:.0f}% ✅")

    print(f"[GPU{gpu_id}] 🎉 All done! ")

def infer_multi_gpu(samples, image_dir, mask_dir, output_dir, gpu_ids):

    import torch.multiprocessing as mp

    num_gpus = len(gpu_ids)
    samples_per_gpu = [[] for _ in range(num_gpus)]
    for idx, sample in enumerate(samples):
        gpu_idx = idx % num_gpus
        samples_per_gpu[gpu_idx].append(sample)

    print(f"📊 Sample Allocation:")
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        print(f"  GPU {gpu_id}: {len(samples_per_gpu[gpu_idx])} Sample")

    print(f"\n⚙️ Start {num_gpus} PieceGPUProcess...")
    print(f"⏳ Each process needs to be loadedDINOv2Model (Approx.10-15Seconds) , Please be patient...\n")

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
        print(f"✓ GPU{gpu_id} Process started")
        processes.append(p)

    print(f"\n{'='*80}")
    print(f"💡 Tip: At initial startup, EachGPUModel needs to be loaded independently, There will be delays")
    print(f"{'='*80}\n")

    for p in processes:
        p.join()

def stage2_train_transformer(feature_cache_dir, gpu_ids=None):

    print("\n" + "="*80)
    print("🚀 Phase2: TransformerTraining")
    print("="*80)

    if gpu_ids is None:
        gpu_ids = GPU_IDS

    device = torch.device(f"cuda:{gpu_ids[0]}")

    use_multi_gpu = len(gpu_ids) > 1

    if use_multi_gpu:
        print(f"📌 Used a lotGPU: {gpu_ids} (DataParallelMode)")
        print(f"📌 MainGPU: {gpu_ids[0]}")
    else:
        print(f"📌 Usage FormGPU: {gpu_ids[0]}")

    print(f"📌 Batch size: {TRANSFORMER_BATCH_SIZE} Sequence/batch")

    print("\n📦 Load Prefetch Features...")
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

    print(f"📊 Dataset Statistics:")
    print(f"  Total Samples: {len(dataset)}")
    print(f"  Training Sets: {train_size} Sample")
    print(f"  Validation Set: {val_size} Sample")

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

    print("\n🧠 CreateTransformerModel...")
    model = create_model()
    model = model.to(device)

    if use_multi_gpu:
        print(f"🔗 EnableDataParallel: {len(gpu_ids)}PieceGPUParallel training")
        model = nn.DataParallel(model, device_ids=gpu_ids)

    total_params, trainable_params, param_size = count_parameters(
        model.module if use_multi_gpu else model
    )
    print(f"📊 Model Parameters:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: {param_size:.2f} MB")
    if use_multi_gpu:
        print(f"  EveryGPUGraphics Memory: ~{param_size / len(gpu_ids):.1f} MB (Model) + DynamicbatchData")

    optimizer = optim.Adam(
        model.parameters(),
        lr=TRANSFORMER_LEARNING_RATE
    )

    import time
    print("\n" + "=" * 80)
    print("🎯 Start training")
    print("=" * 80)
    print(f"📊 Training Sets: {train_size} | Validation Set: {val_size} | Batch: {TRANSFORMER_BATCH_SIZE}")
    print(f"💾 Model: {total_params:,}Parameter ({param_size:.1f}MB)")
    print(f"🏋️ Epochs: {TRANSFORMER_EPOCHS} | Learning Rate: {TRANSFORMER_LEARNING_RATE}")
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

        print(f"\n📈 Training: Loss={train_loss:.4f} (mask={train_mask_loss:.4f}, valid={train_valid_loss:.4f}) | ⏱️ {train_time:.1f}s")
        print(f"📉 Verify: Loss={val_loss:.4f} (mask={val_mask_loss:.4f}, valid={val_valid_loss:.4f}) | ⏱️ {val_time:.1f}s")
        print(f"⚡ EpochTotal time spent: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}{gpu_mem}")

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
            print(f"💾 Save the best model → {TRANSFORMER_BEST_MODEL} (val_loss: {val_loss:.4f} ⬇️)")

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                TRANSFORMER_CHECKPOINT_DIR,
                f'checkpoint_epoch{epoch}.pth'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Savecheckpoint: epoch{epoch}")

    print("\n" + "=" * 80)
    print(f"✅ Phase2Done! Best Validation Loss: {best_val_loss:.4f}")
    print(f"💾 Model saved: {TRANSFORMER_BEST_MODEL}")
    print("=" * 80)

def train_mode(backbone):

    print("\n" + "=" * 80)
    print("🚀 Two-stage training process")
    print("=" * 80)
    print("📌 Phase1: DINOv2Feature extraction (MultipleGPUParallel) ")
    print("📌 Phase2: TransformerTraining (SingleGPU) ")
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
    print("🎉 Two-stage training completed! ")
    print("=" * 80)

def extract_mode(backbone, dataset='train'):

    print("=" * 70)
    print("Mode: End-to-end extraction of environmental characteristics")
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

    print(f"\nLoadTransformerModel: {TRANSFORMER_BEST_MODEL}")

    if not os.path.exists(TRANSFORMER_BEST_MODEL):
        print("⚠️  TransformerModel does not exist, will only fetchDINOv2Features (Incomplete) ")
        transformer = None
    else:
        checkpoint = torch.load(TRANSFORMER_BEST_MODEL, map_location=device)
        config = checkpoint.get('config', {})

        transformer = create_model(**config)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        transformer = transformer.to(device)
        transformer.eval()

        print(f"✅ Model loaded successfully (epoch {checkpoint.get('epoch', 'N/A')})")

    print("\nGet Sample List...")
    samples = get_all_samples(image_dir, mask_dir)
    print(f"Found {len(samples)} samples")

    print("\nStart fetching...")
    print("=" * 70)

    start_time = time.time()
    success_count = 0
    failed_samples = []
    total_missing = 0

    for scene, rsf, view in tqdm(samples, desc="Fetch progress"):
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
    print("Extraction complete!")
    print("=" * 70)
    print(f"\nStatistics:")
    print(f"  Success: {success_count}/{len(samples)}")
    print(f"  Failed: {len(failed_samples)}/{len(samples)}")
    print(f"  Total Missing Frames: {total_missing}")
    if success_count > 0:
        print(f"  Average Missing: {total_missing/success_count:.1f} Frame/Sample")
    print(f"  Elapsed time: {elapsed_time/60:.1f} Minutes")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)

def verify_mode(dataset='train'):

    if dataset == 'train':
        output_dir = ENV_FEATURE_DIR_TRAIN
    else:
        output_dir = ENV_FEATURE_DIR_TEST

    print("=" * 70)
    print("Verify output structure")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}\n")

    if not os.path.exists(output_dir):
        print("❌ Output directory does not exist")
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

                        status = "Completion" if completed else "DINOv2"
                        if zero_frames > 0:
                            print(f"    ⚠️  {filename} ({size:.2f} MB) - {status} - Yes{zero_frames}Zero frames")
                        else:
                            print(f"    ✅ {filename} ({size:.2f} MB) - {status}")

                        total_files += 1
                        total_size += size
                    except:
                        print(f"    ❌ {filename} - Loading failed")
                else:
                    print(f"    ❌ {filename} - Absent")

    print(f"\nTotal: {total_files} File, {total_size:.2f} MB")
    print("=" * 70)

def main():

    import argparse

    parser = argparse.ArgumentParser(description='Unified script for environmental feature extraction')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'extract', 'verify'],
                       help='Operating mode: train=TrainingTransformer, extract=Extract features, verify=Validate Output')
    parser.add_argument('--dataset', type=str, default='train',
                       choices=['train', 'test'],
                       help='Dataset: trainortest')
    parser.add_argument('--device', type=str, default=None,
                       help='Equipment (Use by defaultconfig.pyinDEVICE) ')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (Use by defaultconfig.py) ')

    args = parser.parse_args()

    device = args.device if args.device else DEVICE
    if not torch.cuda.is_available():
        device = 'cpu'

    batch_size = args.batch_size if args.batch_size else BATCH_SIZE

    if args.mode == 'verify':
        verify_mode(args.dataset)
        return

    print("\nLoadDINOv2Model (Pre-training, Used for feature extraction) ...")
    backbone = DINOv2FeatureExtractor(
        weights_path=DINOV2_DEFAULT_WEIGHTS,
        device=device,
        batch_size=batch_size
    )
    print("✅ DINOv2Loading complete\n")

    if args.mode == 'train':
        train_mode(backbone)
    elif args.mode == 'extract':
        extract_mode(backbone, args.dataset)

if __name__ == '__main__':
    main()
