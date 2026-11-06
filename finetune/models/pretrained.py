"""Put the WiFo-RGB Pre-trained weight load encoder. """

import torch
import os
from pathlib import Path

def pretrained_weights(
    model,
    pretrained_path,
    freeze_encoder=True,
    freeze_decoder_layers=3,
    strict=False,
    verbose=True
):

    if verbose:
        print(f"\n{'='*60}")
        print(f"Loading pretrained weights from:")
        print(f"  {pretrained_path}")
        print(f"{'='*60}")

    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained weight file not found: {pretrained_path}")

    checkpoint = torch.load(pretrained_path, map_location='cpu')

    if 'model' in checkpoint:
        pretrained_dict = checkpoint['model']
        if verbose:
            print(f"✅ Loaded checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
    else:
        pretrained_dict = checkpoint
        if verbose:
            print(f"✅ Loaded state dict")

    model_dict = model.state_dict()

    load_info = {
        'total_keys': len(pretrained_dict),
        'matched_keys': 0,
        'skipped_keys': 0,
        'shape_mismatch_keys': 0,
        'missing_keys': 0,
        'loaded_keys': [],
        'skipped_keys_list': [],
        'shape_mismatch_list': [],
        'missing_keys_list': [],
    }

    skip_prefixes = ['res18.', 'res18_align_layer.']

    filtered_dict = {}

    for k, v in pretrained_dict.items():
        if any(k.startswith(prefix) for prefix in skip_prefixes):
            load_info['skipped_keys'] += 1
            load_info['skipped_keys_list'].append(k)
            continue

        if k not in model_dict:
            load_info['missing_keys'] += 1
            load_info['missing_keys_list'].append(k)
            continue

        if v.shape != model_dict[k].shape:
            load_info['shape_mismatch_keys'] += 1
            load_info['shape_mismatch_list'].append(
                f"{k}: pretrained {v.shape} vs model {model_dict[k].shape}"
            )
            continue

        filtered_dict[k] = v
        load_info['matched_keys'] += 1
        load_info['loaded_keys'].append(k)

    model.load_state_dict(filtered_dict, strict=False)

    if verbose:
        print(f"\n📊 Loading Statistics:")
        print(f"   Total keys in pretrained: {load_info['total_keys']}")
        print(f"   ✅ Matched and loaded: {load_info['matched_keys']}")
        print(f"   ⏭️  Skipped (ResNet18): {load_info['skipped_keys']}")
        print(f"   ⚠️  Shape mismatch: {load_info['shape_mismatch_keys']}")
        print(f"   ❌ Missing in model: {load_info['missing_keys']}")

        if load_info['skipped_keys_list'] and verbose:
            print(f"\n⏭️  Skipped keys (first 5):")
            for k in load_info['skipped_keys_list'][:5]:
                print(f"     {k}")

        if load_info['shape_mismatch_list'] and verbose:
            print(f"\n⚠️  Shape mismatch:")
            for item in load_info['shape_mismatch_list']:
                print(f"     {item}")

        if load_info['missing_keys_list'] and verbose:
            print(f"\n❌ Missing keys (not in pretrained, first 5):")
            for k in load_info['missing_keys_list'][:5]:
                print(f"     {k}")

    if freeze_encoder or freeze_decoder_layers > 0:
        if verbose:
            print(f"\n🔒 Freezing Parameters:")

        if freeze_encoder:
            freeze_modules = [
                model.embedding,
                model.blocks,
                model.norm,
            ]
            for module in freeze_modules:
                for param in module.parameters():
                    param.requires_grad = False
            if verbose:
                print(f"   ✅ Encoder frozen (embedding + {len(model.blocks)} blocks + norm)")

        if freeze_decoder_layers > 0:
            for i in range(min(freeze_decoder_layers, len(model.decoder_blocks))):
                for param in model.decoder_blocks[i].parameters():
                    param.requires_grad = False
            if verbose:
                print(f"   ✅ Decoder first {freeze_decoder_layers} blocks frozen")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"\n📈 Parameters:")
            print(f"   Total: {total_params:,}")
            print(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    if verbose:
        print(f"\n{'='*60}")
        print(f"✅ Pretrained weights loaded successfully!")
        print(f"{'='*60}\n")

    return model, load_info

def freeze_module(module):

    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module(module):

    for param in module.parameters():
        param.requires_grad = True

def print_trainable_parameters(model, detailed=False):

    print(f"\n{'='*60}")
    print(f"Trainable Parameters")
    print(f"{'='*60}")

    trainable_params = 0
    frozen_params = 0

    if detailed:
        print(f"\n{'Module':<40} {'Trainable':<12} {'Frozen':<12}")
        print(f"{'-'*64}")

    for name, module in model.named_children():
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)

        trainable_params += module_trainable
        frozen_params += module_frozen

        if detailed:
            print(f"{name:<40} {module_trainable:>10,} {module_frozen:>10,}")

    total_params = trainable_params + frozen_params

    print(f"\n{'='*60}")
    print(f"Total:        {total_params:>10,}")
    print(f"Trainable:    {trainable_params:>10,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen:       {frozen_params:>10,} ({100*frozen_params/total_params:.1f}%)")
    print(f"{'='*60}\n")

def get_pretrained_path(model_size='base', pretrained_dir=None):

    if pretrained_dir is None:
        pretrained_dir = Path(__file__).parent.parent.parent / 'pretrained'
    else:
        pretrained_dir = Path(pretrained_dir)

    filename_map = {
        'base': 'WiFo-RGB-base.pkl',
        'small': 'WiFo-RGB-small.pkl',
        'little': 'WiFo-RGB-little.pkl',
        'tiny': 'WiFo-RGB-tiny.pkl',
    }

    if model_size not in filename_map:
        raise ValueError(f"Unknown model size: {model_size}")

    weight_path = pretrained_dir / filename_map[model_size]

    return str(weight_path)
