"""训练时的随机掩码，以及按有效帧约束的掩码。"""

import torch
import torch.nn as nn
import numpy as np

def patchify_manual_mask(manual_mask, t_patch_size, h_patch_size, w_patch_size):

    B, T = manual_mask.shape
    T_patches = T // t_patch_size

    mask_reshaped = manual_mask.reshape(B, T_patches, t_patch_size)

    mask_patch = mask_reshaped.max(dim=2)[0]

    return mask_patch

def random_masking(x, mask_ratio, manual_mask_patch=None):

    B, L, D = x.shape

    if manual_mask_patch is not None:

        if manual_mask_patch.shape[1] != L:
            L_time = manual_mask_patch.shape[1]
            L_spatial = L // L_time
            manual_mask_expanded = manual_mask_patch.unsqueeze(2).expand(
                B, L_time, L_spatial
            ).reshape(B, L)
        else:
            manual_mask_expanded = manual_mask_patch

        manual_mask_expanded = manual_mask_expanded.float().to(x.device)

        num_valid = (manual_mask_expanded == 0).sum(dim=1)

        len_keep_list = (num_valid * (1 - mask_ratio)).long()

        mask = manual_mask_expanded.clone()
        ids_keep_list = []
        ids_restore_list = []
        x_masked_list = []

        for b in range(B):
            valid_indices = torch.where(manual_mask_expanded[b] == 0)[0]
            num_v = len(valid_indices)
            len_k = len_keep_list[b].item()

            if num_v == 0 or len_k == 0:
                ids_keep = torch.tensor([], dtype=torch.long, device=x.device)
                ids_restore = torch.arange(L, device=x.device)
                mask[b, :] = 1.0
            else:
                perm = torch.randperm(num_v, device=x.device)
                keep_perm = perm[:len_k]
                remove_perm = perm[len_k:]

                ids_keep = valid_indices[keep_perm]
                ids_remove = valid_indices[remove_perm]

                mask[b, ids_remove] = 1.0

                ids_masked = torch.where(mask[b] == 1)[0]
                ids_restore = torch.cat([ids_keep, ids_masked])

                if len(ids_restore) < L:
                    missing = set(range(L)) - set(ids_restore.cpu().numpy())
                    missing_tensor = torch.tensor(list(missing), device=x.device)
                    ids_restore = torch.cat([ids_restore, missing_tensor])

            if len(ids_keep) > 0:
                x_masked_b = torch.index_select(x[b], dim=0, index=ids_keep)
            else:
                x_masked_b = torch.empty(0, D, device=x.device)

            ids_keep_list.append(ids_keep)
            ids_restore_list.append(ids_restore)
            x_masked_list.append(x_masked_b)

        max_keep = max(len(ids) for ids in ids_keep_list)
        if max_keep == 0:
            max_keep = 1

        ids_keep_padded = []
        x_masked_padded = []
        for b in range(B):
            if len(ids_keep_list[b]) < max_keep:
                pad_len = max_keep - len(ids_keep_list[b])
                ids_keep_pad = torch.cat([
                    ids_keep_list[b],
                    torch.zeros(pad_len, dtype=torch.long, device=x.device)
                ])
                x_masked_pad = torch.cat([
                    x_masked_list[b],
                    torch.zeros(pad_len, D, device=x.device)
                ])
            else:
                ids_keep_pad = ids_keep_list[b][:max_keep]
                x_masked_pad = x_masked_list[b][:max_keep]

            ids_keep_padded.append(ids_keep_pad)
            x_masked_padded.append(x_masked_pad)

        ids_keep = torch.stack(ids_keep_padded, dim=0)
        x_masked = torch.stack(x_masked_padded, dim=0)

        ids_restore = torch.stack(ids_restore_list, dim=0)

    else:
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore, ids_keep

def random_restore(x, ids_restore, N, T, H, W, C, mask_token):

    L = T * H * W

    mask_tokens = mask_token.repeat(N, L - x.shape[1], 1)

    x_ = torch.cat([x, mask_tokens], dim=1)

    x_ = torch.gather(
        x_, dim=1,
        index=ids_restore.unsqueeze(-1).repeat(1, 1, C)
    )

    return x_

def causal_masking(x, mask_ratio, T):

    N, L, D = x.shape
    x = x.reshape(N, T, L // T, D)
    N, T, L_spatial, C = x.shape

    len_keep = int(T * (1 - mask_ratio))

    noise = torch.arange(T).unsqueeze(dim=0).repeat(N, 1)
    noise = noise.to(x.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1,
        index=ids_keep.unsqueeze(2).unsqueeze(-1).repeat(1, 1, L_spatial, D)
    )

    mask = torch.ones([N, T, L_spatial], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(
        mask, dim=1,
        index=ids_restore.unsqueeze(2).repeat(1, 1, L_spatial)
    ).reshape(N, -1)

    ids_keep = ids_keep.unsqueeze(2).repeat(1, 1, L_spatial).reshape(N, -1)
    x_masked = x_masked.reshape(N, -1, x_masked.shape[-1])

    return x_masked, mask, ids_restore, ids_keep

def fre_masking(x, mask_ratio, T, H, W):

    N, L, D = x.shape
    x = x.reshape(N, T, H, W, D)

    len_keep = int(W * (1 - mask_ratio))

    noise = torch.arange(W).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N, T, H, 1)
    noise = noise.to(x.device)

    ids_shuffle = torch.argsort(noise, dim=3)
    ids_restore = torch.argsort(ids_shuffle, dim=3)

    ids_keep = ids_shuffle[:, :, :, :len_keep]
    x_masked = torch.gather(
        x, dim=3,
        index=ids_keep.unsqueeze(4).repeat(1, 1, 1, 1, D)
    )

    mask = torch.ones([N, T, H, W], device=x.device)
    mask[:, :, :, :len_keep] = 0
    mask = torch.gather(mask, dim=3, index=ids_restore).reshape(N, -1)

    ids_keep = ids_keep.reshape(N, -1)
    x_masked = x_masked.reshape(N, -1, x_masked.shape[-1])

    return x_masked, mask, ids_restore, ids_keep

if __name__ == '__main__':
    print("Testing masking.py...")

    print("\n1. Testing standard random masking...")
    B, L, D = 2, 3840, 512
    x = torch.randn(B, L, D)
    x_masked, mask, ids_restore, ids_keep = random_masking(x, mask_ratio=0.8)

    print(f"   Input: {x.shape}")
    print(f"   Masked: {x_masked.shape}")
    print(f"   Mask: {mask.shape}, masked ratio: {mask.float().mean():.2f}")
    print(f"   ids_restore: {ids_restore.shape}")
    print(f"   ids_keep: {ids_keep.shape}")

    print("\n2. Testing random masking with manual_mask...")
    T_patches = 15
    manual_mask = torch.zeros(B, T_patches)
    manual_mask[0, 5:8] = 1
    manual_mask[1, 10:13] = 1

    x_masked, mask, ids_restore, ids_keep = random_masking(
        x, mask_ratio=0.75, manual_mask_patch=manual_mask
    )

    print(f"   Manual mask: {manual_mask.shape}, missing: {manual_mask.sum(dim=1)}")
    print(f"   Masked: {x_masked.shape}")
    print(f"   Combined mask ratio: {mask.float().mean():.2f}")

    print("\n3. Testing random_restore...")
    N, T, H, W = 2, 15, 32, 8
    C = 512
    L = T * H * W
    x_kept = torch.randn(N, int(L * 0.2), C)
    mask_token = nn.Parameter(torch.zeros(1, 1, C))
    ids_restore = torch.randperm(L).unsqueeze(0).repeat(N, 1)

    x_restored = random_restore(x_kept, ids_restore, N, T, H, W, C, mask_token)
    print(f"   Kept tokens: {x_kept.shape}")
    print(f"   Restored: {x_restored.shape}")
    assert x_restored.shape == (N, L, C)

    print("\n4. Testing patchify_manual_mask...")
    B, T = 2, 60
    manual_mask_frame = torch.zeros(B, T)
    manual_mask_frame[0, 20:32] = 1
    manual_mask_frame[1, 40:56] = 1

    mask_patch = patchify_manual_mask(manual_mask_frame, t_patch_size=4, h_patch_size=4, w_patch_size=4)
    print(f"   Frame-level mask: {manual_mask_frame.shape}, missing: {manual_mask_frame.sum(dim=1)}")
    print(f"   Patch-level mask: {mask_patch.shape}, missing: {mask_patch.sum(dim=1)}")

    print("\n✅ All tests passed!")
