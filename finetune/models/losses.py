"""只在有效 patch 上计算 CSI 重建损失。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMSELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, wifo_mask, manual_mask=None):

        loss = torch.abs(pred - target) ** 2
        loss = loss.mean(dim=-1)

        wifo_mask = wifo_mask.view(loss.shape)

        if manual_mask is not None:
            manual_mask = manual_mask.view(loss.shape)
            valid_mask = (manual_mask == 0).float()
        else:
            valid_mask = torch.ones_like(loss)

        train_mask = wifo_mask * valid_mask
        num_masked = train_mask.sum()

        if num_masked > 0:
            loss_masked = (loss * train_mask).sum() / num_masked
        else:
            loss_masked = torch.tensor(0.0, device=loss.device)

        visible_mask = (1 - wifo_mask) * valid_mask
        num_visible = visible_mask.sum()

        if num_visible > 0:
            loss_visible = (loss * visible_mask).sum() / num_visible
        else:
            loss_visible = torch.tensor(0.0, device=loss.device)

        loss_dict = {
            'loss_masked': loss_masked.item(),
            'loss_visible': loss_visible.item(),
            'num_masked': num_masked.item(),
            'num_visible': num_visible.item(),
            'mask_ratio': (num_masked / (num_masked + num_visible)).item() if (num_masked + num_visible) > 0 else 0.0,
        }

        return loss_masked, loss_visible, loss_dict

def calculate_masked_loss(pred, target, wifo_mask, manual_mask=None, loss_weights=(1.0, 0.0)):

    criterion = MaskedMSELoss()
    loss_masked, loss_visible, loss_dict = criterion(pred, target, wifo_mask, manual_mask)

    w_masked, w_visible = loss_weights
    total_loss = w_masked * loss_masked + w_visible * loss_visible

    loss_dict['total_loss'] = total_loss.item()
    loss_dict['w_masked'] = w_masked
    loss_dict['w_visible'] = w_visible

    return total_loss, loss_dict

def patchify_csi(csi, t_patch_size, patch_size):

    if csi.dtype == torch.complex64 or csi.dtype == torch.complex128:
        B, T, H, W = csi.shape

        assert T % t_patch_size == 0
        assert H % patch_size == 0
        assert W % patch_size == 0

        T_patches = T // t_patch_size
        H_patches = H // patch_size
        W_patches = W // patch_size

        patches = csi.reshape(
            B, T_patches, t_patch_size, H_patches, patch_size, W_patches, patch_size
        )
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6)
        patches = patches.reshape(B, -1, t_patch_size * patch_size * patch_size)

    else:
        B, C, T, H, W = csi.shape
        assert C == 2

        assert T % t_patch_size == 0
        assert H % patch_size == 0
        assert W % patch_size == 0

        T_patches = T // t_patch_size
        H_patches = H // patch_size
        W_patches = W // patch_size

        patches = csi.reshape(
            B, C, T_patches, t_patch_size, H_patches, patch_size, W_patches, patch_size
        )
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7)
        patches = patches.reshape(B, -1, C * t_patch_size * patch_size * patch_size)

    return patches

def unpatchify_csi(patches, t_patch_size, patch_size, T, H, W, output_complex=True):

    B = patches.shape[0]
    T_patches = T // t_patch_size
    H_patches = H // patch_size
    W_patches = W // patch_size

    if output_complex:
        patches = patches.reshape(
            B, T_patches, H_patches, W_patches, t_patch_size, patch_size, patch_size
        )
        patches = patches.permute(0, 1, 4, 2, 5, 3, 6)
        csi = patches.reshape(B, T, H, W)
    else:
        patch_dim = patches.shape[-1]
        C = 2
        assert patch_dim == C * t_patch_size * patch_size * patch_size

        patches = patches.reshape(
            B, T_patches, H_patches, W_patches, C, t_patch_size, patch_size, patch_size
        )
        patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7)
        csi = patches.reshape(B, C, T, H, W)

    return csi

if __name__ == '__main__':
    print("Testing losses.py...")

    print("\n1. Testing MaskedMSELoss (no manual_mask)...")
    B, L, patch_dim = 2, 3840, 64
    pred = torch.randn(B, L, patch_dim)
    target = torch.randn(B, L, patch_dim)
    wifo_mask = torch.zeros(B, L)
    wifo_mask[:, :int(L * 0.8)] = 1

    criterion = MaskedMSELoss()
    loss_masked, loss_visible, loss_dict = criterion(pred, target, wifo_mask)

    print(f"   Loss masked: {loss_masked.item():.6f}")
    print(f"   Loss visible: {loss_visible.item():.6f}")
    print(f"   Loss dict: {loss_dict}")

    print("\n2. Testing MaskedMSELoss (with manual_mask)...")
    manual_mask = torch.zeros(B, L)
    manual_mask[0, 1000:1500] = 1
    manual_mask[1, 2000:2500] = 1

    loss_masked, loss_visible, loss_dict = criterion(pred, target, wifo_mask, manual_mask)

    print(f"   Loss masked: {loss_masked.item():.6f}")
    print(f"   Loss visible: {loss_visible.item():.6f}")
    print(f"   Loss dict: {loss_dict}")
    print(f"   Manual missing patches: {manual_mask.sum().item()}")

    print("\n3. Testing calculate_masked_loss...")
    total_loss, loss_dict = calculate_masked_loss(
        pred, target, wifo_mask, manual_mask,
        loss_weights=(1.0, 0.1)
    )
    print(f"   Total loss: {total_loss.item():.6f}")
    print(f"   Loss dict: {loss_dict}")

    print("\n4. Testing patchify_csi (real format)...")
    B, T, H, W = 2, 60, 128, 32
    csi_real = torch.randn(B, 2, T, H, W)
    patches = patchify_csi(csi_real, t_patch_size=4, patch_size=4)

    expected_L = (60 // 4) * (128 // 4) * (32 // 4)
    expected_patch_dim = 2 * 4 * 4 * 4
    print(f"   CSI real: {csi_real.shape}")
    print(f"   Patches: {patches.shape}")
    print(f"   Expected: [B, {expected_L}, {expected_patch_dim}]")
    assert patches.shape == (B, expected_L, expected_patch_dim)

    print("\n5. Testing patchify_csi (complex format)...")
    csi_complex = csi_real[:, 0] + 1j * csi_real[:, 1]
    patches_complex = patchify_csi(csi_complex, t_patch_size=4, patch_size=4)

    expected_patch_dim_complex = 4 * 4 * 4
    print(f"   CSI complex: {csi_complex.shape}")
    print(f"   Patches: {patches_complex.shape}")
    print(f"   Expected: [B, {expected_L}, {expected_patch_dim_complex}]")
    assert patches_complex.shape == (B, expected_L, expected_patch_dim_complex)

    print("\n6. Testing unpatchify_csi...")
    csi_reconstructed = unpatchify_csi(
        patches_complex, t_patch_size=4, patch_size=4,
        T=60, H=128, W=32, output_complex=True
    )
    print(f"   Patches: {patches_complex.shape}")
    print(f"   Reconstructed CSI: {csi_reconstructed.shape}")
    assert csi_reconstructed.shape == (B, T, H, W)

    print("\n✅ All tests passed!")
