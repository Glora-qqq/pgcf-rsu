"""Location, Combined Loss of Posture and Correction Factor. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class MultiLevel6DPoseLoss(nn.Module):

    def __init__(self,
                 alpha=1.0,
                 omega=0.5,
                 beta=0.5,
                 gamma=0.1,
                 delta=0.01,
                 dimension_weight=0.0,
                 position_loss_type='hybrid',
                 orientation_loss_type='geodesic'):
        super().__init__()
        self.alpha = alpha
        self.omega = omega
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.position_loss_type = position_loss_type
        self.orientation_loss_type = orientation_loss_type

    def compute_position_loss(self, pred_positions, gt_positions):

        if self.position_loss_type == 'mse':
            loss = F.mse_loss(pred_positions, gt_positions)
        elif self.position_loss_type == 'l1':
            loss = F.l1_loss(pred_positions, gt_positions)
        elif self.position_loss_type == 'hybrid':
            loss = F.l1_loss(pred_positions, gt_positions) +\
                   0.5 * F.mse_loss(pred_positions, gt_positions)
        elif self.position_loss_type == 'huber':
            loss = F.smooth_l1_loss(pred_positions, gt_positions)
        else:
            raise ValueError(f"Unknown location loss type: {self.position_loss_type}")

        loss_x = F.mse_loss(pred_positions[..., 0], gt_positions[..., 0])
        loss_y = F.mse_loss(pred_positions[..., 1], gt_positions[..., 1])
        loss_z = F.mse_loss(pred_positions[..., 2], gt_positions[..., 2])

        loss_details = {
            'pos_x': float(loss_x.item()),
            'pos_y': float(loss_y.item()),
            'pos_z': float(loss_z.item())
        }

        return loss, loss_details

    def compute_orientation_loss(self, pred_angles, gt_angles):

        if self.orientation_loss_type == 'mse':
            return F.mse_loss(pred_angles, gt_angles)

        elif self.orientation_loss_type == 'l1':
            return F.l1_loss(pred_angles, gt_angles)

        elif self.orientation_loss_type == 'geodesic':
            angle_diff = pred_angles - gt_angles

            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

            return torch.mean(angle_diff ** 2)

        else:
            raise ValueError(f"Unsupported orientation loss type: {self.orientation_loss_type}")

    def compute_gps_correction_loss(self, pred_poses, gt_poses, gps_poses, correction_delta):

        pred_positions = pred_poses[..., :3]
        gt_positions = gt_poses[..., :3]
        gps_positions = gps_poses[..., :3]
        correction_positions = correction_delta[..., :3]

        gps_errors = torch.norm(gps_positions - gt_positions, dim=-1)

        corrected_errors = torch.norm(pred_positions - gt_positions, dim=-1)

        improvement = gps_errors - corrected_errors

        correction_loss = F.relu(-improvement).mean()

        correction_magnitude = torch.norm(correction_positions, dim=-1)
        magnitude_penalty = F.relu(correction_magnitude - 50.0).mean()

        return correction_loss + 0.1 * magnitude_penalty

    def compute_temporal_smooth_loss(self, poses_6d, dt=1.0):

        if poses_6d.dim() == 2:
            poses_6d = poses_6d.unsqueeze(0)

        B, T, D = poses_6d.shape

        if T < 2:
            return torch.tensor(0.0, device=poses_6d.device)

        positions = poses_6d[..., :3]
        orientations = poses_6d[..., 3:]

        position_velocities = (positions[:, 1:, :] - positions[:, :-1, :]) / dt

        if T > 2:
            position_accelerations = position_velocities[:, 1:, :] - position_velocities[:, :-1, :]
            position_smooth_loss = torch.norm(position_accelerations, dim=-1).mean()
        else:
            position_smooth_loss = 0.0

        velocity_magnitudes = torch.norm(position_velocities, dim=-1)
        max_reasonable_velocity = 30.0
        velocity_penalty = F.relu(velocity_magnitudes - max_reasonable_velocity).mean()

        angle_velocities = orientations[:, 1:, :] - orientations[:, :-1, :]

        angle_velocities = torch.atan2(torch.sin(angle_velocities), torch.cos(angle_velocities))

        max_reasonable_angular_velocity = 1.0
        angular_velocity_magnitudes = torch.norm(angle_velocities, dim=-1)
        angular_penalty = F.relu(angular_velocity_magnitudes - max_reasonable_angular_velocity).mean()

        return position_smooth_loss + velocity_penalty + 0.5 * angular_penalty

    def compute_confidence_regularization(self, confidence):

        target_confidence = torch.ones_like(confidence) * 0.65
        confidence_reg = F.mse_loss(confidence, target_confidence)

        extreme_penalty = (
            F.relu(confidence - 0.95).mean() +
            F.relu(0.05 - confidence).mean()
        )

        return confidence_reg + extreme_penalty

    def forward(self,
                pred_poses_6d,
                gt_poses_6d,
                gps_poses_6d,
                correction_delta,
                confidence,
                actual_lengths=None,
                use_adaptive=False):

        total_loss = 0.0
        valid_samples = 0

        total_position_loss = 0.0
        total_orientation_loss = 0.0
        total_correction_loss = 0.0
        total_smooth_loss = 0.0
        total_confidence_reg = 0.0

        total_pos_x = 0.0
        total_pos_y = 0.0
        total_pos_z = 0.0

        batch_size = pred_poses_6d.shape[0]

        for b in range(batch_size):
            if actual_lengths is not None:
                valid_len = actual_lengths[b].item()
            else:
                valid_len = pred_poses_6d.shape[1]

            if valid_len <= 0:
                continue

            pred = pred_poses_6d[b, :valid_len, :]
            gt = gt_poses_6d[b, :valid_len, :]
            gps = gps_poses_6d[b, :valid_len, :]
            corr_delta = correction_delta[b, :valid_len, :]
            conf = confidence[b, :valid_len, :]

            if torch.sum(torch.abs(gt)) == 0:
                continue

            zero = pred.new_zeros(())

            if self.alpha != 0:
                position_loss, pos_details = self.compute_position_loss(pred[..., :3], gt[..., :3])
            else:
                position_loss = zero
                pos_details = {'pos_x': 0.0, 'pos_y': 0.0, 'pos_z': 0.0}

            if self.omega != 0:
                orientation_loss = self.compute_orientation_loss(pred[..., 3:], gt[..., 3:])
            else:
                orientation_loss = zero

            if self.beta != 0:
                correction_loss = self.compute_gps_correction_loss(pred, gt, gps, corr_delta)
            else:
                correction_loss = zero

            if self.gamma != 0:
                smooth_loss = self.compute_temporal_smooth_loss(pred)
            else:
                smooth_loss = zero

            if self.delta != 0:
                confidence_reg = self.compute_confidence_regularization(conf)
            else:
                confidence_reg = zero

            total_position_loss += position_loss
            total_orientation_loss += orientation_loss
            total_correction_loss += correction_loss
            total_smooth_loss += smooth_loss
            total_confidence_reg += confidence_reg

            total_pos_x += pos_details['pos_x']
            total_pos_y += pos_details['pos_y']
            total_pos_z += pos_details['pos_z']

            sample_loss = (
                self.alpha * position_loss +
                self.omega * orientation_loss +
                self.beta * correction_loss +
                self.gamma * smooth_loss +
                self.delta * confidence_reg
            )

            total_loss += sample_loss
            valid_samples += 1

        if valid_samples == 0:
            total_loss_value = torch.tensor(0.0, device=pred_poses_6d.device, requires_grad=True)
            loss_dict = {
                'position': 0.0,
                'orientation': 0.0,
                'gps_correction': 0.0,
                'smooth': 0.0,
                'confidence_reg': 0.0,
                'pos_x': 0.0,
                'pos_y': 0.0,
                'pos_z': 0.0
            }
            return total_loss_value, loss_dict

        avg_total_loss = total_loss / valid_samples

        loss_dict = {
            'position': float((total_position_loss / valid_samples).item()),
            'orientation': float((total_orientation_loss / valid_samples).item()),
            'gps_correction': float((total_correction_loss / valid_samples).item()),
            'smooth': float((total_smooth_loss / valid_samples).item()),
            'confidence_reg': float((total_confidence_reg / valid_samples).item()),
            'pos_x': float(total_pos_x / valid_samples),
            'pos_y': float(total_pos_y / valid_samples),
            'pos_z': float(total_pos_z / valid_samples)
        }

        return avg_total_loss, loss_dict
