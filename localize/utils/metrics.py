"""位置误差和轨迹对比指标。"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import directed_hausdorff

class PositionMetrics:

    def __init__(self):
        self.reset()

    def reset(self):

        self.all_position_errors = []
        self.all_gps_errors = []
        self.all_improvements = []
        self.all_confidence_scores = []

    def compute_position_errors(self, pred_positions, gt_positions, actual_lengths=None):

        batch_size = pred_positions.shape[0]
        all_errors = []

        for b in range(batch_size):
            if actual_lengths is not None:
                valid_len = actual_lengths[b].item()
            else:
                valid_len = pred_positions.shape[1]

            if valid_len <= 0:
                continue

            pred = pred_positions[b, :valid_len, :].detach().cpu().numpy()
            gt = gt_positions[b, :valid_len, :].detach().cpu().numpy()

            errors = np.linalg.norm(pred - gt, axis=1)
            all_errors.extend(errors.tolist())

        if len(all_errors) == 0:
            return {'rmse': 0, 'mae': 0, 'max_error': 0, 'median_error': 0}

        all_errors = np.array(all_errors)

        return {
            'rmse': float(np.sqrt(np.mean(all_errors**2))),
            'mae': float(np.mean(all_errors)),
            'max_error': float(np.max(all_errors)),
            'median_error': float(np.median(all_errors)),
            'std_error': float(np.std(all_errors)),
            'p95_error': float(np.percentile(all_errors, 95)),
            'p99_error': float(np.percentile(all_errors, 99))
        }

    def compute_gps_improvement(self, pred_positions, gt_positions, gps_positions, actual_lengths=None):

        batch_size = pred_positions.shape[0]
        improvements = []
        gps_errors = []
        pred_errors = []

        for b in range(batch_size):
            if actual_lengths is not None:
                valid_len = actual_lengths[b].item()
            else:
                valid_len = pred_positions.shape[1]

            if valid_len <= 0:
                continue

            pred = pred_positions[b, :valid_len, :].detach().cpu().numpy()
            gt = gt_positions[b, :valid_len, :].detach().cpu().numpy()
            gps = gps_positions[b, :valid_len, :].detach().cpu().numpy()

            gps_error = np.linalg.norm(gps - gt, axis=1)
            pred_error = np.linalg.norm(pred - gt, axis=1)

            improvement = gps_error - pred_error

            gps_errors.extend(gps_error.tolist())
            pred_errors.extend(pred_error.tolist())
            improvements.extend(improvement.tolist())

        if len(improvements) == 0:
            return {'improvement_mean': 0, 'improvement_ratio': 0, 'better_frames_ratio': 0}

        improvements = np.array(improvements)
        gps_errors = np.array(gps_errors)
        pred_errors = np.array(pred_errors)

        return {
            'improvement_mean': float(np.mean(improvements)),
            'improvement_median': float(np.median(improvements)),
            'improvement_std': float(np.std(improvements)),
            'improvement_ratio': float(np.mean(improvements) / (np.mean(gps_errors) + 1e-6)),
            'better_frames_ratio': float(np.mean(improvements > 0)),
            'gps_rmse': float(np.sqrt(np.mean(gps_errors**2))),
            'pred_rmse': float(np.sqrt(np.mean(pred_errors**2))),
            'rmse_improvement': float(np.sqrt(np.mean(gps_errors**2)) - np.sqrt(np.mean(pred_errors**2)))
        }

    def compute_trajectory_quality(self, pred_positions, actual_lengths=None):

        batch_size = pred_positions.shape[0]
        velocity_stats = []
        acceleration_stats = []
        smoothness_scores = []

        for b in range(batch_size):
            if actual_lengths is not None:
                valid_len = actual_lengths[b].item()
            else:
                valid_len = pred_positions.shape[1]

            if valid_len < 3:
                continue

            traj = pred_positions[b, :valid_len, :].detach().cpu().numpy()

            velocities = np.diff(traj, axis=0)
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)

            accelerations = np.diff(velocities, axis=0)
            acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)

            velocity_stats.append({
                'mean': float(np.mean(velocity_magnitudes)),
                'max': float(np.max(velocity_magnitudes)),
                'std': float(np.std(velocity_magnitudes))
            })

            acceleration_stats.append({
                'mean': float(np.mean(acceleration_magnitudes)),
                'max': float(np.max(acceleration_magnitudes)),
                'std': float(np.std(acceleration_magnitudes))
            })

            if len(acceleration_magnitudes) > 0:
                smoothness = 1.0 / (1.0 + np.std(acceleration_magnitudes))
                smoothness_scores.append(smoothness)

        if len(velocity_stats) == 0:
            return {'velocity': {}, 'acceleration': {}, 'smoothness': 0}

        velocity_summary = {
            'mean_velocity': float(np.mean([v['mean'] for v in velocity_stats])),
            'max_velocity': float(np.max([v['max'] for v in velocity_stats])),
            'velocity_variability': float(np.mean([v['std'] for v in velocity_stats]))
        }

        acceleration_summary = {
            'mean_acceleration': float(np.mean([a['mean'] for a in acceleration_stats])),
            'max_acceleration': float(np.max([a['max'] for a in acceleration_stats])),
            'acceleration_variability': float(np.mean([a['std'] for a in acceleration_stats]))
        }

        return {
            'velocity': velocity_summary,
            'acceleration': acceleration_summary,
            'smoothness': float(np.mean(smoothness_scores)) if smoothness_scores else 0
        }

    def compute_confidence_calibration(self, confidence, pred_positions, gt_positions, actual_lengths=None):

        batch_size = pred_positions.shape[0]
        conf_error_pairs = []

        for b in range(batch_size):
            if actual_lengths is not None:
                valid_len = actual_lengths[b].item()
            else:
                valid_len = pred_positions.shape[1]

            if valid_len <= 0:
                continue

            pred = pred_positions[b, :valid_len, :].detach().cpu().numpy()
            gt = gt_positions[b, :valid_len, :].detach().cpu().numpy()
            conf = confidence[b, :valid_len, :].detach().cpu().numpy()

            errors = np.linalg.norm(pred - gt, axis=1)
            confidences = np.mean(conf, axis=1)

            for error, conf_val in zip(errors, confidences):
                conf_error_pairs.append((conf_val, error))

        if len(conf_error_pairs) == 0:
            return {'correlation': 0, 'calibration_error': 0}

        confidences, errors = zip(*conf_error_pairs)
        confidences = np.array(confidences)
        errors = np.array(errors)

        correlation = float(np.corrcoef(confidences, errors)[0, 1])

        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_calibration_errors = []

        for i in range(n_bins):
            bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_conf = np.mean(confidences[bin_mask])
                bin_error = np.mean(errors[bin_mask])
                calibration_error = abs(bin_conf - (1.0 - bin_error / np.max(errors)))
                bin_calibration_errors.append(calibration_error)

        calibration_error = float(np.mean(bin_calibration_errors)) if bin_calibration_errors else 0

        return {
            'correlation': correlation,
            'calibration_error': calibration_error,
            'mean_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences))
        }

    def compute_all_metrics(self,
                          pred_positions,
                          gt_positions,
                          gps_positions,
                          confidence=None,
                          actual_lengths=None):

        metrics = self.compute_position_errors(pred_positions, gt_positions, actual_lengths)

        gps_improvement = self.compute_gps_improvement(
            pred_positions, gt_positions, gps_positions, actual_lengths
        )

        trajectory_quality = self.compute_trajectory_quality(pred_positions, actual_lengths)

        confidence_calibration = {}
        if confidence is not None:
            confidence_calibration = self.compute_confidence_calibration(
                confidence, pred_positions, gt_positions, actual_lengths
            )

        return {
            'position_errors': metrics,
            'gps_improvement': gps_improvement,
            'trajectory_quality': trajectory_quality,
            'confidence_calibration': confidence_calibration
        }

    def print_metrics(self, metrics_dict):

        print("=" * 60)
        print("位置预测评估结果")
        print("=" * 60)

        pos_metrics = metrics_dict['position_errors']
        print(f"\n📍 位置误差:")
        print(f"  RMSE: {pos_metrics['rmse']:.3f} 米")
        print(f"  MAE:  {pos_metrics['mae']:.3f} 米")
        print(f"  中位数误差: {pos_metrics['median_error']:.3f} 米")
        print(f"  最大误差: {pos_metrics['max_error']:.3f} 米")
        print(f"  95%分位误差: {pos_metrics['p95_error']:.3f} 米")

        gps_metrics = metrics_dict['gps_improvement']
        print(f"\n🛰️ GPS改进效果:")
        print(f"  GPS RMSE: {gps_metrics['gps_rmse']:.3f} 米")
        print(f"  预测 RMSE: {gps_metrics['pred_rmse']:.3f} 米")
        print(f"  RMSE改进: {gps_metrics['rmse_improvement']:.3f} 米")
        print(f"  改进帧比例: {gps_metrics['better_frames_ratio']:.1%}")
        print(f"  平均改进: {gps_metrics['improvement_mean']:.3f} 米")

        traj_metrics = metrics_dict['trajectory_quality']
        if 'velocity' in traj_metrics and traj_metrics['velocity']:
            print(f"\n🚗 轨迹质量:")
            print(f"  平均速度: {traj_metrics['velocity']['mean_velocity']:.2f} m/s")
            print(f"  最大速度: {traj_metrics['velocity']['max_velocity']:.2f} m/s")
            print(f"  平均加速度: {traj_metrics['acceleration']['mean_acceleration']:.2f} m/s²")
            print(f"  轨迹平滑度: {traj_metrics['smoothness']:.3f}")

        conf_metrics = metrics_dict.get('confidence_calibration', {})
        if conf_metrics:
            print(f"\n🎯 置信度校准:")
            print(f"  置信度-误差相关性: {conf_metrics['correlation']:.3f}")
            print(f"  校准误差: {conf_metrics['calibration_error']:.3f}")
            print(f"  平均置信度: {conf_metrics['mean_confidence']:.3f}")

        print("=" * 60)
