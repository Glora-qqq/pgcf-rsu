"""导出帧级特征聚合函数。"""

from .aggregate import (
    patch_to_frame_features,
    aggregate_window_features_to_frames,
    crop_features_by_gps,
)

__all__ = [
    'patch_to_frame_features',
    'aggregate_window_features_to_frames',
    'crop_features_by_gps',
]
