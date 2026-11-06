"""导出坐标变换、路侧天线位置、损失和轨迹指标。"""

from .constants import *
from .constants import get_vehicle_height_correction
from .coordinates import (
    apply_height_correction,
    convert_angles_deg_to_rad,
    rotation_matrix_from_euler,
    rsf_to_world,
)
from .antenna import (
    convert_rsf_extrinsic_to_radians,
    convert_rsf_extrinsic_batch,
    get_rsf_antenna_world_position,
    get_rsf_antenna_world_position_batch,
    get_rsf_full_extrinsic,
    get_rsf_extrinsic_and_antenna_features,
)

from .losses.pose import (
    MultiLevel6DPoseLoss
)

from .metrics import (
    PositionMetrics,
)

__all__ = [
    'SCENES',
    'SCENE_RSF_CONFIG',
    'SCENE_VEHICLES',
    'VEHICLE_HEIGHT_CORRECTION',
    'get_vehicle_height_correction',
    'is_aerial_vehicle',
    'RSF_ANTENNA_ARRAY_POSITION',
    'get_rsf_antenna_position',
    'get_rsf_camera_config',
    'CAMERA_INTRINSIC',
    'CSI_PARAMS',
    'TIMESTAMP_CONFIG',
    'FILE_TEMPLATES',
    'PATH_TEMPLATES',
    'DEG_TO_RAD',
    'RAD_TO_DEG',

    'apply_height_correction',
    'convert_angles_deg_to_rad',
    'rotation_matrix_from_euler',
    'rsf_to_world',

    'convert_rsf_extrinsic_to_radians',
    'convert_rsf_extrinsic_batch',
    'get_rsf_antenna_world_position',
    'get_rsf_antenna_world_position_batch',
    'get_rsf_full_extrinsic',
    'get_rsf_extrinsic_and_antenna_features',

    'MultiLevel6DPoseLoss',
    'PositionMetrics',

]
