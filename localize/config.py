"""Multi-side unit positioning training path, Scenario and Model Configuration. """

import os
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
_DATA = Path(os.environ.get('DATA_ROOT', PROJECT_ROOT.parent / 'data'))
SRC_ROOT = _DATA
DATA_ROOT = _DATA / 'train'

TRAIN_DATA_ROOT = DATA_ROOT
WIFO_FEATURES_ROOT = Path(os.environ.get(
    'WIFO_FEATURES_ROOT',
    PROJECT_ROOT.parent / 'finetune' / 'features' / 'train',
))
LOG_ROOT = PROJECT_ROOT / 'logs'
CHECKPOINT_ROOT = LOG_ROOT / 'checkpoints'

SCENES = ['Scene1', 'Scene2', 'Scene3']

SCENE_RSF_CONFIG = {
    'Scene1': ['RSF1', 'RSF4', 'RSF7'],
    'Scene2': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
    'Scene3': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
}

SAME_ROAD_DISTANCE_THRESHOLD = 200.0

SAME_ROAD_LATERAL_THRESHOLD = 30.0
SAME_ROAD_VIEWING_THRESHOLD = 90.0
SAME_ROAD_DISTANCE_WEIGHT = 0.4
SAME_ROAD_LATERAL_WEIGHT = 0.4
SAME_ROAD_VIEWING_WEIGHT = 0.2

SPATIAL_FEATURE_DIM = 12
SCENE_ADAPTIVE_SCALES = {
    'Scene1': {
        'center_distance_2d': 2.5,
        'height_diff': 2.0,
        'rsf_viewing_angle': 2.5,
        'heading_alignment': 2.5,
        'longitudinal_offset': 2.0,
        'lateral_offset': 4.0,
        'approaching_trend_2d': 3.0,
        'same_road_confidence': 4.0,
        'x_offset': 1.5,
        'y_offset': 1.5,
        'relative_angle': 1.0,
        'elevation_angle': 1.5,
    }
}

class TrainConfig:

    def __init__(
        self,
        mode='multi_rsf',
        scenes=['Scene1', 'Scene2', 'Scene3'],
        time_window=None,
        stride=64,
        batch_size=1,
        num_workers=0,
        num_epochs=300,
        learning_rate=5e-5,
        weight_decay=1e-4,
        same_road_distance_threshold=None,
        same_road_lateral_threshold=None,
        same_road_viewing_threshold=None,
        same_road_distance_weight=None,
        same_road_lateral_weight=None,
        same_road_viewing_weight=None,
        optimal_distance_min=None,
        optimal_distance_max=None,
        optimal_distance_center=None,
        wifo_dim=512,
        spatial_dim=None,
        spatial_quality_estimator_hidden=32,
        use_interpretable_attention=True,
        quality_estimation_dropout=0.2,
        rsf_fusion_type='spatial_quality',
        fusion_output_dim=256,
        fusion_dropout=0.1,
        fusion_num_heads=8,
        fusion_num_layers=1,
        temporal_type='transformer',
        temporal_hidden_dim=256,
        temporal_num_layers=6,
        temporal_nhead=8,
        temporal_dropout=0.1,
        multi_rsf_fusion_mode='spatial_quality',
        position_mode='gps_guided',
        position_hidden_dim=256,
        position_dropout=0.1,
        predict_gps_trust=True,
        use_gps_prior=True,
        use_position_noise=False,
        use_orientation_noise=True,
        car_yaw_turning_ratio=0.10,
        car_yaw_straight_noise_std_deg=3.0,
        at_orientation_noise_std_deg=0.0,
        use_6d_pose=True,
        max_position_correction=5.0,
        max_orientation_correction=0.5,
        predict_uncertainty=True,
        position_smooth_factor=0.1,
        orientation_smooth_factor=0.1,
        learnable_smoothing=False,
        use_adaptive_kalman=True,
        csi_fusion_strategy='average',
        use_input_kalman=True,
        input_kalman_process_noise=0.10,
        input_kalman_measurement_noise=1.6,
        use_at_ukf=True,
        at_ukf_orientation=False,
        at_ukf_process_noise=0.1,
        at_ukf_process_noise_ori=0.01,
        at_ukf_measurement_noise=1.6,
        at_ukf_measurement_noise_ori=0.05,
        at_ukf_dt=0.05,
        use_output_adaptive_filter=False,
        output_car_kalman_process_noise=0.10,
        output_car_kalman_measurement_noise=1.6,
        output_at_ukf_process_noise=0.10,
        output_at_ukf_measurement_noise=0.8,
        loss_alpha=1.0,
        loss_omega=0.5,
        loss_beta=0.5,
        loss_gamma=0.1,
        loss_delta=0.01,
        position_loss_type='hybrid',
        orientation_loss_type='geodesic',
        resume_from=None,
        resume_optimizer=False,
        optimizer='adamw',
        lr_scheduler='cosine',
        cosine_min_lr=5e-7,
        cosine_warmup_epochs=5,
        device='cpu',
        random_seed=42,
        save_every=50,
        log_every=5,
        verbose=False,
        max_sequence_length=1500,
        gradient_clip_norm=0.5,
        spatial_feature_mask=None,
        n_rsu_max=4,
        concat_proj_hidden=None,
        ablation_modality='full',
        geo_proj_hidden=128,
        completeness_json=None,
    ):
        self.scenes = scenes
        self.mode = mode

        self.time_window = time_window
        self.stride = stride if stride else (time_window // 2 if time_window else None)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.same_road_distance_threshold = same_road_distance_threshold if same_road_distance_threshold is not None else globals().get('SAME_ROAD_DISTANCE_THRESHOLD', 200.0)
        self.same_road_lateral_threshold = same_road_lateral_threshold if same_road_lateral_threshold is not None else globals().get('SAME_ROAD_LATERAL_THRESHOLD', 30.0)
        self.same_road_viewing_threshold = same_road_viewing_threshold if same_road_viewing_threshold is not None else globals().get('SAME_ROAD_VIEWING_THRESHOLD', 90.0)
        self.same_road_distance_weight = same_road_distance_weight if same_road_distance_weight is not None else globals().get('SAME_ROAD_DISTANCE_WEIGHT', 0.4)
        self.same_road_lateral_weight = same_road_lateral_weight if same_road_lateral_weight is not None else globals().get('SAME_ROAD_LATERAL_WEIGHT', 0.4)
        self.same_road_viewing_weight = same_road_viewing_weight if same_road_viewing_weight is not None else globals().get('SAME_ROAD_VIEWING_WEIGHT', 0.2)

        weight_sum = self.same_road_distance_weight + self.same_road_lateral_weight + self.same_road_viewing_weight
        if abs(weight_sum - 1.0) > 1e-6:
            if verbose:
                print(f"weights sum = {weight_sum:.3f}, auto normalize to 1.0")
            self.same_road_distance_weight /= weight_sum
            self.same_road_lateral_weight /= weight_sum
            self.same_road_viewing_weight /= weight_sum

        self.optimal_distance_min = optimal_distance_min
        self.optimal_distance_max = optimal_distance_max
        self.optimal_distance_center = optimal_distance_center

        self.wifo_dim = wifo_dim
        self.spatial_dim = globals().get('SPATIAL_FEATURE_DIM', 12) if spatial_dim is None else spatial_dim

        self.fusion_type = rsf_fusion_type
        self.rsf_fusion_type = rsf_fusion_type
        self.fusion_output_dim = fusion_output_dim
        self.fusion_dropout = fusion_dropout
        self.fusion_num_heads = fusion_num_heads
        self.fusion_num_layers = fusion_num_layers
        self.spatial_quality_estimator_hidden = spatial_quality_estimator_hidden
        self.use_interpretable_attention = use_interpretable_attention
        self.quality_estimation_dropout = quality_estimation_dropout

        self.temporal_type = temporal_type
        self.temporal_hidden_dim = temporal_hidden_dim
        self.temporal_num_layers = temporal_num_layers
        self.temporal_nhead = temporal_nhead
        self.temporal_dropout = temporal_dropout

        self.multi_rsf_fusion_mode = multi_rsf_fusion_mode

        self.position_mode = position_mode
        self.position_hidden_dim = position_hidden_dim
        self.position_dropout = position_dropout
        self.predict_gps_trust = predict_gps_trust
        self.use_gps_prior = use_gps_prior

        self.use_position_noise = use_position_noise
        self.use_orientation_noise = use_orientation_noise
        self.car_yaw_turning_ratio = car_yaw_turning_ratio
        self.car_yaw_straight_noise_std_deg = car_yaw_straight_noise_std_deg
        self.at_orientation_noise_std_deg = at_orientation_noise_std_deg

        self.use_6d_pose = use_6d_pose
        self.max_position_correction = max_position_correction
        self.max_orientation_correction = max_orientation_correction
        self.predict_uncertainty = predict_uncertainty
        self.position_smooth_factor = position_smooth_factor
        self.orientation_smooth_factor = orientation_smooth_factor
        self.learnable_smoothing = learnable_smoothing

        self.use_adaptive_kalman = use_adaptive_kalman
        self.csi_fusion_strategy = csi_fusion_strategy
        self.use_input_kalman = use_input_kalman
        self.input_kalman_process_noise = input_kalman_process_noise
        self.input_kalman_measurement_noise = input_kalman_measurement_noise

        self.use_at_ukf = use_at_ukf
        self.at_ukf_orientation = at_ukf_orientation
        self.at_ukf_process_noise = at_ukf_process_noise
        self.at_ukf_process_noise_ori = at_ukf_process_noise_ori
        self.at_ukf_measurement_noise = at_ukf_measurement_noise
        self.at_ukf_measurement_noise_ori = at_ukf_measurement_noise_ori
        self.at_ukf_dt = at_ukf_dt

        self.use_output_adaptive_filter = use_output_adaptive_filter
        self.output_car_kalman_process_noise = output_car_kalman_process_noise
        self.output_car_kalman_measurement_noise = output_car_kalman_measurement_noise
        self.output_at_ukf_process_noise = output_at_ukf_process_noise
        self.output_at_ukf_measurement_noise = output_at_ukf_measurement_noise

        self.loss_alpha = loss_alpha
        self.loss_omega = loss_omega
        self.loss_beta = loss_beta
        self.loss_gamma = loss_gamma
        self.loss_delta = loss_delta
        self.position_loss_type = position_loss_type
        self.orientation_loss_type = orientation_loss_type

        self.resume_from = resume_from
        self.resume_optimizer = resume_optimizer

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.cosine_min_lr = cosine_min_lr
        self.cosine_warmup_epochs = cosine_warmup_epochs

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = random_seed
        self.save_every = save_every
        self.log_every = log_every
        self.verbose = verbose
        self.max_sequence_length = max_sequence_length
        self.gradient_clip_norm = gradient_clip_norm
        self.spatial_feature_mask = spatial_feature_mask
        self.n_rsu_max = n_rsu_max
        self.concat_proj_hidden = concat_proj_hidden if concat_proj_hidden is not None else wifo_dim
        self.ablation_modality = str(ablation_modality or 'full').lower()
        self.geo_proj_hidden = geo_proj_hidden
        self.completeness_json = completeness_json

        self.rsf_configs = {scene: SCENE_RSF_CONFIG[scene] for scene in scenes}
        self.num_rsfs = {scene: len(SCENE_RSF_CONFIG[scene]) for scene in scenes}

    def print_config(self):

        print("=" * 80)
        print("PGCF-RSU Locate Training Configuration")
        print("=" * 80)

        print(f"\n[Scenario Configuration]")
        for scene in self.scenes:
            print(f"  {scene}: {self.rsf_configs[scene]} ({self.num_rsfs[scene]} RSFs)")

        print(f"\n[Data loading]")
        print(f"  Mode: {self.mode}")
        print(f"  Time window: {self.time_window}")
        print(f"  BatchSize: {self.batch_size}")
        print(f"  WiFoFeature Root: {WIFO_FEATURES_ROOT}")

        print(f"\n[Model Architecture]")
        print(f"  WiFoFeatures: {self.wifo_dim}D")
        print(f"  Spatial characteristics: {self.spatial_dim}D")
        print(f"  MultipleRSFFusion Mode: {self.multi_rsf_fusion_mode}")
        print(f"  RSFFusion Type: {self.rsf_fusion_type}")
        print(f"  Fusion Output Dimension: {self.fusion_output_dim}D")
        print(f"  Temporal modeling: {self.temporal_type} ({self.temporal_num_layers}Layer)")
        print(f"  Position Regression: {self.position_mode}")
        print(f"  CSIConvergence Strategy: {self.csi_fusion_strategy}")
        if str(self.csi_fusion_strategy).lower() == "concat_proj":
            print(f"  ConcatProj: n_rsu_max={self.n_rsu_max}, hidden={self.concat_proj_hidden}")

        print(f"\n[Training parameters]")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Optimizer: {self.optimizer}")
        print(f"  Equipment: {self.device}")
        print(f"  Loss weight α/ω/β/γ/δ: "
              f"{self.loss_alpha}/{self.loss_omega}/{self.loss_beta}/"
              f"{self.loss_gamma}/{self.loss_delta}")
        if self.spatial_feature_mask:
            print(f"  Spatial Feature Mask Dimension: {self.spatial_feature_mask}")
        print(f"  Ablation Mode ablation_modality: {self.ablation_modality}")
        if self.ablation_modality == 'gps_geo':
            print(f"  geo_proj_hidden: {self.geo_proj_hidden}")
        if self.ablation_modality in ('no_csi_completion', 'no_completion'):
            print(f"  CSICompleteness Table: {self.completeness_json or '(default data_analysis)'}")

        print("=" * 80)

def create_output_dirs():

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    print("[INFO] Output directory created")
