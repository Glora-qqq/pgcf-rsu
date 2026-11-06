"""视觉补全阶段的数据路径、DINOv2 和 Transformer 配置。"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..'))
DATA_ROOT = os.environ.get('DATA_ROOT', os.path.join(_REPO, 'data'))
PROJECT_ROOT = DATA_ROOT

BASE_DIR = os.environ.get('RGB_WORK_DIR', os.path.join(_HERE, 'outputs'))

IMAGE_DIR_TRAIN = os.environ.get('IMAGE_DIR_TRAIN', os.path.join(DATA_ROOT, 'images', 'train'))
IMAGE_DIR_TEST = os.environ.get('IMAGE_DIR_TEST', os.path.join(DATA_ROOT, 'images', 'test'))

MASK_DIR_TRAIN = os.path.join(BASE_DIR, 'mask-RGB-train')
MASK_DIR_TEST = os.path.join(BASE_DIR, 'mask-RGB-test')

ENV_FEATURE_DIR_TRAIN = os.path.join(BASE_DIR, 'features/train')
ENV_FEATURE_DIR_TEST = os.path.join(BASE_DIR, 'features/test')

DINOV2_ROOT = os.environ.get('DINOV2_ROOT', os.path.join(_REPO, 'third_party', 'dinov2'))
DINOV2_WEIGHTS_DIR = os.environ.get('DINOV2_WEIGHTS_DIR', os.path.join(DINOV2_ROOT, 'weights'))
DINOV2_VITL14_WEIGHTS = os.path.join(DINOV2_WEIGHTS_DIR, 'dinov2_vitl14_pretrain.pth')
DINOV2_VITL14_REG4_WEIGHTS = os.path.join(DINOV2_WEIGHTS_DIR, 'dinov2_vitl14_reg4_pretrain.pth')

DINOV2_DEFAULT_WEIGHTS = os.environ.get('DINOV2_WEIGHTS', DINOV2_VITL14_REG4_WEIGHTS)

SCENES = ['Scene1', 'Scene2', 'Scene3']

SCENE_RSF_MAP = {
    'Scene1': ['RSF1', 'RSF4', 'RSF7'],
    'Scene2': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
    'Scene3': ['RSF1', 'RSF3', 'RSF7', 'RSF11'],
}

ALL_RSFS = ['RSF1', 'RSF3', 'RSF4', 'RSF7', 'RSF11']

VIEWS = ['left', 'middle', 'right']

TOTAL_FRAMES = 750

IMAGE_NAME_TEMPLATE = '{scene}_{rsf}_snapshot{frame}_{view}_RGB.png'

MASK_NAME_TEMPLATE = '{scene}_{rsf}_RGBmask.npy'

FEATURE_NAME_TEMPLATE = '{scene}_{rsf}_{view}_features.npy'

DINOV2_MODEL_TYPE = 'vitl14_reg4'

FEATURE_DIM = 1024

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GPU_IDS = [1 , 3]

BATCH_SIZE_PER_GPU = 256

ENABLE_MULTI_GPU = len(GPU_IDS) > 1

DEVICE = f'cuda:{GPU_IDS[0]}' if len(GPU_IDS) > 0 else 'cpu'

BATCH_SIZE = BATCH_SIZE_PER_GPU

SAVE_MASK_WITH_FEATURES = True

SHOW_PROGRESS = True

TRANSFORMER_D_MODEL = FEATURE_DIM
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 8
TRANSFORMER_DIM_FEEDFORWARD = 4096
TRANSFORMER_DROPOUT = 0.1

USE_POSITIONAL_ENCODING = True
MAX_SEQ_LENGTH = TOTAL_FRAMES

TRANSFORMER_LEARNING_RATE = 1e-4
TRANSFORMER_BATCH_SIZE = 128
TRANSFORMER_EPOCHS = 50
TRANSFORMER_WARMUP_EPOCHS = 5

LOSS_TYPE = 'mse'
MASK_LOSS_WEIGHT = 1.0
VALID_LOSS_WEIGHT = 0.1

TRANSFORMER_CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
TRANSFORMER_BEST_MODEL = os.path.join(TRANSFORMER_CHECKPOINT_DIR, 'best_model.pth')

def get_image_path(image_dir, scene, rsf, frame, view):

    filename = IMAGE_NAME_TEMPLATE.format(
        scene=scene,
        rsf=rsf,
        frame=frame,
        view=view
    )
    return os.path.join(image_dir, filename)

def get_mask_path(mask_dir, scene, rsf):

    filename = MASK_NAME_TEMPLATE.format(scene=scene, rsf=rsf)
    return os.path.join(mask_dir, scene, filename)

def get_feature_path(feature_dir, scene, rsf, view):

    filename = FEATURE_NAME_TEMPLATE.format(scene=scene, rsf=rsf, view=view)
    rsf_dir = os.path.join(feature_dir, scene, rsf)
    os.makedirs(rsf_dir, exist_ok=True)
    return os.path.join(rsf_dir, filename)

def get_all_samples(image_dir, mask_dir):

    samples = []

    for scene in SCENES:
        rsf_list = SCENE_RSF_MAP.get(scene, [])

        for rsf in rsf_list:
            mask_path = get_mask_path(mask_dir, scene, rsf)

            if os.path.exists(mask_path):
                for view in VIEWS:
                    samples.append((scene, rsf, view))

    return samples

def print_config():

    print("=" * 70)
    print("环境特征提取配置")
    print("=" * 70)

    print("\n📁 路径配置:")
    print(f"  图像目录(训练): {IMAGE_DIR_TRAIN}")
    print(f"  Mask目录(训练): {MASK_DIR_TRAIN}")
    print(f"  环境特征目录(训练): {ENV_FEATURE_DIR_TRAIN}")
    print(f"  DINOv2权重: {DINOV2_DEFAULT_WEIGHTS}")
    print(f"  Transformer模型: {TRANSFORMER_BEST_MODEL}")

    print("\n🎯 数据配置:")
    print(f"  场景: {SCENES}")
    print(f"  RSF配置:")
    for scene, rsfs in SCENE_RSF_MAP.items():
        print(f"    {scene}: {rsfs} ({len(rsfs)}个RSF)")
    print(f"  视角: {VIEWS}")
    print(f"  总帧数: {TOTAL_FRAMES}")
    print(f"  理论样本数: {sum(len(rsfs) for rsfs in SCENE_RSF_MAP.values()) * len(VIEWS)}")

    print("\n🤖 模型配置:")
    print(f"  模型类型: {DINOV2_MODEL_TYPE}")
    print(f"  特征维度: {FEATURE_DIM}")
    print(f"  图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")

    print("\n⚙️ 运行配置:")
    print(f"  GPU列表: {GPU_IDS}")
    print(f"  多GPU并行: {'启用' if ENABLE_MULTI_GPU else '禁用'}")
    print(f"  主设备: {DEVICE}")
    print(f"  每GPU批处理大小: {BATCH_SIZE_PER_GPU}")

    print("\n🔄 Transformer补全配置:")
    print(f"  模型维度: {TRANSFORMER_D_MODEL}")
    print(f"  注意力头数: {TRANSFORMER_NHEAD}")
    print(f"  层数: {TRANSFORMER_NUM_LAYERS}")
    print(f"  序列长度: {MAX_SEQ_LENGTH}")
    print(f"  批大小: {TRANSFORMER_BATCH_SIZE}")
    print(f"  学习率: {TRANSFORMER_LEARNING_RATE}")
    print(f"  训练轮数: {TRANSFORMER_EPOCHS}")

    train_samples = get_all_samples(IMAGE_DIR_TRAIN, MASK_DIR_TRAIN)
    print(f"\n📊 训练集样本数: {len(train_samples)}")

    print("=" * 70)

if __name__ == '__main__':
    print_config()
