# PGCF-RSU

**P**rogressive **G**ated **C**ross-modal **F**usion for multi-**RSU** visual-wireless localization.

中文名：渐进式门控跨模态融合的多路侧单元视觉-无线协同定位。简称 **PGCF-RSU**。

方法面向复杂城市和地空混合交通：用多视角视觉补全无线信道里缺失的观测，再按路侧几何质量把多个路侧单元融合起来，在 GPS 先验上回归地面车辆和空中出租车的六维位姿。代码对应 [2025中国智能车未来挑战赛](https://ivfc.caa.org.cn/info/1017/1121.htm) 离线测试算法大赛赛题三 [无线基座模型 WiFo 赋能的多模态信息智能车定位](https://www.codabench.org/competitions/11207/#/participate-tab)。

## 方法

<img src="docs/figures/overview.jpg" alt="方法总流程" width="720">

```text
多视角 RGB
  → visual/          朝向编码与交叉注意力，补全缺失帧，得到 1024 维视觉特征
CSI + 视觉特征
  → finetune/        渐进门控注入 WiFo，导出每个路侧单元的 512 维特征
GPS + 12 维几何量
  → localize/        按观测质量融合多路侧单元，预测位姿修正
```

门控从很小的初值慢慢打开，避免视觉特征一开始盖过已经预训练好的信道表征。

四组评测的平均绝对误差为 **0.1846、0.2057、0.3080、0.1581 m**。相对原始 GPS 分别下降 **86.69%、85.06%、84.69%、88.49%**，相对 KF/UKF 滤波先验分别下降 **75.01%、72.09%、78.48%、78.69%**。

## 环境

Python 3.9，PyTorch 2.x。

```bash
pip install -r requirements.txt
git clone https://github.com/facebookresearch/dinov2 third_party/dinov2
```

```bash
export DATA_ROOT=/path/to/competition_data
export DINOV2_ROOT=/path/to/dinov2
export DINOV2_WEIGHTS=/path/to/dinov2_vitl14_reg4_pretrain.pth
export WIFO_WEIGHTS=/path/to/WiFo-RGB.pkl
```

`DATA_ROOT` 下需要 `train/`、`test/`、`images/train/`、`images/test/`，以及 RGB 覆盖率文本。赛题数据、WiFo 权重和 DINOv2 权重都不在本仓库里。

阶段之间的文件名需要保持不变：`visual/pack_features.py` 写出 `{Scene}_{RSF}_features.npz`，键名是 `fused`；`finetune` 导出 `{vehicle}_RSF{n}_wifo_features.npy`，定位阶段按这个名字读取。

## 运行

视觉特征，在 `visual/`：

```bash
python build_masks.py
python train_completion.py
python infer_completion.py
python train_fusion.py
python infer_fusion.py
python pack_features.py --fused-dir outputs/fused_features --out-dir ../finetune/fusion-train-RGB
```

微调与导出，在 `finetune/`：

```bash
python masks/build.py
python masks_test/build.py
python train.py
python infer.py
```

定位，在 `localize/`：

```bash
python train.py
python infer.py
```

`localize/infer.py` 读取定位权重，轨迹写到 `localize/predictions/Scene_1`、`Scene_2`、`Scene_3`。地面车辆用二维卡尔曼，空中目标用三维 UKF，开关在 `localize/config_test.py`。

## 代码结构

三个目录各对应一阶段。上一节的命令就是要运行的程序，路径和训练参数改同目录的 `config.py`。`data/`、`models/`、`utils/` 由这些脚本调用，不用单独执行。

- `visual/`：`build_masks.py` 生成缺失帧掩码，`train_completion.py` 与 `infer_completion.py` 补全视觉特征，`train_fusion.py` 与 `infer_fusion.py` 融合三个视角，`pack_features.py` 打包给微调。
- `finetune/`：`masks/build.py`、`masks_test/build.py` 生成掩码，`train.py` 微调，`infer.py` 导出特征。多卡用 `infer_parallel.py`，中断后用 `resume.py` 接着训。
- `localize/`：`train.py` 训练位姿修正，`infer.py` 写出三个场景的轨迹。滤波开关在 `config_test.py`。

## 致谢

- 感谢中国智能车未来挑战赛组委会和赛题三提供数据与 Codabench 评测。
- 信道编码器使用 WiFo：Boxun Liu, Shijian Gao, Xuanyu Liu, Xiang Cheng, Liuqing Yang, *WiFo: Wireless Foundation Model for Channel Prediction*。
- 视觉编码器使用 Meta [DINOv2](https://github.com/facebookresearch/dinov2)，Apache-2.0。

## 许可

本仓库代码使用 MIT License，见 `LICENSE`。WiFo 权重和 DINOv2 遵循各自的发布条款。
