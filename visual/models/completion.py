"""用 Transformer 根据相邻帧补全缺失的视觉特征。"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=750, dropout=0.1):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):

        if x.dim() == 3 and x.size(0) < x.size(1):
            x = x + self.pe[:x.size(1), :].unsqueeze(0)
        else:
            x = x + self.pe[:x.size(0), :].unsqueeze(1)

        return self.dropout(x)

class TransformerFeatureCompletion(nn.Module):

    def __init__(
        self,
        d_model=1024,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=750
    ):

        super().__init__()

        self.d_model = d_model
        self.max_seq_length = max_seq_length

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_length,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_projection = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        assert src.size(-1) == self.d_model,\
            f"输入特征维度 {src.size(-1)} != 模型维度 {self.d_model}"

        src = self.pos_encoder(src)

        output = self.transformer_encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        output = self.output_projection(output)

        return output

    def predict(self, features, mask):

        self.eval()

        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).bool()

            device = next(self.parameters()).device
            features = features.to(device)
            mask = mask.to(device)

            output = self.forward(features, src_key_padding_mask=mask)

            return output.cpu().numpy()

def create_model(config_dict=None):

    if config_dict is None:
        config_dict = {
            'd_model': TRANSFORMER_D_MODEL,
            'nhead': TRANSFORMER_NHEAD,
            'num_layers': TRANSFORMER_NUM_LAYERS,
            'dim_feedforward': TRANSFORMER_DIM_FEEDFORWARD,
            'dropout': TRANSFORMER_DROPOUT,
            'max_seq_length': MAX_SEQ_LENGTH,
        }

    model = TransformerFeatureCompletion(**config_dict)

    return model

def count_parameters(model):

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = total * 4 / 1024 / 1024

    return total, trainable, param_size

if __name__ == '__main__':
    import numpy as np

    print("=" * 70)
    print("Transformer特征补全模型测试")
    print("=" * 70)

    print("\n创建模型...")
    model = create_model()
    print(f"模型架构: {model.__class__.__name__}")

    print("\n模型参数:")
    count_parameters(model)

    print("\n测试前向传播...")
    batch_size = 2
    seq_len = 750
    d_model = 1024

    features = torch.randn(batch_size, seq_len, d_model)
    mask = torch.rand(batch_size, seq_len) < 0.1

    features[mask] = 0

    print(f"  输入形状: {features.shape}")
    print(f"  Mask形状: {mask.shape}")
    print(f"  缺失帧数: {mask.sum().item()} / {batch_size * seq_len}")

    model.eval()
    with torch.no_grad():
        output = model(features, src_key_padding_mask=mask)

    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min():.3f}, {output.max():.3f}]")

    print("\n测试predict方法...")
    features_np = features.numpy()
    mask_np = mask.numpy()

    completed = model.predict(features_np, mask_np)
    print(f"  补全后形状: {completed.shape}")
    print(f"  补全后范围: [{completed.min():.3f}, {completed.max():.3f}]")

    missing_indices = np.where(mask_np[0])[0]
    if len(missing_indices) > 0:
        print(f"\n检查第1个样本的缺失帧补全:")
        print(f"  缺失帧索引: {missing_indices[:5]}...")
        print(f"  补全前范数: {np.linalg.norm(features_np[0, missing_indices[0]]):.3f}")
        print(f"  补全后范数: {np.linalg.norm(completed[0, missing_indices[0]]):.3f}")

    print("\n" + "=" * 70)
    print("✅ 测试通过！")
    print("=" * 70)
