"""用 Transformer 对融合后的特征做时间建模。"""

import torch
import torch.nn as nn
import math
from typing import Optional, Literal

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerTemporalModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        x = self.input_proj(x)

        x = self.pos_encoder(x)

        if mask is not None:
            mask = mask.bool()

        out = self.transformer_encoder(x, src_key_padding_mask=mask)

        out = self.output_norm(out)

        return out

class LSTMTemporalModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        out, _ = self.lstm(x)

        out = self.output_norm(out)

        return out

class TemporalModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        model_type: Literal['transformer', 'lstm'] = 'transformer',
        hidden_dim: int = 256,
        num_layers: int = 4,
        **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.model_type = model_type
        self.hidden_dim = hidden_dim

        if model_type == 'transformer':
            self.model = TransformerTemporalModel(
                input_dim=input_dim,
                d_model=hidden_dim,
                num_layers=num_layers,
                **kwargs
            )
            self.output_dim = hidden_dim

        elif model_type == 'lstm':
            self.model = LSTMTemporalModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                **kwargs
            )
            self.output_dim = self.model.output_dim

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        return self.model(x, mask)

if __name__ == '__main__':
    print("\n=== Test Transformer Temporal Model ===")

    B, T = 4, 128
    input_dim = 256

    transformer_model = TemporalModel(
        input_dim=input_dim,
        model_type='transformer',
        hidden_dim=256,
        num_layers=4,
        nhead=8
    )

    x = torch.randn(B, T, input_dim)

    out_transformer = transformer_model(x)

    print(f"Input: {x.shape}")
    print(f"Output (Transformer): {out_transformer.shape}")
    print(f"Parameters: {sum(p.numel() for p in transformer_model.parameters()) / 1e6:.2f}M")

    print("\n=== Test LSTM Temporal Model ===")

    lstm_model = TemporalModel(
        input_dim=input_dim,
        model_type='lstm',
        hidden_dim=256,
        num_layers=2,
        bidirectional=True
    )

    out_lstm = lstm_model(x)

    print(f"Input: {x.shape}")
    print(f"Output (LSTM): {out_lstm.shape}")
    print(f"Parameters: {sum(p.numel() for p in lstm_model.parameters()) / 1e6:.2f}M")

    print("\n=== Model Comparison ===")
    print(f"Transformer output dim: {transformer_model.output_dim}")
    print(f"LSTM output dim: {lstm_model.output_dim}")

    print("\n=== Test with Mask ===")
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, -10:] = True

    out_masked = transformer_model(x, mask)
    print(f"Output with mask: {out_masked.shape}")

    print("\n=== All tests passed! ===")
