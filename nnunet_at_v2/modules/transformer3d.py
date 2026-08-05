from __future__ import annotations

import torch
from torch import nn


class ResidualBottleneckTransformer3D(nn.Module):
    """Stage-5 3D bottleneck Transformer with a zero-initialized residual gate."""

    def __init__(
        self,
        channels: int = 320,
        num_heads: int = 8,
        ffn_dim: int = 1280,
        dropout: float = 0.0,
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )

        self.channels = channels

        # Shape-aware 3D convolutional positional encoding.
        self.position_encoding = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, channels),
            nn.Dropout(dropout),
        )

        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected [B, C, D, H, W], received shape {tuple(x.shape)}"
            )

        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, received {x.shape[1]}"
            )

        residual = x
        output_dtype = x.dtype

        # Keep the Transformer branch in FP32 for AMP stability.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            base_tokens = x_float.flatten(2).transpose(1, 2)
            positioned = x_float + self.position_encoding(x_float)

            tokens = positioned.flatten(2).transpose(1, 2)

            normalized = self.norm1(tokens)
            attention_output, _ = self.attention(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )
            transformed = tokens + attention_output
            transformed = transformed + self.ffn(self.norm2(transformed))

            # Gate only the Transformer-induced feature change.
            delta = transformed - base_tokens
            output_tokens = base_tokens + self.residual_scale.float() * delta

            output = (
                output_tokens.transpose(1, 2)
                .contiguous()
                .reshape(residual.shape)
            )

        return output.to(dtype=output_dtype)