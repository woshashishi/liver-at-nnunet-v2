from __future__ import annotations

import torch
from torch import nn


class ResidualBottleneckTransformer3D(nn.Module):
    """
    Lightweight residual 3D Transformer for the nnU-Net bottleneck.

    v2 specification:
    - embedding_dim = min(channels, 256)
    - 1 Transformer layer
    - 4 attention heads
    - FFN ratio = 2
    - dynamic depthwise Conv3D positional encoding
    - zero-initialized residual scale
    """

    def __init__(
        self,
        channels: int,
        embedding_dim: int | None = None,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be positive, got {channels}"
            )

        if embedding_dim is None:
            embedding_dim = min(
                channels,
                256,
            )

        embedding_dim = int(
            embedding_dim
        )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if embedding_dim > channels:
            raise ValueError(
                "embedding_dim must not exceed "
                f"input channels: {embedding_dim}>{channels}"
            )

        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be positive, got {num_heads}"
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) "
                f"must be divisible by "
                f"num_heads ({num_heads})"
            )

        if ffn_dim is None:
            ffn_dim = 2 * embedding_dim

        ffn_dim = int(
            ffn_dim
        )

        if ffn_dim <= 0:
            raise ValueError(
                f"ffn_dim must be positive, got {ffn_dim}"
            )

        self.channels = int(
            channels
        )

        self.embedding_dim = (
            embedding_dim
        )

        self.num_heads = int(
            num_heads
        )

        self.ffn_dim = (
            ffn_dim
        )

        if self.embedding_dim == self.channels:

            self.input_projection = (
                nn.Identity()
            )

            self.output_projection = (
                nn.Identity()
            )

        else:

            self.input_projection = (
                nn.Conv3d(
                    self.channels,
                    self.embedding_dim,
                    kernel_size=1,
                    bias=False,
                )
            )

            self.output_projection = (
                nn.Conv3d(
                    self.embedding_dim,
                    self.channels,
                    kernel_size=1,
                    bias=False,
                )
            )

        # Dynamic shape-aware positional encoding.
        self.position_encoding = nn.Conv3d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=3,
            padding=1,
            groups=self.embedding_dim,
            bias=True,
        )

        self.norm1 = nn.LayerNorm(
            self.embedding_dim
        )

        self.attention = (
            nn.MultiheadAttention(
                embed_dim=self.embedding_dim,
                num_heads=self.num_heads,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.norm2 = nn.LayerNorm(
            self.embedding_dim
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                self.embedding_dim,
                self.ffn_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.ffn_dim,
                self.embedding_dim,
            ),
            nn.Dropout(dropout),
        )

        self.residual_scale = nn.Parameter(
            torch.tensor(
                float(
                    residual_scale_init
                )
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 5:
            raise ValueError(
                "Expected [B,C,D,H,W], "
                f"received {tuple(x.shape)}"
            )

        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, "
                f"received {x.shape[1]}"
            )

        residual = x
        output_dtype = x.dtype

        # Transformer branch stays FP32 for AMP stability.
        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):

            x_float = x.float()

            embedded = (
                self.input_projection(
                    x_float
                )
            )

            embedded_shape = (
                embedded.shape
            )

            base_tokens = (
                embedded
                .flatten(2)
                .transpose(1, 2)
            )

            positioned = (
                embedded
                + self.position_encoding(
                    embedded
                )
            )

            tokens = (
                positioned
                .flatten(2)
                .transpose(1, 2)
            )

            normalized = (
                self.norm1(tokens)
            )

            attention_output, _ = (
                self.attention(
                    normalized,
                    normalized,
                    normalized,
                    need_weights=False,
                )
            )

            transformed = (
                tokens
                + attention_output
            )

            transformed = (
                transformed
                + self.ffn(
                    self.norm2(
                        transformed
                    )
                )
            )

            # Transformer-induced feature change
            # in the reduced embedding space.
            delta_tokens = (
                transformed
                - base_tokens
            )

            delta_embedded = (
                delta_tokens
                .transpose(1, 2)
                .contiguous()
                .reshape(
                    embedded_shape
                )
            )

            delta_full = (
                self.output_projection(
                    delta_embedded
                )
            )

            output = (
                x_float
                + self.residual_scale.float()
                * delta_full
            )

        return output.to(
            dtype=output_dtype
        )
