from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn


class ChannelAttention3D(nn.Module):
    """3D channel attention using a shared 1x1x1 MLP on avg/max pooled features."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        weights = self.shared_mlp(self.avg_pool(x))
        weights = weights + self.shared_mlp(self.max_pool(x))
        return self.gate(weights)


class SpatialAttention3D(nn.Module):
    """3D spatial attention from channel-wise mean and maximum maps."""

    _SUPPORTED_KERNELS: Final[tuple[int, ...]] = (3, 7)

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size not in self._SUPPORTED_KERNELS:
            raise ValueError(
                f"kernel_size must be one of {self._SUPPORTED_KERNELS}, got {kernel_size}"
            )
        self.conv = nn.Conv3d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        mean_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        return self.gate(self.conv(torch.cat((mean_map, max_map), dim=1)))


class CBAMLite3D(nn.Module):
    """Sequential 3D CBAM-lite. Input and output shapes are identical."""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 3,
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        if residual_scale_init < 0:
            raise ValueError(
                f"residual_scale_init must be non-negative, got {residual_scale_init}"
            )
        self.channels = channels
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.channel_attention = ChannelAttention3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D(spatial_kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"CBAMLite3D expects NCDHW input, got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        residual = x
        attended = x * self.channel_attention(x)
        attended = attended * self.spatial_attention(attended)
        return residual + self.residual_scale * attended
