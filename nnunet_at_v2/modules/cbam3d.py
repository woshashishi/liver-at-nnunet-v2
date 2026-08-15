from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


def _normalize_spatial_kernel(
    kernel_size: int | Sequence[int],
) -> tuple[int, int, int]:
    """
    Normalize scalar/3D CBAM spatial kernel specification.
    """
    if isinstance(kernel_size, int):
        kernel = (
            kernel_size,
            kernel_size,
            kernel_size,
        )
    else:
        kernel = tuple(int(v) for v in kernel_size)

    if len(kernel) != 3:
        raise ValueError(
            f"Spatial kernel must be 3D, got {kernel}"
        )

    # Current project uses 1/3.
    # 7 remains accepted for backward compatibility.
    supported = {1, 3, 7}

    if any(v not in supported for v in kernel):
        raise ValueError(
            f"Each kernel axis must be one of "
            f"{sorted(supported)}, got {kernel}"
        )

    return kernel


class ChannelAttention3D(nn.Module):
    """
    3D channel attention using a shared 1x1x1 MLP on
    average- and max-pooled features.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be positive, got {channels}"
            )

        if reduction <= 0:
            raise ValueError(
                f"reduction must be positive, got {reduction}"
            )

        hidden = max(
            channels // reduction,
            1,
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv3d(
                channels,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                hidden,
                channels,
                kernel_size=1,
                bias=False,
            ),
        )

        self.gate = nn.Sigmoid()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        weights = self.shared_mlp(
            self.avg_pool(x)
        )

        weights = weights + self.shared_mlp(
            self.max_pool(x)
        )

        return self.gate(weights)


class SpatialAttention3D(nn.Module):
    """
    Plan-aware 3D spatial attention.

    Supports isotropic:
        (3,3,3)

    and anisotropic:
        (1,3,3)
        (3,1,3)
        (3,3,1)
    """

    def __init__(
        self,
        kernel_size: int | Sequence[int] = 3,
    ) -> None:
        super().__init__()

        kernel = _normalize_spatial_kernel(
            kernel_size
        )

        padding = tuple(
            value // 2
            for value in kernel
        )

        self.kernel_size = kernel

        self.conv = nn.Conv3d(
            2,
            1,
            kernel_size=kernel,
            padding=padding,
            bias=False,
        )

        self.gate = nn.Sigmoid()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        mean_map = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        max_map = torch.amax(
            x,
            dim=1,
            keepdim=True,
        )

        maps = torch.cat(
            (
                mean_map,
                max_map,
            ),
            dim=1,
        )

        return self.gate(
            self.conv(maps)
        )


class CBAMLite3D(nn.Module):
    """
    Sequential 3D CBAM-lite with zero-initialized
    residual integration.

    Input and output shapes are identical.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int | Sequence[int] = 3,
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()

        if residual_scale_init < 0:
            raise ValueError(
                "residual_scale_init must be "
                f"non-negative, got {residual_scale_init}"
            )

        self.channels = int(channels)

        self.residual_scale = nn.Parameter(
            torch.tensor(
                float(residual_scale_init)
            )
        )

        self.channel_attention = (
            ChannelAttention3D(
                channels=self.channels,
                reduction=reduction,
            )
        )

        self.spatial_attention = (
            SpatialAttention3D(
                spatial_kernel_size
            )
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        if x.ndim != 5:
            raise ValueError(
                "CBAMLite3D expects NCDHW input, "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, "
                f"got {x.shape[1]}"
            )

        residual = x

        # Explicit FP32 branch for stable AMP behavior.
        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):

            x_float = x.float()

            attended = (
                x_float
                * self.channel_attention(x_float)
            )

            attended = (
                attended
                * self.spatial_attention(attended)
            )

            output = (
                x_float
                + self.residual_scale.float()
                * attended
            )

        return output.to(
            dtype=residual.dtype
        )
