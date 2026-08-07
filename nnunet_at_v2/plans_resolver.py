from __future__ import annotations

from typing import Sequence


def select_attention_stages(
    n_stages: int,
    count: int = 2,
) -> tuple[int, ...]:
    """选择最后 count 个有效 skip 阶段，排除 Stage 0 和 bottleneck。"""
    valid_stages = list(range(1, n_stages - 1))

    if count < 1 or count > len(valid_stages):
        raise ValueError("Invalid attention stage count")

    return tuple(valid_stages[-count:])


def resolve_spatial_kernel(
    stage_spacing: Sequence[float],
    stage_kernel: Sequence[int],
    threshold: float = 2.0,
) -> tuple[int, int, int]:
    """根据 plans kernel 和阶段 spacing 解析 CBAM 空间卷积核。"""
    if len(stage_spacing) != 3 or len(stage_kernel) != 3:
        raise ValueError("Only 3D plans are supported")

    result = [3, 3, 3]
    kernel_one_axes = [
        axis for axis, value in enumerate(stage_kernel) if value == 1
    ]

    if kernel_one_axes:
        for axis in kernel_one_axes:
            result[axis] = 1
    elif max(stage_spacing) / min(stage_spacing) >= threshold:
        result[stage_spacing.index(max(stage_spacing))] = 1

    return tuple(result)

