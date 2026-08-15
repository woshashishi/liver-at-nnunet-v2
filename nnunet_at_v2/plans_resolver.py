from __future__ import annotations

from typing import Sequence


def select_attention_stages(
    n_stages: int,
    count: int = 2,
) -> tuple[int, ...]:
    """
    Select the last `count` valid encoder skip stages.

    Stage 0 is excluded.
    The final encoder stage (bottleneck) is excluded.
    """
    valid_stages = list(range(1, n_stages - 1))

    if count < 1 or count > len(valid_stages):
        raise ValueError(
            f"Invalid attention stage count={count} "
            f"for n_stages={n_stages}"
        )

    return tuple(valid_stages[-count:])


def compute_stage_spacings(
    base_spacing: Sequence[float],
    stage_strides: Sequence[Sequence[int]],
) -> tuple[tuple[float, float, float], ...]:
    """
    Compute effective voxel spacing at every encoder stage.

    stage_strides follows nnU-Net encoder stage order.
    For example:
        base spacing
        Stage0 stride (1,1,1)
        Stage1 stride (2,2,2)
        ...
    """
    if len(base_spacing) != 3:
        raise ValueError(
            f"Only 3D spacing is supported, got {base_spacing}"
        )

    current = tuple(float(v) for v in base_spacing)
    spacings = []

    for stage_index, stride in enumerate(stage_strides):

        if len(stride) != 3:
            raise ValueError(
                f"Stage {stage_index} stride must be 3D, "
                f"got {stride}"
            )

        stride = tuple(int(v) for v in stride)

        if any(v <= 0 for v in stride):
            raise ValueError(
                f"Invalid stride at stage {stage_index}: {stride}"
            )

        current = tuple(
            current[axis] * stride[axis]
            for axis in range(3)
        )

        spacings.append(current)

    return tuple(spacings)


def resolve_spatial_kernel(
    stage_spacing: Sequence[float],
    stage_kernel: Sequence[int],
    threshold: float = 2.0,
) -> tuple[int, int, int]:
    """
    Resolve anisotropy-aware CBAM spatial kernel.

    Priority:
    1. Respect nnU-Net stage kernel axes that are already 1.
    2. Otherwise, if effective spacing anisotropy >= threshold,
       suppress the coarsest axis with kernel size 1.
    3. Otherwise use isotropic 3x3x3.
    """
    if len(stage_spacing) != 3 or len(stage_kernel) != 3:
        raise ValueError("Only 3D plans are supported")

    if threshold <= 1:
        raise ValueError(
            f"threshold must be > 1, got {threshold}"
        )

    spacing = tuple(float(v) for v in stage_spacing)
    kernel = tuple(int(v) for v in stage_kernel)

    if min(spacing) <= 0:
        raise ValueError(
            f"Spacing must be positive, got {spacing}"
        )

    result = [3, 3, 3]

    kernel_one_axes = [
        axis
        for axis, value in enumerate(kernel)
        if value == 1
    ]

    if kernel_one_axes:

        for axis in kernel_one_axes:
            result[axis] = 1

    elif max(spacing) / min(spacing) >= threshold:

        max_axis = spacing.index(max(spacing))
        result[max_axis] = 1

    return tuple(result)


def resolve_attention_layout(
    base_spacing: Sequence[float],
    stage_strides: Sequence[Sequence[int]],
    stage_kernels: Sequence[Sequence[int]],
    count: int = 2,
    anisotropy_threshold: float = 2.0,
) -> tuple[
    tuple[int, tuple[int, int, int]],
    ...
]:
    """
    Resolve Attention stage indices and spatial kernels directly
    from nnU-Net plans.

    Returns:
        (
            (stage_index, spatial_kernel),
            ...
        )
    """
    n_stages = len(stage_kernels)

    if len(stage_strides) != n_stages:
        raise ValueError(
            "stage_strides and stage_kernels must have "
            "the same number of stages"
        )

    spacings = compute_stage_spacings(
        base_spacing,
        stage_strides,
    )

    stage_indices = select_attention_stages(
        n_stages=n_stages,
        count=count,
    )

    layout = []

    for stage_index in stage_indices:

        spatial_kernel = resolve_spatial_kernel(
            stage_spacing=spacings[stage_index],
            stage_kernel=stage_kernels[stage_index],
            threshold=anisotropy_threshold,
        )

        layout.append(
            (
                stage_index,
                spatial_kernel,
            )
        )

    return tuple(layout)
