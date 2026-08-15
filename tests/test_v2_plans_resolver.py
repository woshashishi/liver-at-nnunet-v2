import pytest

from nnunet_at_v2.plans_resolver import (
    compute_stage_spacings,
    resolve_attention_layout,
    resolve_spatial_kernel,
    select_attention_stages,
)


def test_attention_stages():
    assert select_attention_stages(
        6
    ) == (3, 4)


def test_isotropic_kernel():
    assert resolve_spatial_kernel(
        [8.0, 6.14, 6.14],
        [3, 3, 3],
    ) == (3, 3, 3)


def test_spacing_anisotropy():
    assert resolve_spatial_kernel(
        [4.0, 1.0, 1.0],
        [3, 3, 3],
    ) == (1, 3, 3)


def test_plan_kernel_priority():
    assert resolve_spatial_kernel(
        [1.0, 1.0, 1.0],
        [1, 3, 3],
    ) == (1, 3, 3)


def test_stage_spacing_accumulation():

    spacing = compute_stage_spacings(
        base_spacing=[
            1.0,
            0.7675,
            0.7675,
        ],
        stage_strides=[
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
    )

    assert spacing[0] == pytest.approx(
        (1.0, 0.7675, 0.7675)
    )

    assert spacing[3] == pytest.approx(
        (8.0, 6.14, 6.14)
    )

    assert spacing[4] == pytest.approx(
        (16.0, 12.28, 12.28)
    )


def test_liver_like_attention_layout():

    layout = resolve_attention_layout(
        base_spacing=[
            1.0,
            0.7675,
            0.7675,
        ],
        stage_strides=[
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        stage_kernels=[
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
    )

    assert layout == (
        (3, (3, 3, 3)),
        (4, (3, 3, 3)),
    )


def test_anisotropic_attention_layout():

    layout = resolve_attention_layout(
        base_spacing=[
            4.0,
            1.0,
            1.0,
        ],
        stage_strides=[
            [1, 1, 1],
            [1, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        stage_kernels=[
            [1, 3, 3],
            [1, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
    )

    assert layout[0][0] == 3
    assert layout[1][0] == 4

    assert layout[0][1] == (
        1,
        3,
        3,
    )

    assert layout[1][1] == (
        1,
        3,
        3,
    )
