from nnunet_at_v2.plans_resolver import (
    resolve_spatial_kernel,
    select_attention_stages,
)


def test_attention_stages():
    assert select_attention_stages(6) == (3, 4)


def test_isotropic_kernel():
    assert resolve_spatial_kernel(
        [8.0, 6.14, 6.14], [3, 3, 3]
    ) == (3, 3, 3)


def test_spacing_anisotropy():
    assert resolve_spatial_kernel(
        [4.0, 1.0, 1.0], [3, 3, 3]
    ) == (1, 3, 3)


def test_plan_kernel_priority():
    assert resolve_spatial_kernel(
        [1.0, 1.0, 1.0], [1, 3, 3]
    ) == (1, 3, 3)

