from __future__ import annotations

import gc
import json
import os
import tempfile
from pathlib import Path

import torch

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunet_at_v2.modules.cbam3d import CBAMLite3D
from nnunet_at_v2.modules.transformer3d import (
    ResidualBottleneckTransformer3D,
)
from nnunet_at_v2.trainers.nnUNetTrainer_CBAMLite import (
    nnUNetTrainer_CBAMLite,
)
from nnunet_at_v2.trainers.nnUNetTrainer_TransformerBottleneck import (
    nnUNetTrainer_TransformerBottleneck,
)
from nnunet_at_v2.trainers.nnUNetTrainer_CBAMTransformer import (
    nnUNetTrainer_CBAMTransformer,
)


def _configuration():
    root = Path(
        os.environ.get(
            "nnUNet_preprocessed",
            "/root/autodl-tmp/nnunet_project/nnUNet_preprocessed",
        )
    )

    plans_file = (
        root
        / "Dataset003_Liver"
        / "nnUNetPlans.json"
    )

    with open(plans_file, encoding="utf-8") as f:
        plans = json.load(f)

    pm = PlansManager(plans)
    cm = pm.get_configuration("3d_fullres")

    return pm, cm


def _build(trainer):
    pm, cm = _configuration()

    return trainer.build_network_architecture(
        plans_manager=pm,
        configuration_manager=cm,
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=True,
    )


def _state_dict_equal(a, b):
    sa = a.state_dict()
    sb = b.state_dict()

    assert sa.keys() == sb.keys()

    for key in sa:
        assert torch.equal(
            sa[key],
            sb[key],
        ), f"state mismatch: {key}"


def test_cbam_zero_gate_identity():
    module = CBAMLite3D(
        channels=32,
        reduction=16,
        spatial_kernel_size=(3, 3, 3),
        residual_scale_init=0.0,
    )

    module.eval()

    x = torch.randn(
        1,
        32,
        4,
        8,
        8,
    )

    with torch.no_grad():
        y = module(x)

    assert torch.equal(x, y)


def test_cbam_anisotropic_zero_gate_identity():
    module = CBAMLite3D(
        channels=32,
        reduction=16,
        spatial_kernel_size=(1, 3, 3),
        residual_scale_init=0.0,
    )

    module.eval()

    x = torch.randn(
        1,
        32,
        3,
        8,
        8,
    )

    with torch.no_grad():
        y = module(x)

    assert torch.equal(x, y)


def test_transformer_zero_gate_identity():
    module = ResidualBottleneckTransformer3D(
        channels=320,
        embedding_dim=256,
        num_heads=4,
        ffn_dim=512,
        residual_scale_init=0.0,
    )

    module.eval()

    x = torch.randn(
        1,
        320,
        2,
        3,
        3,
    )

    with torch.no_grad():
        y = module(x)

    assert torch.equal(x, y)


def test_b_checkpoint_state_dict_roundtrip():
    model1 = _build(
        nnUNetTrainer_CBAMLite
    )

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "b.pt"

        torch.save(
            model1.state_dict(),
            checkpoint,
        )

        model2 = _build(
            nnUNetTrainer_CBAMLite
        )

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        model2.load_state_dict(
            state,
            strict=True,
        )

        _state_dict_equal(
            model1,
            model2,
        )

        assert (
            model1.cbam_stage_indices
            == model2.cbam_stage_indices
        )

        assert (
            model1.cbam_spatial_kernel_sizes
            == model2.cbam_spatial_kernel_sizes
        )

    del model1, model2
    gc.collect()


def test_c_checkpoint_state_dict_roundtrip():
    model1 = _build(
        nnUNetTrainer_TransformerBottleneck
    )

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "c.pt"

        torch.save(
            model1.state_dict(),
            checkpoint,
        )

        model2 = _build(
            nnUNetTrainer_TransformerBottleneck
        )

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        model2.load_state_dict(
            state,
            strict=True,
        )

        _state_dict_equal(
            model1,
            model2,
        )

        assert (
            model1.transformer_stage_index
            == model2.transformer_stage_index
        )

        assert (
            model1.transformer_embedding_dim
            == 256
        )

        assert (
            model1.transformer_num_heads
            == 4
        )

        assert (
            model1.transformer_ffn_dim
            == 512
        )

    del model1, model2
    gc.collect()


def test_d_checkpoint_state_dict_roundtrip():
    model1 = _build(
        nnUNetTrainer_CBAMTransformer
    )

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "d.pt"

        torch.save(
            model1.state_dict(),
            checkpoint,
        )

        model2 = _build(
            nnUNetTrainer_CBAMTransformer
        )

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        model2.load_state_dict(
            state,
            strict=True,
        )

        _state_dict_equal(
            model1,
            model2,
        )

        assert (
            model1.cbam_stage_indices
            == model2.cbam_stage_indices
        )

        assert (
            model1.transformer_stage_index
            == model2.transformer_stage_index
        )

        assert (
            model1.transformer_embedding_dim
            == 256
        )

    del model1, model2
    gc.collect()
