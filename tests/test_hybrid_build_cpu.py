from __future__ import annotations

import gc
import json
import os
from pathlib import Path

from torch import nn

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunet_at_v2.modules.cbam3d import CBAMLite3D
from nnunet_at_v2.modules.transformer3d import (
    ResidualBottleneckTransformer3D,
)
from nnunet_at_v2.trainers.nnUNetTrainer_CBAMLite import (
    EncoderStageWithCBAM,
    nnUNetTrainer_CBAMLite,
)
from nnunet_at_v2.trainers.nnUNetTrainer_TransformerBottleneck import (
    nnUNetTrainer_TransformerBottleneck,
)
from nnunet_at_v2.trainers.nnUNetTrainer_CBAMTransformer import (
    nnUNetTrainer_CBAMTransformer,
)


def _get_configuration():
    root = os.environ.get("nnUNet_preprocessed")
    assert root, "nnUNet_preprocessed environment variable is not set"

    plans_file = (
        Path(root)
        / "Dataset003_Liver"
        / "nnUNetPlans.json"
    )

    with open(plans_file, encoding="utf-8") as f:
        plans = json.load(f)

    pm = PlansManager(plans)
    cm = pm.get_configuration("3d_fullres")

    return pm, cm


def _build(trainer):
    pm, cm = _get_configuration()

    return trainer.build_network_architecture(
        plans_manager=pm,
        configuration_manager=cm,
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=True,
    )


def _cbam_signature(network):
    assert tuple(network.cbam_stage_indices) == (3, 4)

    modules = [
        m for m in network.modules()
        if isinstance(m, CBAMLite3D)
    ]

    assert len(modules) == 2

    signature = []

    for stage_index in network.cbam_stage_indices:
        wrapper = network.encoder.stages[stage_index]

        assert isinstance(
            wrapper,
            EncoderStageWithCBAM,
        )

        cbam = wrapper.cbam

        signature.append(
            (
                int(stage_index),
                int(cbam.channels),
                tuple(
                    cbam.spatial_attention.conv.kernel_size
                ),
                float(
                    cbam.residual_scale.detach().cpu()
                ),
            )
        )

    return (
        tuple(signature),
        dict(network.cbam_spatial_kernel_sizes),
        dict(network.cbam_stage_spacings),
    )


def _transformer_signature(network):
    modules = [
        m for m in network.modules()
        if isinstance(
            m,
            ResidualBottleneckTransformer3D,
        )
    ]

    assert len(modules) == 1

    module = modules[0]

    assert isinstance(
        module.input_projection,
        nn.Conv3d,
    )

    assert isinstance(
        module.output_projection,
        nn.Conv3d,
    )

    return (
        int(network.transformer_stage_index),
        int(network.transformer_input_channels),
        int(network.transformer_embedding_dim),
        int(network.transformer_num_heads),
        int(network.transformer_ffn_dim),
        int(module.position_encoding.groups),
        tuple(module.input_projection.weight.shape),
        tuple(module.output_projection.weight.shape),
        float(
            module.residual_scale.detach().cpu()
        ),
    )


def test_hybrid_reuses_exact_b_and_c_components():
    # Group B
    network_b = _build(
        nnUNetTrainer_CBAMLite
    )

    b_cbam = _cbam_signature(
        network_b
    )

    del network_b
    gc.collect()

    # Group C
    network_c = _build(
        nnUNetTrainer_TransformerBottleneck
    )

    c_transformer = _transformer_signature(
        network_c
    )

    del network_c
    gc.collect()

    # Group D
    network_d = _build(
        nnUNetTrainer_CBAMTransformer
    )

    d_cbam = _cbam_signature(
        network_d
    )

    d_transformer = _transformer_signature(
        network_d
    )

    # Fairness constraints
    assert d_cbam == b_cbam
    assert d_transformer == c_transformer

    # Dataset003 expected Transformer-v2 configuration
    assert d_transformer[0] == 5
    assert d_transformer[1] == 320
    assert d_transformer[2] == 256
    assert d_transformer[3] == 4
    assert d_transformer[4] == 512
    assert d_transformer[5] == 256

    # 256 <- 320 and 320 <- 256 projections
    assert d_transformer[6] == (256, 320, 1, 1, 1)
    assert d_transformer[7] == (320, 256, 1, 1, 1)

    # zero-init residual gate
    assert d_transformer[8] == 0.0
