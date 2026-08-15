from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
)

from nnunet_at_v2.modules.transformer3d import (
    ResidualBottleneckTransformer3D,
)
from nnunet_at_v2.trainers.nnUNetTrainer_TransformerBottleneck import (
    nnUNetTrainer_TransformerBottleneck,
)


def test_transformer_module_shape_and_zero_gate():

    module = ResidualBottleneckTransformer3D(
        channels=320,
        embedding_dim=256,
        num_heads=4,
        ffn_dim=512,
        residual_scale_init=0.0,
    )

    x = torch.randn(
        1,
        320,
        2,
        3,
        3,
    )

    y = module(x)

    assert y.shape == x.shape

    assert torch.equal(
        y,
        x,
    )

    assert (
        module.embedding_dim
        == 256
    )

    assert (
        module.num_heads
        == 4
    )

    assert (
        module.ffn_dim
        == 512
    )


def test_transformer_real_liver_plans():

    root = Path(
        os.environ.get(
            "nnUNet_preprocessed",
            "/root/autodl-tmp/"
            "nnunet_project/"
            "nnUNet_preprocessed",
        )
    )

    plans_file = (
        root
        / "Dataset003_Liver"
        / "nnUNetPlans.json"
    )

    with open(
        plans_file,
        encoding="utf-8",
    ) as f:
        plans = json.load(f)

    plans_manager = (
        PlansManager(plans)
    )

    configuration_manager = (
        plans_manager
        .get_configuration(
            "3d_fullres"
        )
    )

    network = (
        nnUNetTrainer_TransformerBottleneck
        .build_network_architecture(
            plans_manager=plans_manager,
            configuration_manager=(
                configuration_manager
            ),
            num_input_channels=1,
            num_output_channels=3,
            enable_deep_supervision=True,
        )
    )

    assert (
        network.transformer_stage_index
        == 5
    )

    assert (
        network.transformer_input_channels
        == 320
    )

    assert (
        network.transformer_embedding_dim
        == 256
    )

    assert (
        network.transformer_num_heads
        == 4
    )

    assert (
        network.transformer_ffn_dim
        == 512
    )

    modules = [
        module
        for module in network.modules()
        if isinstance(
            module,
            ResidualBottleneckTransformer3D,
        )
    ]

    assert len(modules) == 1

    module = modules[0]

    assert (
        module.position_encoding.groups
        == 256
    )

    assert float(
        module.residual_scale
        .detach()
        .cpu()
    ) == 0.0
