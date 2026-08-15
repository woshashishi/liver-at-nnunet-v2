from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunet_at_v2.modules.cbam3d import CBAMLite3D
from nnunet_at_v2.trainers.nnUNetTrainer_CBAMLite import (
    EncoderStageWithCBAM,
    nnUNetTrainer_CBAMLite,
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


def test_cbam_real_liver_plans():
    pm, cm = _get_configuration()

    network = nnUNetTrainer_CBAMLite.build_network_architecture(
        plans_manager=pm,
        configuration_manager=cm,
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=True,
    )

    assert tuple(network.cbam_stage_indices) == (3, 4)

    assert isinstance(
        network.encoder.stages[3],
        EncoderStageWithCBAM,
    )
    assert isinstance(
        network.encoder.stages[4],
        EncoderStageWithCBAM,
    )

    cbams = [
        m for m in network.modules()
        if isinstance(m, CBAMLite3D)
    ]

    assert len(cbams) == 2

    assert network.cbam_stage_spacings[3] == pytest.approx(
        (8.0, 6.140625, 6.140625)
    )
    assert network.cbam_stage_spacings[4] == pytest.approx(
        (16.0, 12.28125, 12.28125)
    )

    assert network.cbam_spatial_kernel_sizes == {
        3: (3, 3, 3),
        4: (3, 3, 3),
    }

    for stage_index in network.cbam_stage_indices:
        module = network.encoder.stages[stage_index].cbam

        assert tuple(
            module.spatial_attention.conv.kernel_size
        ) == network.cbam_spatial_kernel_sizes[stage_index]

        assert float(
            module.residual_scale.detach().cpu()
        ) == 0.0
