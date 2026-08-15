from __future__ import annotations

import json
import os
from pathlib import Path

from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
)

from nnunet_at_v2.modules.cbam3d import (
    CBAMLite3D,
)
from nnunet_at_v2.trainers.nnUNetTrainer_CBAMLite import (
    EncoderStageWithCBAM,
    nnUNetTrainer_CBAMLite,
)


ROOT = Path(
    os.environ.get(
        "nnUNet_preprocessed",
        "/root/autodl-tmp/nnunet_project/"
        "nnUNet_preprocessed",
    )
)

plans_file = (
    ROOT
    / "Dataset003_Liver"
    / "nnUNetPlans.json"
)

with open(
    plans_file,
    encoding="utf-8",
) as f:
    plans = json.load(f)

plans_manager = PlansManager(plans)

configuration_manager = (
    plans_manager.get_configuration(
        "3d_fullres"
    )
)

network = (
    nnUNetTrainer_CBAMLite
    .build_network_architecture(
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=True,
    )
)


assert tuple(
    network.cbam_stage_indices
) == (3, 4)


assert isinstance(
    network.encoder.stages[3],
    EncoderStageWithCBAM,
)

assert isinstance(
    network.encoder.stages[4],
    EncoderStageWithCBAM,
)


cbams = [
    module
    for module in network.modules()
    if isinstance(
        module,
        CBAMLite3D,
    )
]

assert len(cbams) == 2


resolved_kernels = {
    stage_index:
    tuple(
        network
        .encoder
        .stages[stage_index]
        .cbam
        .spatial_attention
        .conv
        .kernel_size
    )
    for stage_index
    in network.cbam_stage_indices
}


assert resolved_kernels == (
    network.cbam_spatial_kernel_sizes
)


for module in cbams:

    assert float(
        module
        .residual_scale
        .detach()
        .cpu()
    ) == 0.0


print("CBAM_PLAN_AWARE_CPU_PASS")

print(
    "encoder_channels:",
    tuple(
        network.encoder.output_channels
    ),
)

print(
    "spacing:",
    tuple(
        configuration_manager.spacing
    ),
)

print(
    "cbam_stage_indices:",
    network.cbam_stage_indices,
)

print(
    "cbam_stage_spacings:",
    network.cbam_stage_spacings,
)

print(
    "cbam_spatial_kernel_sizes:",
    network.cbam_spatial_kernel_sizes,
)

print(
    "resolved_conv_kernels:",
    resolved_kernels,
)
