from __future__ import annotations

import json
from pathlib import Path

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunet_at_v2.modules.cbam3d import CBAMLite3D
from nnunet_at_v2.modules.transformer3d import (
    ResidualBottleneckTransformer3D,
)
from nnunet_at_v2.trainers.nnUNetTrainer_CBAMLite import (
    EncoderStageWithCBAM,
)
from nnunet_at_v2.trainers.nnUNetTrainer_CBAMTransformer import (
    nnUNetTrainer_CBAMTransformer,
)

preprocessed = Path(
    "/root/autodl-tmp/nnunet_project/"
    "nnUNet_preprocessed/Dataset003_Liver"
)

with open(preprocessed / "nnUNetPlans.json", encoding="utf-8") as f:
    plans = json.load(f)

plans_manager = PlansManager(plans)
configuration_manager = plans_manager.get_configuration("3d_fullres")

network = nnUNetTrainer_CBAMTransformer.build_network_architecture(
    plans_manager=plans_manager,
    configuration_manager=configuration_manager,
    num_input_channels=1,
    num_output_channels=3,
    enable_deep_supervision=True,
)

cbam_modules = [
    module for module in network.modules()
    if isinstance(module, CBAMLite3D)
]
transformer_modules = [
    module for module in network.modules()
    if isinstance(module, ResidualBottleneckTransformer3D)
]

assert isinstance(network.encoder.stages[3], EncoderStageWithCBAM)
assert isinstance(network.encoder.stages[4], EncoderStageWithCBAM)
assert len(cbam_modules) == 2
assert len(transformer_modules) == 1
assert tuple(network.cbam_stage_indices) == (3, 4)
assert network.transformer_stage_index == 5

cbam_gates = [
    float(module.residual_scale.detach().cpu())
    for module in cbam_modules
]
transformer_gates = [
    float(module.residual_scale.detach().cpu())
    for module in transformer_modules
]

assert cbam_gates == [0.0, 0.0]
assert transformer_gates == [0.0]

parameters = sum(parameter.numel() for parameter in network.parameters())

print("HYBRID_CPU_BUILD_PASS")
print("encoder_channels:", tuple(network.encoder.output_channels))
print("cbam_stage_indices:", network.cbam_stage_indices)
print("transformer_stage_index:", network.transformer_stage_index)
print("cbam_modules:", len(cbam_modules))
print("transformer_modules:", len(transformer_modules))
print("cbam_gates:", cbam_gates)
print("transformer_gates:", transformer_gates)
print("total_parameters:", parameters)
