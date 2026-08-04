from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from nnunet_at_v2.modules.cbam3d import CBAMLite3D


class EncoderStageWithCBAM(nn.Module):
    """Run an existing encoder stage and then apply shape-preserving CBAM-lite."""

    def __init__(
        self,
        stage: nn.Module,
        channels: int,
        reduction: int,
        spatial_kernel_size: int,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.cbam = CBAMLite3D(channels, reduction, spatial_kernel_size)

    def forward(self, x):
        return self.cbam(self.stage(x))


class nnUNetTrainer_CBAMLite(nnUNetTrainer):
    """Group B: official nnU-Net plus 3D CBAM-lite after encoder stages 3 and 4."""

    cbam_stage_indices: Sequence[int] = (3, 4)
    cbam_reduction: int = 16
    cbam_spatial_kernel_size: int = 3
    expected_encoder_channels: Sequence[int] = (32, 64, 128, 256, 320, 320)

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        network = nnUNetTrainer.build_network_architecture(
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
        )

        if not hasattr(network, "encoder") or not hasattr(network.encoder, "stages"):
            raise TypeError("CBAM-lite integration expects network.encoder.stages")

        output_channels = tuple(int(c) for c in network.encoder.output_channels)
        expected = tuple(nnUNetTrainer_CBAMLite.expected_encoder_channels)
        if output_channels != expected:
            raise RuntimeError(
                f"Unexpected encoder channels. Expected {expected}, got {output_channels}."
            )

        for stage_index in nnUNetTrainer_CBAMLite.cbam_stage_indices:
            network.encoder.stages[stage_index] = EncoderStageWithCBAM(
                stage=network.encoder.stages[stage_index],
                channels=output_channels[stage_index],
                reduction=nnUNetTrainer_CBAMLite.cbam_reduction,
                spatial_kernel_size=nnUNetTrainer_CBAMLite.cbam_spatial_kernel_size,
            )
        return network


class nnUNetTrainer_CBAMLite_5epochs(nnUNetTrainer_CBAMLite):
    """Five-epoch engineering sanity trainer. Not a paper result."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            unpack_dataset=unpack_dataset,
            device=device,
        )
        self.num_epochs = 5
