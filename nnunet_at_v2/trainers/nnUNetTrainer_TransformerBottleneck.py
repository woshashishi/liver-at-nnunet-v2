

from __future__ import annotations

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)

from nnunet_at_v2.modules import ResidualBottleneckTransformer3D


class nnUNetTrainer_TransformerBottleneck(nnUNetTrainer):
    """nnU-Net with one residual Transformer block after encoder stage 5."""

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

        stage_index = 5
        stage_channels = int(network.encoder.output_channels[stage_index])

        network.encoder.stages[stage_index] = nn.Sequential(
            network.encoder.stages[stage_index],
            ResidualBottleneckTransformer3D(
                channels=stage_channels,
                num_heads=8,
                ffn_dim=1280,
                dropout=0.0,
                residual_scale_init=0.0,
            ),
        )

        network.transformer_stage_index = stage_index
        return network


class nnUNetTrainer_TransformerBottleneck_5epochs(
    nnUNetTrainer_TransformerBottleneck
):
    """Five-epoch engineering sanity trainer."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        self.num_epochs = 5
