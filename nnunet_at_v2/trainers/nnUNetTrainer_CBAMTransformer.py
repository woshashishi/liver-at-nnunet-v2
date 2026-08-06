from __future__ import annotations

import torch
from torch import nn

from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)

from nnunet_at_v2.trainers.nnUNetTrainer_CBAMLite import (
    EncoderStageWithCBAM,
    nnUNetTrainer_CBAMLite as _CBAMTrainer,
)
from nnunet_at_v2.trainers.nnUNetTrainer_TransformerBottleneck import (
    nnUNetTrainer_TransformerBottleneck as _TransformerTrainer,
)


class nnUNetTrainer_CBAMTransformer(_TransformerTrainer):
    """
    Group D Hybrid:
    - CBAM-lite after encoder stages 3 and 4
    - residual bottleneck Transformer after encoder stage 5
    """

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # Reuse the frozen Transformer-only implementation first.
        network = (
            _TransformerTrainer.build_network_architecture(
                plans_manager=plans_manager,
                configuration_manager=configuration_manager,
                num_input_channels=num_input_channels,
                num_output_channels=num_output_channels,
                enable_deep_supervision=enable_deep_supervision,
            )
        )

        if not hasattr(network, "encoder") or not hasattr(network.encoder, "stages"):
            raise TypeError(
                "Hybrid integration expects network.encoder.stages"
            )

        output_channels = tuple(
            int(channel) for channel in network.encoder.output_channels
        )
        expected_channels = tuple(
            _CBAMTrainer.expected_encoder_channels
        )

        if output_channels != expected_channels:
            raise RuntimeError(
                "Unexpected encoder channels. "
                f"Expected {expected_channels}, got {output_channels}."
            )

        # Reuse exactly the same CBAM stages and parameters as Group B.
        for stage_index in _CBAMTrainer.cbam_stage_indices:
            network.encoder.stages[stage_index] = EncoderStageWithCBAM(
                stage=network.encoder.stages[stage_index],
                channels=output_channels[stage_index],
                reduction=_CBAMTrainer.cbam_reduction,
                spatial_kernel_size=(
                    _CBAMTrainer.cbam_spatial_kernel_size
                ),
            )

        network.cbam_stage_indices = tuple(
            _CBAMTrainer.cbam_stage_indices
        )
        return network


class nnUNetTrainer_CBAMTransformer_5epochs(
    nnUNetTrainer_CBAMTransformer
):
    """Five-epoch Hybrid engineering sanity trainer."""

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
