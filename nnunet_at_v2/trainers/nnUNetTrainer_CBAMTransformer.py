from __future__ import annotations

import torch
from torch import nn

from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)

from nnunet_at_v2.trainers.nnUNetTrainer_CBAMLite import (
    add_plan_aware_cbam,
    nnUNetTrainer_CBAMLite as _CBAMTrainer,
)
from nnunet_at_v2.trainers.nnUNetTrainer_TransformerBottleneck import (
    nnUNetTrainer_TransformerBottleneck as _TransformerTrainer,
)


class nnUNetTrainer_CBAMTransformer(
    _TransformerTrainer
):
    """
    Group D Hybrid:

    - exactly the same plan-aware CBAM implementation
      as Group B
    - exactly the same bottleneck Transformer
      implementation as Group C

    No Hybrid-specific Attention tuning.
    """

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        # Build the Transformer-only network first.
        # This guarantees C and D share the same
        # Transformer implementation.
        network = (
            _TransformerTrainer
            .build_network_architecture(
                plans_manager=plans_manager,
                configuration_manager=configuration_manager,
                num_input_channels=num_input_channels,
                num_output_channels=num_output_channels,
                enable_deep_supervision=(
                    enable_deep_supervision
                ),
            )
        )

        # Add exactly the same Attention implementation
        # and parameters used by Group B.
        network = add_plan_aware_cbam(
            network=network,
            configuration_manager=configuration_manager,
            reduction=_CBAMTrainer.cbam_reduction,
            stage_count=_CBAMTrainer.cbam_stage_count,
            anisotropy_threshold=(
                _CBAMTrainer
                .cbam_anisotropy_threshold
            ),
            residual_scale_init=(
                _CBAMTrainer
                .cbam_residual_scale_init
            ),
        )

        return network


class nnUNetTrainer_CBAMTransformer_5epochs(
    nnUNetTrainer_CBAMTransformer
):
    """
    Five-epoch Hybrid engineering sanity trainer.
    """

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
