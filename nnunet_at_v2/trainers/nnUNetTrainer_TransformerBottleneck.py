from __future__ import annotations

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (
    nnUNetTrainer,
)
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)

from nnunet_at_v2.modules import (
    ResidualBottleneckTransformer3D,
)


def add_bottleneck_transformer(
    network: nn.Module,
    embedding_cap: int = 256,
    num_heads: int = 4,
    mlp_ratio: int = 2,
    dropout: float = 0.0,
    residual_scale_init: float = 0.0,
) -> nn.Module:
    """
    Add the shared v2 bottleneck Transformer.

    Used identically by Group C and Group D.
    """

    if (
        not hasattr(network, "encoder")
        or not hasattr(
            network.encoder,
            "stages",
        )
    ):
        raise TypeError(
            "Transformer integration expects "
            "network.encoder.stages"
        )

    output_channels = tuple(
        int(channel)
        for channel
        in network.encoder.output_channels
    )

    if len(output_channels) < 2:
        raise RuntimeError(
            "Unexpected encoder stage count"
        )

    # Plans/network aware:
    # bottleneck is always the final encoder stage.
    stage_index = (
        len(output_channels) - 1
    )

    stage_channels = (
        output_channels[
            stage_index
        ]
    )

    embedding_dim = min(
        stage_channels,
        embedding_cap,
    )

    ffn_dim = (
        embedding_dim
        * mlp_ratio
    )

    transformer = (
        ResidualBottleneckTransformer3D(
            channels=stage_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            residual_scale_init=(
                residual_scale_init
            ),
        )
    )

    network.encoder.stages[
        stage_index
    ] = nn.Sequential(
        network.encoder.stages[
            stage_index
        ],
        transformer,
    )

    # Reproducibility metadata.
    network.transformer_stage_index = (
        stage_index
    )

    network.transformer_input_channels = (
        stage_channels
    )

    network.transformer_embedding_dim = (
        embedding_dim
    )

    network.transformer_num_heads = (
        num_heads
    )

    network.transformer_ffn_dim = (
        ffn_dim
    )

    return network


class nnUNetTrainer_TransformerBottleneck(
    nnUNetTrainer
):
    """
    Group C:
    official nnU-Net plus lightweight
    residual bottleneck Transformer v2.
    """

    transformer_embedding_cap: int = 256
    transformer_num_heads: int = 4
    transformer_mlp_ratio: int = 2
    transformer_dropout: float = 0.0
    transformer_residual_scale_init: float = 0.0

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        network = (
            nnUNetTrainer
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

        return add_bottleneck_transformer(
            network=network,
            embedding_cap=(
                nnUNetTrainer_TransformerBottleneck
                .transformer_embedding_cap
            ),
            num_heads=(
                nnUNetTrainer_TransformerBottleneck
                .transformer_num_heads
            ),
            mlp_ratio=(
                nnUNetTrainer_TransformerBottleneck
                .transformer_mlp_ratio
            ),
            dropout=(
                nnUNetTrainer_TransformerBottleneck
                .transformer_dropout
            ),
            residual_scale_init=(
                nnUNetTrainer_TransformerBottleneck
                .transformer_residual_scale_init
            ),
        )


class nnUNetTrainer_TransformerBottleneck_5epochs(
    nnUNetTrainer_TransformerBottleneck
):
    """
    Five-epoch engineering sanity trainer.
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
