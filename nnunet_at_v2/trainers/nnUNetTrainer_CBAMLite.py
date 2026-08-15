from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (
    nnUNetTrainer,
)
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)

from nnunet_at_v2.modules.cbam3d import CBAMLite3D
from nnunet_at_v2.plans_resolver import (
    compute_stage_spacings,
    resolve_attention_layout,
)


class EncoderStageWithCBAM(nn.Module):
    """
    Run an existing encoder stage and then apply
    shape-preserving CBAM-lite.
    """

    def __init__(
        self,
        stage: nn.Module,
        channels: int,
        reduction: int,
        spatial_kernel_size: int | Sequence[int],
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()

        self.stage = stage

        self.cbam = CBAMLite3D(
            channels=channels,
            reduction=reduction,
            spatial_kernel_size=spatial_kernel_size,
            residual_scale_init=residual_scale_init,
        )

    def forward(self, x):
        return self.cbam(
            self.stage(x)
        )


def add_plan_aware_cbam(
    network: nn.Module,
    configuration_manager: ConfigurationManager,
    reduction: int = 16,
    stage_count: int = 2,
    anisotropy_threshold: float = 2.0,
    residual_scale_init: float = 0.0,
) -> nn.Module:
    """
    Attach exactly the same plan-aware CBAM implementation
    to Attention-only and Hybrid networks.

    No stage index, channel count, or spatial kernel is
    hard-coded here.
    """

    if (
        not hasattr(network, "encoder")
        or not hasattr(network.encoder, "stages")
    ):
        raise TypeError(
            "Plan-aware CBAM integration expects "
            "network.encoder.stages"
        )

    output_channels = tuple(
        int(channel)
        for channel
        in network.encoder.output_channels
    )

    n_stages = len(output_channels)

    arch_kwargs = (
        configuration_manager
        .network_arch_init_kwargs
    )

    if "kernel_sizes" not in arch_kwargs:
        raise KeyError(
            "configuration_manager.network_arch_init_kwargs "
            "does not contain 'kernel_sizes'"
        )

    if "strides" not in arch_kwargs:
        raise KeyError(
            "configuration_manager.network_arch_init_kwargs "
            "does not contain 'strides'"
        )

    stage_kernels = tuple(
        tuple(int(v) for v in kernel)
        for kernel in arch_kwargs["kernel_sizes"]
    )

    stage_strides = tuple(
        tuple(int(v) for v in stride)
        for stride in arch_kwargs["strides"]
    )

    if len(stage_kernels) != n_stages:
        raise RuntimeError(
            "Plan/network stage mismatch: "
            f"{len(stage_kernels)} kernels but "
            f"{n_stages} encoder stages"
        )

    if len(stage_strides) != n_stages:
        raise RuntimeError(
            "Plan/network stage mismatch: "
            f"{len(stage_strides)} strides but "
            f"{n_stages} encoder stages"
        )

    base_spacing = tuple(
        float(v)
        for v
        in configuration_manager.spacing
    )

    layout = resolve_attention_layout(
        base_spacing=base_spacing,
        stage_strides=stage_strides,
        stage_kernels=stage_kernels,
        count=stage_count,
        anisotropy_threshold=anisotropy_threshold,
    )

    stage_spacings = compute_stage_spacings(
        base_spacing=base_spacing,
        stage_strides=stage_strides,
    )

    for stage_index, spatial_kernel in layout:

        network.encoder.stages[stage_index] = (
            EncoderStageWithCBAM(
                stage=network.encoder.stages[
                    stage_index
                ],
                channels=output_channels[
                    stage_index
                ],
                reduction=reduction,
                spatial_kernel_size=spatial_kernel,
                residual_scale_init=(
                    residual_scale_init
                ),
            )
        )

    # Store resolved metadata in the model.
    # Useful for tests/debug.json/reproducibility.
    network.cbam_stage_indices = tuple(
        stage_index
        for stage_index, _
        in layout
    )

    network.cbam_spatial_kernel_sizes = {
        int(stage_index): tuple(
            int(v)
            for v in spatial_kernel
        )
        for stage_index, spatial_kernel
        in layout
    }

    network.cbam_stage_spacings = {
        int(stage_index): tuple(
            float(v)
            for v
            in stage_spacings[stage_index]
        )
        for stage_index, _
        in layout
    }

    return network


class nnUNetTrainer_CBAMLite(nnUNetTrainer):
    """
    Group B:
    official nnU-Net plus plan-aware anisotropy-aware
    residual CBAM-lite.
    """

    cbam_reduction: int = 16
    cbam_stage_count: int = 2
    cbam_anisotropy_threshold: float = 2.0
    cbam_residual_scale_init: float = 0.0

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

        return add_plan_aware_cbam(
            network=network,
            configuration_manager=configuration_manager,
            reduction=(
                nnUNetTrainer_CBAMLite
                .cbam_reduction
            ),
            stage_count=(
                nnUNetTrainer_CBAMLite
                .cbam_stage_count
            ),
            anisotropy_threshold=(
                nnUNetTrainer_CBAMLite
                .cbam_anisotropy_threshold
            ),
            residual_scale_init=(
                nnUNetTrainer_CBAMLite
                .cbam_residual_scale_init
            ),
        )


class nnUNetTrainer_CBAMLite_5epochs(
    nnUNetTrainer_CBAMLite
):
    """
    Five-epoch engineering sanity trainer.
    Not a paper result.
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


# Legacy experiment namespaces retained so historical
# checkpoints/results remain discoverable.
class nnUNetTrainer_CBAMLiteResidual(
    nnUNetTrainer_CBAMLite
):
    pass


class nnUNetTrainer_CBAMLiteResidual_5epochs(
    nnUNetTrainer_CBAMLite_5epochs
):
    pass


class nnUNetTrainer_CBAMLiteZeroInit(
    nnUNetTrainer_CBAMLite
):
    pass


class nnUNetTrainer_CBAMLiteZeroInit_5epochs(
    nnUNetTrainer_CBAMLite_5epochs
):
    pass
