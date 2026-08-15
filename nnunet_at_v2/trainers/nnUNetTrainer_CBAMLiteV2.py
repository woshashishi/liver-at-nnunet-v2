from __future__ import annotations

import torch

from nnunet_at_v2.trainers.nnUNetTrainer_CBAMLite import (
    nnUNetTrainer_CBAMLite,
)


class nnUNetTrainer_CBAMLiteV2(
    nnUNetTrainer_CBAMLite
):
    """
    Formal v2 Attention trainer.

    Architecture:
    - plans-aware stage selection
    - anisotropy-aware spatial kernel
    - zero-initialized residual CBAM gate
    """


class nnUNetTrainer_CBAMLiteV2_5epochs(
    nnUNetTrainer_CBAMLiteV2
):
    """
    Five-epoch engineering trainer.
    Not a formal paper result.
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
