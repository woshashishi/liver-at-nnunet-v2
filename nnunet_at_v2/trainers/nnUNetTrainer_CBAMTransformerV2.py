from __future__ import annotations

import torch

from nnunet_at_v2.trainers.nnUNetTrainer_CBAMTransformer import (
    nnUNetTrainer_CBAMTransformer,
)


class nnUNetTrainer_CBAMTransformerV2(
    nnUNetTrainer_CBAMTransformer
):
    """
    Formal v2 Hybrid trainer.

    Exactly:
        B-v2 Attention
        +
        C-v2 Transformer
    """


class nnUNetTrainer_CBAMTransformerV2_5epochs(
    nnUNetTrainer_CBAMTransformerV2
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
