from __future__ import annotations

import torch

from nnunet_at_v2.trainers.nnUNetTrainer_TransformerBottleneck import (
    nnUNetTrainer_TransformerBottleneck,
)


class nnUNetTrainer_TransformerBottleneckV2(
    nnUNetTrainer_TransformerBottleneck
):
    """
    Formal v2 Transformer trainer.

    Architecture:
    - final encoder bottleneck
    - embedding cap 256
    - 4 attention heads
    - FFN ratio 2
    - dynamic depthwise 3D positional encoding
    - zero-initialized residual gate
    """


class nnUNetTrainer_TransformerBottleneckV2_5epochs(
    nnUNetTrainer_TransformerBottleneckV2
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
