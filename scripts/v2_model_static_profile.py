from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunet_at_v2.trainers.nnUNetTrainer_CBAMLiteV2 import (
    nnUNetTrainer_CBAMLiteV2,
)
from nnunet_at_v2.trainers.nnUNetTrainer_TransformerBottleneckV2 import (
    nnUNetTrainer_TransformerBottleneckV2,
)
from nnunet_at_v2.trainers.nnUNetTrainer_CBAMTransformerV2 import (
    nnUNetTrainer_CBAMTransformerV2,
)


ROOT = Path(
    os.environ.get(
        "nnUNet_preprocessed",
        "/root/autodl-tmp/nnunet_project/nnUNet_preprocessed",
    )
)

plans_file = ROOT / "Dataset003_Liver" / "nnUNetPlans.json"

with open(plans_file, encoding="utf-8") as f:
    plans = json.load(f)

pm = PlansManager(plans)
cm = pm.get_configuration("3d_fullres")

MODELS = {
    "A": nnUNetTrainer,
    "B_v2": nnUNetTrainer_CBAMLiteV2,
    "C_v2": nnUNetTrainer_TransformerBottleneckV2,
    "D_v2": nnUNetTrainer_CBAMTransformerV2,
}

rows = []

for name, trainer in MODELS.items():

    net = trainer.build_network_architecture(
        plans_manager=pm,
        configuration_manager=cm,
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=True,
    )

    total = sum(
        p.numel()
        for p in net.parameters()
    )

    trainable = sum(
        p.numel()
        for p in net.parameters()
        if p.requires_grad
    )

    row = {
        "model": name,
        "total_params": total,
        "trainable_params": trainable,
    }

    if hasattr(net, "cbam_stage_indices"):
        row["cbam_stages"] = str(
            tuple(net.cbam_stage_indices)
        )
        row["cbam_kernels"] = str(
            net.cbam_spatial_kernel_sizes
        )

    if hasattr(net, "transformer_stage_index"):
        row["transformer_stage"] = (
            net.transformer_stage_index
        )
        row["embedding_dim"] = (
            net.transformer_embedding_dim
        )
        row["num_heads"] = (
            net.transformer_num_heads
        )
        row["ffn_dim"] = (
            net.transformer_ffn_dim
        )

    rows.append(row)


df = pd.DataFrame(rows)

baseline = int(
    df.loc[
        df["model"] == "A",
        "total_params",
    ].iloc[0]
)

df["extra_params_vs_A"] = (
    df["total_params"] - baseline
)

df["extra_params_pct_vs_A"] = (
    100.0
    * df["extra_params_vs_A"]
    / baseline
)

out = "results_csv/v2_model_static_profile.csv"

df.to_csv(
    out,
    index=False,
)

print(df.to_string(index=False))
print("\nSaved:", out)
