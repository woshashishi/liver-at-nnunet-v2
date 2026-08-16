from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch

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


assert torch.cuda.is_available(), "CUDA is not available"

device = torch.device("cuda:0")

root = Path(
    os.environ.get(
        "nnUNet_preprocessed",
        "/root/autodl-tmp/nnunet_project/nnUNet_preprocessed",
    )
)

plans_file = root / "Dataset003_Liver" / "nnUNetPlans.json"

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


def build(trainer):
    net = trainer.build_network_architecture(
        plans_manager=pm,
        configuration_manager=cm,
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=True,
    )
    return net.to(device)


def synthetic_loss(outputs):
    if isinstance(outputs, (tuple, list)):
        assert len(outputs) > 1, "Deep supervision outputs expected"
        return sum(
            o.float().square().mean()
            for o in outputs
        )

    return outputs.float().square().mean()


def check_outputs(outputs):
    outs = outputs if isinstance(outputs, (tuple, list)) else [outputs]

    assert len(outs) >= 1

    for i, out in enumerate(outs):
        assert torch.isfinite(out).all(), f"non-finite output {i}"

    return [tuple(x.shape) for x in outs]


def check_grads(net):
    grads = [
        p.grad
        for p in net.parameters()
        if p.requires_grad and p.grad is not None
    ]

    assert len(grads) > 0, "No gradients produced"

    for grad in grads:
        assert torch.isfinite(grad).all(), "non-finite gradient"

    return len(grads)


print("GPU:", torch.cuda.get_device_name(0))
print("=" * 80)

for name, trainer in MODELS.items():

    print(f"\nMODEL {name}")

    torch.cuda.empty_cache()
    gc.collect()

    net = build(trainer)
    net.train()

    # 64^3 is deliberate:
    # 5 downsamplings -> 2^3 bottleneck, not 1^3.
    x = torch.randn(
        1,
        1,
        64,
        64,
        64,
        device=device,
    )

    # --------------------------------------------------
    # FP32 forward + backward
    # --------------------------------------------------
    net.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats(device)

    outputs = net(x)

    shapes = check_outputs(outputs)

    loss = synthetic_loss(outputs)

    assert torch.isfinite(loss), "FP32 loss is non-finite"

    loss.backward()

    grad_count = check_grads(net)

    fp32_peak = (
        torch.cuda.max_memory_allocated(device)
        / 1024**2
    )

    print("FP32:")
    print("  output shapes:", shapes)
    print("  loss:", float(loss.detach()))
    print("  grad tensors:", grad_count)
    print(f"  peak allocated: {fp32_peak:.1f} MiB")

    del outputs, loss

    # --------------------------------------------------
    # AMP forward + backward
    # --------------------------------------------------
    net.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats(device)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        outputs = net(x)
        amp_loss = synthetic_loss(outputs)

    assert torch.isfinite(amp_loss), "AMP loss is non-finite"

    amp_loss.backward()

    amp_grad_count = check_grads(net)

    amp_peak = (
        torch.cuda.max_memory_allocated(device)
        / 1024**2
    )

    print("AMP:")
    print("  loss:", float(amp_loss.detach()))
    print("  grad tensors:", amp_grad_count)
    print(f"  peak allocated: {amp_peak:.1f} MiB")

    print(f"{name}_GPU_FORWARD_BACKWARD_PASS")

    del outputs
    del amp_loss
    del x
    del net

    gc.collect()
    torch.cuda.empty_cache()


print("\n" + "=" * 80)
print("V2_GPU_FORWARD_BACKWARD_AMP_PASS")
