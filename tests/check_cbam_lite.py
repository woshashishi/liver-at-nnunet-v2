from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunet_at_v2.modules.cbam3d import CBAMLite3D
from nnunet_at_v2.trainers.nnUNetTrainer_CBAMLite import (
    EncoderStageWithCBAM,
    nnUNetTrainer_CBAMLite,
)

PROJECT = Path("/root/autodl-tmp/nnunet_project")
PREPROCESSED = PROJECT / "nnUNet_preprocessed" / "Dataset003_Liver"


def load_managers():
    plans = json.loads((PREPROCESSED / "nnUNetPlans.json").read_text())
    dataset_json = json.loads((PREPROCESSED / "dataset.json").read_text())
    pm = PlansManager(plans)
    cm = pm.get_configuration("3d_fullres")
    n_in = determine_num_input_channels(pm, cm, dataset_json)
    n_out = pm.get_label_manager(dataset_json).num_segmentation_heads
    return pm, cm, n_in, n_out


def build_network(deep_supervision: bool):
    pm, cm, n_in, n_out = load_managers()
    return nnUNetTrainer_CBAMLite.build_network_architecture(
        pm, cm, n_in, n_out, enable_deep_supervision=deep_supervision
    )


def module_test() -> None:
    module = CBAMLite3D(256, reduction=16, spatial_kernel_size=3).cuda()
    x = torch.randn(2, 256, 8, 8, 8, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = module(x)
        loss = y.square().mean()
    loss.backward()
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("MODULE_TEST_OK", tuple(y.shape), "loss=", float(loss.detach()))


def build_test() -> None:
    network = build_network(deep_supervision=True)
    assert isinstance(network.encoder.stages[3], EncoderStageWithCBAM)
    assert isinstance(network.encoder.stages[4], EncoderStageWithCBAM)
    assert not isinstance(network.encoder.stages[5], EncoderStageWithCBAM)
    total = sum(p.numel() for p in network.parameters())
    cbam = sum(p.numel() for name, p in network.named_parameters() if ".cbam." in name)
    print("BUILD_TEST_OK")
    print("network=", type(network).__name__)
    print("encoder_channels=", list(network.encoder.output_channels))
    print("cbam_stages=[3, 4]")
    print("total_parameters=", total)
    print("cbam_parameters=", cbam)


def full_patch_test() -> None:
    os.environ["nnUNet_compile"] = "false"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    network = build_network(deep_supervision=True).cuda().train()
    x = torch.randn(1, 1, 128, 128, 128, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = network(x)
        output_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        loss = sum(output.float().square().mean() for output in output_list)
    loss.backward()
    assert all(torch.isfinite(o).all() for o in output_list)
    grads = [p.grad for n, p in network.named_parameters() if ".cbam." in n and p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "cbam_state.pt"
        torch.save(network.state_dict(), checkpoint)
        restored = build_network(deep_supervision=True).cuda()
        restored.load_state_dict(torch.load(checkpoint, map_location="cuda"))
    print("FULL_PATCH_TEST_OK")
    print("output_shapes=", [tuple(o.shape) for o in output_list])
    print("loss=", float(loss.detach()))
    print("max_cuda_memory_gib=", round(torch.cuda.max_memory_allocated() / 1024**3, 3))
    print("checkpoint_reload=OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("module", "build", "full"))
    args = parser.parse_args()
    if args.mode in {"module", "full"} and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    {"module": module_test, "build": build_test, "full": full_patch_test}[args.mode]()


if __name__ == "__main__":
    main()
