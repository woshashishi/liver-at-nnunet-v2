import json
from pathlib import Path

import torch

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunet_at_v2.modules.cbam3d import CBAMLite3D
from nnunet_at_v2.modules.transformer3d import ResidualBottleneckTransformer3D
from nnunet_at_v2.trainers.nnUNetTrainer_CBAMTransformer import (
    nnUNetTrainer_CBAMTransformer,
)

assert torch.cuda.is_available(), "CUDA不可用"

torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)
torch.set_num_threads(1)

plans_path = Path(
    "/root/autodl-tmp/nnunet_project/"
    "nnUNet_preprocessed/Dataset003_Liver/nnUNetPlans.json"
)

with plans_path.open(encoding="utf-8") as f:
    plans = json.load(f)

plans_manager = PlansManager(plans)
configuration_manager = plans_manager.get_configuration("3d_fullres")

network = nnUNetTrainer_CBAMTransformer.build_network_architecture(
    plans_manager=plans_manager,
    configuration_manager=configuration_manager,
    num_input_channels=1,
    num_output_channels=3,
    enable_deep_supervision=True,
).cuda().train()

cbams = [m for m in network.modules() if isinstance(m, CBAMLite3D)]
transformers = [
    m for m in network.modules()
    if isinstance(m, ResidualBottleneckTransformer3D)
]

assert len(cbams) == 2, len(cbams)
assert len(transformers) == 1, len(transformers)

gates = [
    cbams[0].residual_scale,
    cbams[1].residual_scale,
    transformers[0].residual_scale,
]

before = [float(g.detach().cpu()) for g in gates]

optimizer = torch.optim.SGD(network.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()

x = torch.randn(
    2, 1, 128, 128, 128,
    device="cuda",
    dtype=torch.float32,
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
optimizer.zero_grad(set_to_none=True)

with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
    enabled=True,
):
    outputs = network(x)
    outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    loss = sum(out.float().square().mean() for out in outputs)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)

gradients = [
    None if gate.grad is None else float(gate.grad.detach().cpu())
    for gate in gates
]

assert all(g is not None for g in gradients), gradients
assert all(torch.isfinite(torch.tensor(g)) for g in gradients), gradients
assert all(abs(g) > 0 for g in gradients), gradients

scaler.step(optimizer)
scaler.update()
torch.cuda.synchronize()

after = [float(g.detach().cpu()) for g in gates]

assert all(a != b for a, b in zip(before, after)), (before, after)

peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
parameters = sum(p.numel() for p in network.parameters())

print("HYBRID_GPU_STEP_PASS")
print("output_shapes:", [tuple(out.shape) for out in outputs])
print("loss:", float(loss.detach().cpu()))
print("gate_before:", before)
print("gate_gradients:", gradients)
print("gate_after:", after)
print("total_parameters:", parameters)
print("peak_allocated_GiB:", round(peak_allocated, 3))
print("peak_reserved_GiB:", round(peak_reserved, 3))
