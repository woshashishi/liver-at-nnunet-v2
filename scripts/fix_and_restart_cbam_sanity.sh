#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/autodl-tmp/nnunet_project
REPO="$PROJECT/code/liver_at_project"
TRAINER_FILE="$REPO/nnunet_at_v2/trainers/nnUNetTrainer_CBAMLite.py"
RUN_ID=LIVER_B_F0_CBAMLITE_SANITY_001
RESULT="$PROJECT/nnUNet_results/Dataset003_Liver/nnUNetTrainer_CBAMLite_5epochs__nnUNetPlans__3d_fullres/fold_0"

cd "$REPO"

python - <<'PY'
from pathlib import Path

path = Path("nnunet_at_v2/trainers/nnUNetTrainer_CBAMLite.py")
text = path.read_text(encoding="utf-8")

if "import torch\n" not in text:
    text = text.replace("from torch import nn\n", "import torch\nfrom torch import nn\n")

marker = "class nnUNetTrainer_CBAMLite_5epochs"
if marker not in text:
    raise RuntimeError(f"Cannot find {marker}")

head = text.split(marker, 1)[0]
replacement = '''class nnUNetTrainer_CBAMLite_5epochs(nnUNetTrainer_CBAMLite):
    """Five-epoch engineering sanity trainer. Not a paper result."""

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
'''

path.write_text(head + replacement, encoding="utf-8")
print("PATCH_OK:", path)
PY

python -m py_compile "$TRAINER_FILE"

python - <<'PY'
import inspect
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunet_at_v2.trainers.nnUNetTrainer_CBAMLite import (
    nnUNetTrainer_CBAMLite_5epochs,
)

base_sig = inspect.signature(nnUNetTrainer.__init__)
child_sig = inspect.signature(nnUNetTrainer_CBAMLite_5epochs.__init__)

print("base_signature =", base_sig)
print("child_signature=", child_sig)

base_names = [p for p in base_sig.parameters if p != "self"]
child_names = [p for p in child_sig.parameters if p != "self"]
if base_names != child_names:
    raise RuntimeError(
        f"Constructor parameters still differ: base={base_names}, child={child_names}"
    )

print("SIGNATURE_MATCH_OK")
PY

rm -f \
  "$PROJECT/records/runs/${RUN_ID}.log" \
  "$PROJECT/records/runs/${RUN_ID}_manifest.txt" \
  "$PROJECT/records/runs/${RUN_ID}.status"

rm -rf "$RESULT"
screen -wipe >/dev/null 2>&1 || true

source "$PROJECT/nnunet_env.sh"

screen -dmS liver_b_cbam_sanity \
  bash "$REPO/scripts/run_cbam_lite_sanity.sh"

sleep 8

echo "=== screen ==="
screen -ls || true

echo
echo "=== latest log ==="
tail -40 "$PROJECT/records/runs/${RUN_ID}.log" 2>/dev/null || true
