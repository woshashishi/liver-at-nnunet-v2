#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=/root/autodl-tmp/nnunet_project
REPO="$PROJECT/code/liver_at_project"
ENV_FILE="$PROJECT/nnunet_env.sh"
EXPORT_LINE='export nnUNet_extTrainer="/root/autodl-tmp/nnunet_project/code/liver_at_project"'
cd "$REPO"
python -m pip install -e .
if ! grep -Fqx "$EXPORT_LINE" "$ENV_FILE"; then
    printf '\n%s\n' "$EXPORT_LINE" >> "$ENV_FILE"
fi
source "$ENV_FILE"
python - <<'PY'
import nnunet_at_v2
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
for name in ("nnUNetTrainer_CBAMLite", "nnUNetTrainer_CBAMLite_5epochs"):
    cls = recursive_find_trainer_class_by_name(name)
    if cls is None:
        raise RuntimeError(f"Trainer discovery failed: {name}")
    print(name, "->", cls)
print("research package ->", nnunet_at_v2.__file__)
print("INSTALL_AND_DISCOVERY_OK")
PY
