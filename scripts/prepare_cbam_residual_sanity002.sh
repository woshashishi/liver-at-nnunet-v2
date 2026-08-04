#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/autodl-tmp/nnunet_project
REPO="$PROJECT/code/liver_at_project"
TRAINER_FILE="$REPO/nnunet_at_v2/trainers/nnUNetTrainer_CBAMLite.py"
CONFIG_FILE="$REPO/configs/cbam_lite_stage34.yaml"
RUN_SCRIPT="$REPO/scripts/run_cbam_lite_residual_sanity_002.sh"
MANIFEST_001="$PROJECT/records/runs/LIVER_B_F0_CBAMLITE_SANITY_001_manifest.txt"

cd "$REPO"

if pgrep -af nnUNetv2_train >/dev/null; then
    echo "检测到 nnUNetv2_train 正在运行，拒绝准备新实验。"
    exit 10
fi

python - <<'PY'
from pathlib import Path

path = Path("nnunet_at_v2/trainers/nnUNetTrainer_CBAMLite.py")
text = path.read_text(encoding="utf-8")

addition = """

class nnUNetTrainer_CBAMLiteResidual(nnUNetTrainer_CBAMLite):
    \"\"\"Residual CBAM-lite formal trainer with a distinct result namespace.\"\"\"


class nnUNetTrainer_CBAMLiteResidual_5epochs(nnUNetTrainer_CBAMLite_5epochs):
    \"\"\"Five-epoch residual CBAM-lite sanity trainer.\"\"\"
"""

if "class nnUNetTrainer_CBAMLiteResidual(" not in text:
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
    print("ADDED residual trainer aliases")
else:
    print("Residual trainer aliases already exist")
PY

python - <<'PY'
from pathlib import Path

path = Path("configs/cbam_lite_stage34.yaml")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "sanity_trainer: nnUNetTrainer_CBAMLite_5epochs",
    "sanity_trainer: nnUNetTrainer_CBAMLiteResidual_5epochs",
)
text = text.replace(
    "formal_trainer: nnUNetTrainer_CBAMLite",
    "formal_trainer: nnUNetTrainer_CBAMLiteResidual",
)
path.write_text(text, encoding="utf-8")
print("UPDATED:", path)
PY

python -m py_compile "$TRAINER_FILE"
source "$PROJECT/nnunet_env.sh"

python - <<'PY'
import inspect
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

for name in (
    "nnUNetTrainer_CBAMLiteResidual",
    "nnUNetTrainer_CBAMLiteResidual_5epochs",
):
    cls = recursive_find_trainer_class_by_name(name)
    if cls is None:
        raise RuntimeError(f"Trainer discovery failed: {name}")
    print(name, "->", cls)
    print("signature:", inspect.signature(cls.__init__))

print("RESIDUAL_TRAINER_DISCOVERY_OK")
PY

if [[ -f "$MANIFEST_001" ]] && ! grep -q '^quality_gate=' "$MANIFEST_001"; then
    {
        echo "quality_gate=REJECTED"
        echo "quality_reason=Mean_Dice_0.004573_vs_baseline_sanity_0.447887"
        echo "effective_trainer_commit=f77f2cb"
        echo "provenance_note=trainer_worktree_was_committed_after_launch"
        echo "superseded_by=LIVER_B_F0_CBAMLITE_RESIDUAL_SANITY_002"
    } >> "$MANIFEST_001"
fi

cat > "$RUN_SCRIPT" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/autodl-tmp/nnunet_project
REPO="$PROJECT/code/liver_at_project"
RUN_ID=LIVER_B_F0_CBAMLITE_RESIDUAL_SANITY_002
TRAINER=nnUNetTrainer_CBAMLiteResidual_5epochs

LOG="$PROJECT/records/runs/${RUN_ID}.log"
MANIFEST="$PROJECT/records/runs/${RUN_ID}_manifest.txt"
STATUS="$PROJECT/records/runs/${RUN_ID}.status"
RESULT="$PROJECT/nnUNet_results/Dataset003_Liver/${TRAINER}__nnUNetPlans__3d_fullres/fold_0"
VAL="$RESULT/validation"

mkdir -p "$PROJECT/records/runs"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
source "$PROJECT/nnunet_env.sh"
cd "$REPO"

if pgrep -af "nnUNetv2_train 3 3d_fullres 0.*${TRAINER}" >/dev/null; then
    echo "已有同名 sanity 任务运行，拒绝重复启动。"
    exit 10
fi

if [[ -e "$RESULT" ]]; then
    echo "结果目录已存在，拒绝覆盖：$RESULT"
    exit 11
fi

{
    echo "run_id=$RUN_ID"
    echo "start_time=$(date -Is)"
    echo "dataset=Dataset003_Liver"
    echo "group=B_CBAMLiteResidual"
    echo "fold=0"
    echo "configuration=3d_fullres"
    echo "plans=nnUNetPlans"
    echo "trainer=$TRAINER"
    echo "patch_size=128x128x128"
    echo "batch_size=2"
    echo "epochs=5"
    echo "cbam_stages=3,4"
    echo "cbam_reduction=16"
    echo "cbam_spatial_kernel=3x3x3"
    echo "cbam_gating=residual_sequential_multiplicative"
    echo "residual_scale=1.0"
    echo "research_commit=$(git rev-parse HEAD)"
    echo "official_commit=$(git -C "$PROJECT/code/nnUNet_official" rev-parse HEAD)"
    echo "command=nnUNetv2_train 3 3d_fullres 0 -tr $TRAINER -p nnUNetPlans --npz"
} | tee "$MANIFEST"

set +e
PYTHONUNBUFFERED=1 nnUNetv2_train     3 3d_fullres 0     -tr "$TRAINER"     -p nnUNetPlans     --npz     2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

{
    echo "end_time=$(date -Is)"
    echo "exit_code=$RC"
} | tee -a "$LOG" "$MANIFEST"

if [[ "$RC" -ne 0 ]]; then
    echo "FAILED" > "$STATUS"
    exit "$RC"
fi

for file in     "$RESULT/checkpoint_final.pth"     "$RESULT/checkpoint_best.pth"     "$RESULT/debug.json"     "$VAL/summary.json"
do
    [[ -s "$file" ]] || {
        echo "缺少结果文件：$file"
        echo "FAILED_OUTPUT_CHECK" > "$STATUS"
        exit 20
    }
done

NII="$(find "$VAL" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)"
NPZ="$(find "$VAL" -maxdepth 1 -type f -name '*.npz' | wc -l)"

{
    echo "validation_nifti=$NII"
    echo "validation_npz=$NPZ"
    echo "result_path=$RESULT"
    echo "result_size=$(du -sh "$RESULT" | awk '{print $1}')"
} | tee -a "$MANIFEST"

if [[ "$NII" -ne 27 || "$NPZ" -ne 27 ]]; then
    echo "FAILED_VALIDATION_COUNT" > "$STATUS"
    exit 21
fi

echo "SUCCESS" > "$STATUS"
echo "CBAM_LITE_RESIDUAL_SANITY_OK"
EOF

chmod +x "$RUN_SCRIPT"

echo
echo "=== Git status ==="
git status -sb
echo
echo "PREPARE_RESIDUAL_SANITY_002_OK"
echo "Run script: $RUN_SCRIPT"
