#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/root/autodl-tmp/nnunet_project
REPO="$PROJECT/code/liver_at_project"
RUN_ID=LIVER_B_F0_CBAMLITE_ZEROINIT_SANITY_003
TRAINER=nnUNetTrainer_CBAMLiteZeroInit_5epochs

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
    echo "group=B_CBAMLiteZeroInit"
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
    echo "cbam_gating=learnable_zero_init_residual"
    echo "learnable_residual_scale_init=0.0"
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
echo "CBAM_LITE_ZEROINIT_SANITY_OK"
