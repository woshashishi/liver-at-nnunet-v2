#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/root/autodl-tmp/nnunet_project/code/liver_at_project
CBAM_FILE="$REPO/nnunet_at_v2/modules/cbam3d.py"
CONFIG_FILE="$REPO/configs/cbam_lite_stage34.yaml"

cd "$REPO"

if pgrep -af nnUNetv2_train >/dev/null; then
    echo "检测到 nnUNetv2_train 正在运行，拒绝修改代码。"
    exit 10
fi

cp -f "$CBAM_FILE" "${CBAM_FILE}.before_residual"

python - <<'PY'
from pathlib import Path

path = Path("nnunet_at_v2/modules/cbam3d.py")
text = path.read_text(encoding="utf-8")

old_init = '''    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.channel_attention = ChannelAttention3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D(spatial_kernel_size)
'''

new_init = '''    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 3,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if residual_scale < 0:
            raise ValueError(
                f"residual_scale must be non-negative, got {residual_scale}"
            )
        self.channels = channels
        self.residual_scale = float(residual_scale)
        self.channel_attention = ChannelAttention3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D(spatial_kernel_size)
'''

old_forward = '''        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
'''

new_forward = '''        residual = x
        attended = x * self.channel_attention(x)
        attended = attended * self.spatial_attention(attended)
        return residual + self.residual_scale * attended
'''

if old_init not in text:
    raise RuntimeError("未找到预期的 CBAMLite3D.__init__ 代码，未修改文件")
if old_forward not in text:
    raise RuntimeError("未找到预期的 CBAMLite3D.forward 代码，未修改文件")

text = text.replace(old_init, new_init, 1)
text = text.replace(old_forward, new_forward, 1)
path.write_text(text, encoding="utf-8")
print("PATCHED:", path)
PY

python - <<'PY'
from pathlib import Path

path = Path("configs/cbam_lite_stage34.yaml")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "  gating: sequential_multiplicative\n",
    "  gating: residual_sequential_multiplicative\n"
    "  residual_scale: 1.0\n",
)
path.write_text(text, encoding="utf-8")
print("UPDATED:", path)
PY

python -m py_compile   nnunet_at_v2/modules/cbam3d.py   nnunet_at_v2/trainers/nnUNetTrainer_CBAMLite.py

source /root/autodl-tmp/nnunet_project/nnunet_env.sh

python tests/check_cbam_lite.py module
python tests/check_cbam_lite.py build
python tests/check_cbam_lite.py full

echo
echo "=== Git diff ==="
git diff --check
git diff --   nnunet_at_v2/modules/cbam3d.py   configs/cbam_lite_stage34.yaml

echo
echo "RESIDUAL_CBAM_PATCH_AND_TEST_OK"
