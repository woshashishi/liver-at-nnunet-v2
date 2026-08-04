# CBAM-lite Stage 3/4 bundle

Extract this bundle at the root of `liver_at_project`.

## Install and discover the external trainer

```bash
chmod +x scripts/install_cbam_lite_exttrainer.sh
bash scripts/install_cbam_lite_exttrainer.sh
source /root/autodl-tmp/nnunet_project/nnunet_env.sh
```

## Tests, in order

```bash
python tests/check_cbam_lite.py module
python tests/check_cbam_lite.py build
python tests/check_cbam_lite.py full
```

Do not start training unless all three commands print `*_OK`.

## Five-epoch engineering sanity

```bash
nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer_CBAMLite_5epochs -p nnUNetPlans --npz
```
