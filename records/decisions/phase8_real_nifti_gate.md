# Phase 8 Real-NIfTI Evaluation Gate

## Status

PARTIAL PASS — real surface-metric smoke pending higher-memory environment.

## Completed

Synthetic metric tests:

- 9/9 passed
- Dice
- HD95 physical-spacing behavior
- NSD
- ASSD
- empty-mask handling
- 26-connectivity
- lesion-wise recall
- lesion FP/case

Real NIfTI baseline cross-check:

Case: liver_101
Model: A / nnUNetTrainer Fold 0

Historical tumor Dice:
0.898073141237031

Phase 8 streaming tumor Dice:
0.898073141237031

Absolute difference:
0

NIfTI affine check:
PASS

NIfTI spacing check:
PASS

## Resource limitation

Current CPU-only AutoDL container cgroup memory limit:

2147483648 bytes = 2 GiB

Real tumor ROI:

277 x 198 x 219

surface-distance==0.1 was terminated by the OS during real 3D
surface-distance calculation due to the 2 GiB memory limit.

This is treated as an infrastructure/resource limitation, not a
metric-correctness failure.

## Remaining gate

Before formal-freeze:

Run at least one real GT-positive case in a higher-memory environment
and verify successful finite output for:

- tumor HD95 mm
- tumor NSD at 3 mm
- tumor ASSD mm

Do not create the formal-freeze Git tag until this gate passes.
