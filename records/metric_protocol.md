# Phase 8 Metric Protocol

## Scope

Dataset003_Liver formal A/B/C/D evaluation.

This protocol is frozen before formal five-fold evaluation.

## Labels

- Background: 0
- Liver: 1
- Tumor: 2

Liver class metrics use label 1 only, matching class-wise nnU-Net evaluation.

For tumor false-positive localization, anatomical liver is defined as GT label 1 or 2.

## Volumetric tumor metrics

Primary tumor overlap metrics are evaluated on GT tumor-positive cases only.

Metrics:

- Tumor Dice
- Tumor Precision
- Tumor Sensitivity / Recall

### Empty-mask policy

GT positive / prediction positive:
standard metric definitions.

GT positive / prediction empty:

- Dice = 0
- Precision = 0
- Recall = 0
- NSD = 0
- HD95 = physical image diagonal
- ASSD = physical image diagonal
- surface_failure = true

GT empty / prediction positive:

- excluded from primary tumor overlap/surface averages
- prediction is evaluated using tumor-free FP fields
- predicted tumor voxels and predicted lesion count are retained

GT empty / prediction empty:

- excluded from primary tumor overlap/surface averages
- tumor-free FP = 0

This policy prevents complete tumor misses from silently disappearing from paired analysis.

## Surface metrics

Implementation:

surface-distance==0.1

Distances use NIfTI physical spacing and are reported in millimetres.

HD95:

area-weighted robust Hausdorff distance at the 95th percentile.

NSD:

surface Dice at a pre-specified tolerance of 3.0 mm.

ASSD:

arithmetic mean of the area-weighted directional average surface distances:

GT -> prediction

and

prediction -> GT.

The 3.0 mm NSD tolerance is a project-level pre-specified choice, not a value defined by the project guide.

## Lesion connectivity

3D connectivity = 26.

No GT or predicted lesion is removed by a minimum-size filter.

Minimum component size = 1 voxel.

## Lesion detection

A GT lesion is detected if any predicted tumor voxel overlaps the GT component.

Matching is overlap-based and many-to-many.

A predicted connected component that overlaps no GT tumor component is counted as one lesion-level false positive.

## Lesion size strata

Stratification is based only on GT 26-connected physical component volume.

Small:

V <= 213.758332536 mm^3

Medium:

213.758332536 < V <= 1414.246408246 mm^3

Large:

V > 1414.246408246 mm^3

These thresholds were derived from the GT distribution before formal cross-validation.

## Tumor FP diagnostic fields

inside_liver_fp_voxels:

predicted tumor, not GT tumor, located within GT anatomical liver.

outside_liver_fp_voxels:

predicted tumor, not GT tumor, located in GT background.

largest_outside_fp_component:

largest 26-connected outside-liver FP component in voxels.

## Required case-level fields

- liver_dice
- liver_hd95_mm
- liver_nsd_3mm
- liver_assd_mm
- tumor_dice
- tumor_hd95_mm
- tumor_nsd_3mm
- tumor_assd_mm
- tumor_precision
- tumor_recall
- lesion_recall
- lesion_fp_per_case
- small/medium/large lesion recall
- inside_liver_fp_voxels
- outside_liver_fp_voxels
- largest_outside_fp_component
- gt_tumor_voxels
- pred_tumor_voxels

## Freeze rule

After this protocol passes unit tests and the formal-freeze Git tag is created:

- NSD tolerance must not change.
- empty-mask handling must not change.
- lesion connectivity must not change.
- lesion matching must not change.
- minimum lesion filtering must not change.
- size thresholds must not change.

Any required change after formal training begins invalidates the previous formal evaluation protocol.
