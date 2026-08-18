# Phase 8 Lesion Definition Freeze

## Dataset

Dataset003_Liver

Tumor label: 2

## Connectivity decision

Formal lesion-level evaluation uses **3D 26-connectivity**.

The GT-only sensitivity audit showed:

- 6-connectivity components: 1037
- 26-connectivity components: 908
- 1-voxel components: 103 -> 18
- <=5-voxel components: 140 -> 35
- Tumor voxel counts were identical for all 131 cases.

The connectivity rule was selected using GT only, before formal
cross-validation results.

## Minimum lesion size

No GT lesion is removed according to size.

minimum_gt_lesion_voxels = 1

No predicted lesion is removed according to size.

minimum_predicted_lesion_voxels = 1

This avoids introducing an arbitrary post-hoc lesion-size exclusion.

## Lesion matching

A GT lesion is considered detected when at least one voxel overlaps
with any predicted tumor component.

A predicted component with no overlap with any GT tumor component is
counted as a lesion-level false positive.

## Size stratification

Size groups are defined from all GT 26-connected components using
physical volume tertiles.

Small:
V <= 213.758332536 mm^3

Medium:
213.758332536 < V <= 1414.246408246 mm^3

Large:
V > 1414.246408246 mm^3

Equivalent-sphere diameter reference only:

P33 = 7.418362 mm
P67 = 13.926449 mm

The physical-volume thresholds, not equivalent diameter, are the
formal stratification criteria.

## Freeze rule

Connectivity, minimum lesion size, matching rule, and lesion-size
thresholds must not be changed after formal cross-validation begins.
