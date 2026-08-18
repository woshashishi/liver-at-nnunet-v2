# Phase 9 Statistical Analysis Plan

## Statistical unit

The statistical unit is the individual OOF case.

Five folds are used only to generate out-of-fold predictions.

Fold-level means are NOT treated as n=5 independent observations.

## Models

A: baseline nnU-Net

B: Attention variant

C: Transformer variant

D: Hybrid Attention + Transformer variant

## Core paired comparisons

The five pre-specified comparisons are:

1. A vs B
2. A vs C
3. A vs D
4. B vs D
5. C vs D

No comparison is selected according to observed p-values.

## Analysis population

All models must have exactly one OOF record per case.

For each metric, A/B/C/D must have identical valid/missing case masks.

Model-specific missingness is treated as an evaluation error and stops
formal statistical analysis.

For primary tumor overlap/surface metrics, GT tumor-free cases are
excluded identically for all models according to the frozen Phase 8
metric protocol.

Tumor-free false-positive metrics are analyzed separately.

## Descriptive statistics

For each model and metric report:

- number of valid cases
- mean
- median
- standard deviation
- Q1
- Q3
- IQR
- 95% percentile bootstrap CI for the mean
- 95% percentile bootstrap CI for the median

Bootstrap resampling unit:

case.

Default bootstrap replicates:

10000.

Random seed:

20260818.

## Paired delta

For a comparison LEFT vs RIGHT:

delta = RIGHT - LEFT.

Report:

- mean paired delta
- median paired delta
- 95% paired bootstrap CI for mean delta
- 95% paired bootstrap CI for median delta

The sign is not automatically interpreted as improvement because some
metrics are higher-is-better while distance/error metrics are
lower-is-better.

## Hypothesis testing

Primary paired test:

two-sided Wilcoxon signed-rank test.

Zero handling:

Pratt method.

All-zero paired differences:

p = 1.0.

Alpha:

0.05.

## Multiple comparisons

Within each metric, the five pre-specified A-B, A-C, A-D, B-D and C-D
p-values are corrected using Holm-Bonferroni.

Both raw and Holm-adjusted p-values are retained.

## Effect size

Paired rank-biserial correlation is reported.

Definition:

positive effect means RIGHT > LEFT.

Negative effect means RIGHT < LEFT.

All-zero paired differences yield effect size 0.

## Formal integrity rules

Formal statistical analysis must stop if:

- duplicate case/model records exist
- any case is missing A/B/C/D coverage
- model-specific metric missingness occurs
- OOF case IDs are duplicated
- the evaluation protocol differs between models

The script and statistical plan must be frozen before formal
cross-validation results are analyzed.
