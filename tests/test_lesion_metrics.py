import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from lesion_metrics import (  # noqa: E402
    evaluate_case_arrays,
    image_diagonal_mm,
)


with open(
    ROOT / "config/phase8_metric_protocol.json",
    encoding="utf-8",
) as f:
    CFG = json.load(f)


def run(gt, pred, spacing=(1.0, 1.0, 1.0)):
    return evaluate_case_arrays(
        gt=gt,
        pred=pred,
        spacing=spacing,
        config=CFG,
        case="synthetic",
        model="test",
        fold="0",
    )


def test_exact_tumor_overlap():
    gt = np.zeros((12, 12, 12), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[3:7, 3:7, 3:7] = 2
    pred[3:7, 3:7, 3:7] = 2

    r = run(gt, pred)

    assert r["tumor_dice"] == pytest.approx(1.0)
    assert r["tumor_precision"] == pytest.approx(1.0)
    assert r["tumor_recall"] == pytest.approx(1.0)

    assert r["tumor_hd95_mm"] == pytest.approx(0.0)
    assert r["tumor_nsd_3mm"] == pytest.approx(1.0)
    assert r["tumor_assd_mm"] == pytest.approx(0.0)

    assert r["gt_lesion_count"] == 1
    assert r["detected_gt_lesion_count"] == 1
    assert r["lesion_recall"] == pytest.approx(1.0)
    assert r["lesion_fp_per_case"] == 0


def test_physical_spacing_scales_surface_distance():
    gt = np.zeros((12, 12, 12), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[2:5, 3:6, 3:6] = 2
    pred[7:10, 3:6, 3:6] = 2

    r1 = run(
        gt,
        pred,
        spacing=(1.0, 1.0, 1.0),
    )

    r2 = run(
        gt,
        pred,
        spacing=(2.0, 2.0, 2.0),
    )

    assert r1["tumor_hd95_mm"] > 0

    assert r2["tumor_hd95_mm"] == pytest.approx(
        2.0 * r1["tumor_hd95_mm"],
        rel=1e-6,
        abs=1e-6,
    )

    assert r2["tumor_assd_mm"] == pytest.approx(
        2.0 * r1["tumor_assd_mm"],
        rel=1e-6,
        abs=1e-6,
    )


def test_gt_positive_prediction_empty():
    gt = np.zeros((10, 11, 12), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[3:6, 3:6, 3:6] = 2

    spacing = (2.0, 1.0, 0.5)

    r = run(
        gt,
        pred,
        spacing=spacing,
    )

    expected_penalty = image_diagonal_mm(
        gt.shape,
        spacing,
    )

    assert r["tumor_dice"] == pytest.approx(0.0)
    assert r["tumor_precision"] == pytest.approx(0.0)
    assert r["tumor_recall"] == pytest.approx(0.0)

    assert r["tumor_nsd_3mm"] == pytest.approx(0.0)

    assert r["tumor_hd95_mm"] == pytest.approx(
        expected_penalty
    )

    assert r["tumor_assd_mm"] == pytest.approx(
        expected_penalty
    )

    assert r["tumor_surface_failure"] is True

    assert r["gt_lesion_count"] == 1
    assert r["predicted_lesion_count"] == 0
    assert r["lesion_recall"] == pytest.approx(0.0)


def test_gt_empty_prediction_positive():
    gt = np.zeros((12, 12, 12), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[2:9, 2:9, 2:9] = 1
    pred[2:9, 2:9, 2:9] = 1

    pred[10, 10, 10] = 2

    r = run(gt, pred)

    assert np.isnan(r["tumor_dice"])
    assert np.isnan(r["tumor_precision"])
    assert np.isnan(r["tumor_recall"])
    assert np.isnan(r["tumor_hd95_mm"])
    assert np.isnan(r["tumor_nsd_3mm"])

    assert r["gt_lesion_count"] == 0
    assert r["predicted_lesion_count"] == 1
    assert r["lesion_fp_per_case"] == 1

    assert r["pred_tumor_voxels"] == 1


def test_gt_empty_prediction_empty():
    gt = np.zeros((12, 12, 12), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[2:9, 2:9, 2:9] = 1
    pred[2:9, 2:9, 2:9] = 1

    r = run(gt, pred)

    assert np.isnan(r["tumor_dice"])
    assert np.isnan(r["tumor_hd95_mm"])

    assert r["gt_lesion_count"] == 0
    assert r["predicted_lesion_count"] == 0
    assert r["lesion_fp_per_case"] == 0
    assert r["pred_tumor_voxels"] == 0


def test_26_connectivity_corner_touching_is_one_lesion():
    gt = np.zeros((10, 10, 10), dtype=np.uint8)
    pred = np.zeros_like(gt)

    # Corner-connected in 3D:
    # separate with 6-connectivity,
    # one component with 26-connectivity.
    gt[2, 2, 2] = 2
    gt[3, 3, 3] = 2

    pred[2, 2, 2] = 2

    r = run(gt, pred)

    assert r["gt_lesion_count"] == 1
    assert r["detected_gt_lesion_count"] == 1
    assert r["lesion_recall"] == pytest.approx(1.0)


def test_multiple_lesions_partial_detection():
    gt = np.zeros((12, 12, 12), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[2, 2, 2] = 2
    gt[8, 8, 8] = 2

    pred[2, 2, 2] = 2

    r = run(gt, pred)

    assert r["gt_lesion_count"] == 2
    assert r["detected_gt_lesion_count"] == 1
    assert r["lesion_recall"] == pytest.approx(0.5)


def test_prediction_false_positive_component():
    gt = np.zeros((12, 12, 12), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[2, 2, 2] = 2

    pred[2, 2, 2] = 2
    pred[9, 9, 9] = 2

    r = run(gt, pred)

    assert r["predicted_lesion_count"] == 2
    assert r["lesion_fp_per_case"] == 1
    assert r["lesion_recall"] == pytest.approx(1.0)


def test_complete_separation():
    gt = np.zeros((14, 14, 14), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[2:5, 2:5, 2:5] = 2
    pred[9:12, 9:12, 9:12] = 2

    r = run(gt, pred)

    assert r["tumor_dice"] == pytest.approx(0.0)
    assert r["tumor_precision"] == pytest.approx(0.0)
    assert r["tumor_recall"] == pytest.approx(0.0)

    assert r["tumor_hd95_mm"] > 0
    assert r["lesion_recall"] == pytest.approx(0.0)
    assert r["lesion_fp_per_case"] == 1
