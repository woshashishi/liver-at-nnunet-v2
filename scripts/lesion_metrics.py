from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label
from surface_distance import metrics as sd_metrics


CONN26 = np.ones((3, 3, 3), dtype=np.uint8)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def image_diagonal_mm(shape, spacing):
    shape = np.asarray(shape, dtype=float)
    spacing = np.asarray(spacing, dtype=float)

    return float(
        np.sqrt(
            np.sum(
                ((shape - 1.0) * spacing) ** 2
            )
        )
    )


def binary_overlap_metrics(gt, pred):
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    g = int(gt.sum())
    p = int(pred.sum())
    tp = int(np.logical_and(gt, pred).sum())

    # GT empty is excluded from primary tumor metrics.
    if g == 0:
        return {
            "dice": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "tp": tp,
            "gt_voxels": g,
            "pred_voxels": p,
        }

    dice = (
        2.0 * tp / (g + p)
        if (g + p) > 0
        else 0.0
    )

    # A complete miss is explicitly penalized.
    precision = (
        tp / p
        if p > 0
        else 0.0
    )

    recall = tp / g

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "gt_voxels": g,
        "pred_voxels": p,
    }


def surface_metrics(
    gt,
    pred,
    spacing,
    tolerance_mm,
):
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    gt_any = bool(gt.any())
    pred_any = bool(pred.any())

    diagonal = image_diagonal_mm(
        gt.shape,
        spacing,
    )

    if not gt_any and not pred_any:
        return {
            "hd95_mm": 0.0,
            "nsd": 1.0,
            "assd_mm": 0.0,
            "surface_failure": False,
        }

    if gt_any != pred_any:
        return {
            "hd95_mm": diagonal,
            "nsd": 0.0,
            "assd_mm": diagonal,
            "surface_failure": True,
        }

    # Memory-safe evaluation:
    # Cropping to the joint foreground bounding box preserves
    # surface distances while reducing memory use on large CT volumes.
    crop = union_bbox_slices(gt, pred)

    gt_eval = gt[crop]
    pred_eval = pred[crop]

    sd = sd_metrics.compute_surface_distances(
        gt_eval,
        pred_eval,
        spacing_mm=spacing,
    )

    hd95 = float(
        sd_metrics.compute_robust_hausdorff(
            sd,
            95.0,
        )
    )

    nsd = float(
        sd_metrics.compute_surface_dice_at_tolerance(
            sd,
            tolerance_mm=tolerance_mm,
        )
    )

    directional_assd = (
        sd_metrics.compute_average_surface_distance(
            sd
        )
    )

    assd = float(
        np.mean(
            np.asarray(
                directional_assd,
                dtype=float,
            )
        )
    )

    return {
        "hd95_mm": hd95,
        "nsd": nsd,
        "assd_mm": assd,
        "surface_failure": False,
    }


def union_bbox_slices(a, b):
    union = np.logical_or(a, b)

    if not union.any():
        return None

    slices = []

    for axis in range(3):
        other_axes = tuple(
            i for i in range(3)
            if i != axis
        )

        projection = np.any(
            union,
            axis=other_axes,
        )

        idx = np.flatnonzero(projection)

        slices.append(
            slice(
                int(idx[0]),
                int(idx[-1]) + 1,
            )
        )

    return tuple(slices)


def lesion_statistics(
    gt_tumor,
    pred_tumor,
    spacing,
    config,
):
    crop = union_bbox_slices(
        gt_tumor,
        pred_tumor,
    )

    if crop is None:
        return {
            "gt_lesion_count": 0,
            "detected_gt_lesion_count": 0,
            "predicted_lesion_count": 0,
            "lesion_recall": np.nan,
            "lesion_fp_per_case": 0,

            "small_gt_lesions": 0,
            "small_detected_lesions": 0,
            "small_lesion_recall": np.nan,

            "medium_gt_lesions": 0,
            "medium_detected_lesions": 0,
            "medium_lesion_recall": np.nan,

            "large_gt_lesions": 0,
            "large_detected_lesions": 0,
            "large_lesion_recall": np.nan,
        }

    gt = np.asarray(
        gt_tumor[crop],
        dtype=bool,
    )

    pred = np.asarray(
        pred_tumor[crop],
        dtype=bool,
    )

    gt_cc, n_gt = label(
        gt,
        structure=CONN26,
    )

    pred_cc, n_pred = label(
        pred,
        structure=CONN26,
    )

    detected_gt = set(
        int(x)
        for x in np.unique(
            gt_cc[pred]
        )
        if int(x) != 0
    )

    matched_pred = set(
        int(x)
        for x in np.unique(
            pred_cc[gt]
        )
        if int(x) != 0
    )

    lesion_recall = (
        len(detected_gt) / n_gt
        if n_gt > 0
        else np.nan
    )

    lesion_fp = (
        int(n_pred) - len(matched_pred)
    )

    sizes = config["lesion_level"][
        "size_stratification"
    ]

    p33 = float(
        sizes["small"][
            "volume_mm3_max_inclusive"
        ]
    )

    p67 = float(
        sizes["medium"][
            "volume_mm3_max_inclusive"
        ]
    )

    voxel_volume = float(
        np.prod(
            np.asarray(
                spacing,
                dtype=float,
            )
        )
    )

    counts = np.bincount(
        gt_cc.ravel(),
        minlength=int(n_gt) + 1,
    )

    group_total = {
        "small": 0,
        "medium": 0,
        "large": 0,
    }

    group_detected = {
        "small": 0,
        "medium": 0,
        "large": 0,
    }

    for lesion_id in range(
        1,
        int(n_gt) + 1,
    ):
        volume_mm3 = (
            int(counts[lesion_id])
            * voxel_volume
        )

        if volume_mm3 <= p33:
            group = "small"
        elif volume_mm3 <= p67:
            group = "medium"
        else:
            group = "large"

        group_total[group] += 1

        if lesion_id in detected_gt:
            group_detected[group] += 1

    out = {
        "gt_lesion_count": int(n_gt),
        "detected_gt_lesion_count":
            int(len(detected_gt)),

        "predicted_lesion_count":
            int(n_pred),

        "lesion_recall":
            float(lesion_recall)
            if n_gt > 0
            else np.nan,

        "lesion_fp_per_case":
            int(lesion_fp),
    }

    for group in (
        "small",
        "medium",
        "large",
    ):
        total = group_total[group]
        detected = group_detected[group]

        out[f"{group}_gt_lesions"] = int(total)
        out[f"{group}_detected_lesions"] = int(
            detected
        )

        out[f"{group}_lesion_recall"] = (
            float(detected / total)
            if total > 0
            else np.nan
        )

    return out


def largest_component_voxels(mask):
    mask = np.asarray(mask, dtype=bool)

    if not mask.any():
        return 0

    crop = union_bbox_slices(
        mask,
        np.zeros_like(mask, dtype=bool),
    )

    cc, n = label(
        mask[crop],
        structure=CONN26,
    )

    if n == 0:
        return 0

    counts = np.bincount(
        cc.ravel()
    )

    return int(
        counts[1:].max()
    )


def evaluate_case_arrays(
    gt,
    pred,
    spacing,
    config,
    case="unknown",
    model="unknown",
    fold="unknown",
):
    gt = np.asarray(gt)
    pred = np.asarray(pred)

    if gt.shape != pred.shape:
        raise ValueError(
            f"shape mismatch: "
            f"GT={gt.shape}, Pred={pred.shape}"
        )

    if gt.ndim != 3:
        raise ValueError(
            f"expected 3D arrays, got {gt.ndim}D"
        )

    spacing = tuple(
        float(x)
        for x in spacing
    )

    if len(spacing) != 3:
        raise ValueError(
            f"expected 3D spacing, got {spacing}"
        )

    labels = config["labels"]

    liver_label = int(labels["liver"])
    tumor_label = int(labels["tumor"])

    gt_liver = (gt == liver_label)
    pred_liver = (pred == liver_label)

    gt_tumor = (gt == tumor_label)
    pred_tumor = (pred == tumor_label)

    gt_liver_anatomy = np.logical_or(
        gt == liver_label,
        gt == tumor_label,
    )

    liver_overlap = binary_overlap_metrics(
        gt_liver,
        pred_liver,
    )

    tol = float(
        config["surface_metrics"][
            "nsd_tolerance_mm"
        ]
    )

    liver_surface = surface_metrics(
        gt_liver,
        pred_liver,
        spacing,
        tol,
    )

    tumor_overlap = binary_overlap_metrics(
        gt_tumor,
        pred_tumor,
    )

    gt_tumor_positive = bool(
        gt_tumor.any()
    )

    pred_tumor_positive = bool(
        pred_tumor.any()
    )

    if gt_tumor_positive:
        tumor_surface = surface_metrics(
            gt_tumor,
            pred_tumor,
            spacing,
            tol,
        )
    else:
        # Tumor-free GT cases are excluded from
        # primary tumor surface metrics.
        tumor_surface = {
            "hd95_mm": np.nan,
            "nsd": np.nan,
            "assd_mm": np.nan,
            "surface_failure": False,
        }

    lesion = lesion_statistics(
        gt_tumor,
        pred_tumor,
        spacing,
        config,
    )

    tumor_fp = np.logical_and(
        pred_tumor,
        np.logical_not(gt_tumor),
    )

    inside_fp = np.logical_and(
        tumor_fp,
        gt_liver_anatomy,
    )

    outside_fp = np.logical_and(
        tumor_fp,
        np.logical_not(
            gt_liver_anatomy
        ),
    )

    row = {
        "case": case,
        "fold": fold,
        "model": model,

        "shape_x": int(gt.shape[0]),
        "shape_y": int(gt.shape[1]),
        "shape_z": int(gt.shape[2]),

        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],

        "gt_tumor_positive":
            gt_tumor_positive,

        "pred_tumor_positive":
            pred_tumor_positive,

        "liver_dice":
            liver_overlap["dice"],

        "liver_hd95_mm":
            liver_surface["hd95_mm"],

        "liver_nsd_3mm":
            liver_surface["nsd"],

        "liver_assd_mm":
            liver_surface["assd_mm"],

        "liver_surface_failure":
            liver_surface[
                "surface_failure"
            ],

        "tumor_dice":
            tumor_overlap["dice"],

        "tumor_precision":
            tumor_overlap["precision"],

        "tumor_recall":
            tumor_overlap["recall"],

        "tumor_hd95_mm":
            tumor_surface["hd95_mm"],

        "tumor_nsd_3mm":
            tumor_surface["nsd"],

        "tumor_assd_mm":
            tumor_surface["assd_mm"],

        "tumor_surface_failure":
            tumor_surface[
                "surface_failure"
            ],

        "gt_tumor_voxels":
            int(gt_tumor.sum()),

        "pred_tumor_voxels":
            int(pred_tumor.sum()),

        "inside_liver_fp_voxels":
            int(inside_fp.sum()),

        "outside_liver_fp_voxels":
            int(outside_fp.sum()),

        "largest_outside_fp_component":
            largest_component_voxels(
                outside_fp
            ),
    }

    row.update(lesion)

    return row


def evaluate_nifti(
    gt_path,
    pred_path,
    config,
    case,
    model,
    fold,
):
    gt_img = nib.load(
        str(gt_path),
        mmap="r",
    )

    pred_img = nib.load(
        str(pred_path),
        mmap="r",
    )

    if gt_img.shape != pred_img.shape:
        raise ValueError(
            f"NIfTI shape mismatch: "
            f"{gt_img.shape} vs "
            f"{pred_img.shape}"
        )

    gt_spacing = tuple(
        float(x)
        for x in gt_img.header.get_zooms()[:3]
    )

    pred_spacing = tuple(
        float(x)
        for x in pred_img.header.get_zooms()[:3]
    )

    if not np.allclose(
        gt_spacing,
        pred_spacing,
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError(
            f"spacing mismatch: "
            f"{gt_spacing} vs "
            f"{pred_spacing}"
        )

    if not np.allclose(
        gt_img.affine,
        pred_img.affine,
        rtol=0,
        atol=1e-4,
    ):
        raise ValueError(
            "GT/pred affine mismatch"
        )

    gt = np.asanyarray(
        gt_img.dataobj
    )

    pred = np.asanyarray(
        pred_img.dataobj
    )

    return evaluate_case_arrays(
        gt=gt,
        pred=pred,
        spacing=gt_spacing,
        config=config,
        case=case,
        model=model,
        fold=fold,
    )


def write_row(row, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new = pd.DataFrame([row])

    if out_csv.exists():
        old = pd.read_csv(out_csv)

        df = pd.concat(
            [old, new],
            ignore_index=True,
        )

        key = [
            "case",
            "fold",
            "model",
        ]

        df = df.drop_duplicates(
            subset=key,
            keep="last",
        )
    else:
        df = new

    df.to_csv(
        out_csv,
        index=False,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--gt",
        required=True,
    )

    p.add_argument(
        "--pred",
        required=True,
    )

    p.add_argument(
        "--case",
        required=True,
    )

    p.add_argument(
        "--model",
        required=True,
    )

    p.add_argument(
        "--fold",
        required=True,
    )

    p.add_argument(
        "--config",
        default=(
            "config/"
            "phase8_metric_protocol.json"
        ),
    )

    p.add_argument(
        "--out-csv",
        default=None,
    )

    args = p.parse_args()

    cfg = load_config(
        args.config
    )

    row = evaluate_nifti(
        gt_path=args.gt,
        pred_path=args.pred,
        config=cfg,
        case=args.case,
        model=args.model,
        fold=args.fold,
    )

    print(
        pd.Series(row).to_string()
    )

    if args.out_csv:
        write_row(
            row,
            args.out_csv,
        )

        print(
            "\nSaved:",
            args.out_csv,
        )


if __name__ == "__main__":
    main()
