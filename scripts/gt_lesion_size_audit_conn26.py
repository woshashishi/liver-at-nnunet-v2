from __future__ import annotations

import gc
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label

from gt_lesion_size_audit_streaming import (
    DSU,
    decompress_nii_gz,
)

ROOT = Path("/root/autodl-tmp/nnunet_project")

GTROOT = (
    ROOT / "nnUNet_preprocessed"
    / "Dataset003_Liver"
    / "gt_segmentations"
)

PROJECT = ROOT / "code/liver_at_project"
OUTROOT = PROJECT / "results_csv"
WORKROOT = ROOT / "tmp_lesion_audit_conn26"

OUTROOT.mkdir(parents=True, exist_ok=True)
WORKROOT.mkdir(parents=True, exist_ok=True)

LESION_OUT = OUTROOT / "liver_gt_lesion_size_audit_conn26.csv"
CASE_OUT = OUTROOT / "liver_gt_case_tumor_audit_conn26.csv"

# Within one axial slice:
# 8-connectivity.
#
# Together with connections to all 3x3 neighboring pixels
# on the previous slice => exact 3D 26-connectivity.
STRUCTURE_2D = np.ones((3, 3), dtype=np.uint8)


def aligned_slices(n, delta):
    if delta == 0:
        return slice(0, n), slice(0, n)

    if delta > 0:
        # current index = previous index + delta
        return slice(delta, n), slice(0, n - delta)

    # delta < 0
    return slice(0, n + delta), slice(-delta, n)


def scan_conn26(nii_path: Path, tumor_label=2):
    img = nib.load(str(nii_path), mmap="r")

    if len(img.shape) != 3:
        raise RuntimeError(f"Expected 3D volume, got {img.shape}")

    proxy = img.dataobj
    nx, ny, nz = img.shape

    spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(spacing))

    dsu = DSU()

    prev_global = np.zeros(
        (nx, ny),
        dtype=np.int32
    )

    total_tumor_voxels = 0

    for z in range(nz):
        sl = np.asanyarray(proxy[:, :, z])

        tumor = (sl == tumor_label)

        nvox = int(tumor.sum())
        total_tumor_voxels += nvox

        if nvox == 0:
            prev_global.fill(0)
            continue

        local_labels, n_local = label(
            tumor,
            structure=STRUCTURE_2D,
        )

        counts = np.bincount(
            local_labels.ravel(),
            minlength=n_local + 1,
        )

        gid_lookup = np.zeros(
            n_local + 1,
            dtype=np.int32,
        )

        for local_id in range(1, n_local + 1):
            gid_lookup[local_id] = dsu.new(
                int(counts[local_id])
            )

        current_global = gid_lookup[local_labels]

        # 26-connectivity between adjacent z slices:
        # dx, dy ∈ {-1, 0, 1}
        for dx in (-1, 0, 1):
            cx, px = aligned_slices(nx, dx)

            for dy in (-1, 0, 1):
                cy, py = aligned_slices(ny, dy)

                cur = current_global[cx, cy]
                prv = prev_global[px, py]

                mask = (cur > 0) & (prv > 0)

                if not np.any(mask):
                    continue

                pairs = np.column_stack((
                    cur[mask],
                    prv[mask],
                ))

                pairs = np.unique(pairs, axis=0)

                for a, b in pairs:
                    dsu.union(int(a), int(b))

        prev_global = current_global

        del sl, tumor
        del local_labels, gid_lookup
        gc.collect()

    sizes = dsu.component_sizes()

    return (
        sorted(sizes, reverse=True),
        total_tumor_voxels,
        voxel_volume_mm3,
        spacing,
    )


def main():
    files = sorted(GTROOT.glob("liver_*.nii.gz"))

    if CASE_OUT.exists():
        old_cases = pd.read_csv(CASE_OUT)
        case_rows = old_cases.to_dict("records")
        done = set(old_cases.case.astype(str))
    else:
        case_rows = []
        done = set()

    if LESION_OUT.exists():
        lesion_rows = pd.read_csv(
            LESION_OUT
        ).to_dict("records")
    else:
        lesion_rows = []

    print("GT cases:", len(files), flush=True)
    print("Already completed:", len(done), flush=True)

    for i, src in enumerate(files, 1):
        case = src.name.replace(".nii.gz", "")

        if case in done:
            print(
                f"[{i:03d}/{len(files):03d}] {case}: SKIP",
                flush=True,
            )
            continue

        tmp = WORKROOT / f"{case}.nii"

        print(
            f"[{i:03d}/{len(files):03d}] {case}: decompress",
            flush=True,
        )

        try:
            decompress_nii_gz(src, tmp)

            print(
                f"[{i:03d}/{len(files):03d}] {case}: scan",
                flush=True,
            )

            sizes, total_vox, voxel_mm3, spacing = scan_conn26(tmp)

            total_mm3 = total_vox * voxel_mm3

            case_rows.append({
                "case": case,
                "n_lesions": len(sizes),
                "total_tumor_voxels": total_vox,
                "voxel_volume_mm3": voxel_mm3,
                "total_tumor_volume_mm3": total_mm3,
                "total_tumor_volume_ml": total_mm3 / 1000.0,
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
                "connectivity": 26,
            })

            for lesion_id, voxels in enumerate(sizes, 1):
                volume = voxels * voxel_mm3

                equiv_diameter = (
                    (6.0 * volume / math.pi) ** (1.0 / 3.0)
                    if volume > 0 else 0.0
                )

                lesion_rows.append({
                    "case": case,
                    "lesion_id": lesion_id,
                    "voxels": voxels,
                    "volume_mm3": volume,
                    "volume_ml": volume / 1000.0,
                    "equiv_diameter_mm": equiv_diameter,
                    "connectivity": 26,
                })

            pd.DataFrame(case_rows).to_csv(
                CASE_OUT,
                index=False,
            )

            pd.DataFrame(lesion_rows).to_csv(
                LESION_OUT,
                index=False,
            )

            print(
                f"    lesions={len(sizes)} "
                f"tumor_voxels={total_vox}",
                flush=True,
            )

        finally:
            if tmp.exists():
                tmp.unlink()

            gc.collect()

    print("\nSaved:", LESION_OUT)
    print("Saved:", CASE_OUT)


if __name__ == "__main__":
    main()
