from __future__ import annotations

import gc
import gzip
import math
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label


ROOT = Path("/root/autodl-tmp/nnunet_project")

GTROOT = (
    ROOT
    / "nnUNet_preprocessed"
    / "Dataset003_Liver"
    / "gt_segmentations"
)

PROJECT = (
    ROOT
    / "code"
    / "liver_at_project"
)

OUTROOT = PROJECT / "results_csv"
WORKROOT = ROOT / "tmp_lesion_audit"

OUTROOT.mkdir(parents=True, exist_ok=True)
WORKROOT.mkdir(parents=True, exist_ok=True)

LESION_OUT = OUTROOT / "liver_gt_lesion_size_audit.csv"
CASE_OUT = OUTROOT / "liver_gt_case_tumor_audit.csv"


# ---------------------------------------------------------
# Union-Find / Disjoint Set
# ---------------------------------------------------------

class DSU:
    def __init__(self):
        # id=0 reserved for background
        self.parent = [0]
        self.size = [0]

    def new(self, voxel_count: int) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        self.size.append(int(voxel_count))
        return idx

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra = self.find(int(a))
        rb = self.find(int(b))

        if ra == rb:
            return ra

        # union by accumulated voxel count
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.size[rb] = 0

        return ra

    def component_sizes(self):
        out = []

        for i in range(1, len(self.parent)):
            if self.find(i) == i and self.size[i] > 0:
                out.append(int(self.size[i]))

        return out


# 2D 4-connectivity + same-pixel connection between
# adjacent slices = exact 3D 6-connectivity.
STRUCTURE_2D = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)


def decompress_nii_gz(src: Path, dst: Path):
    """
    Decompress one case to temporary .nii.

    This allows nibabel mmap access instead of holding the
    entire compressed volume in RAM.
    """
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)


def streaming_component_sizes(
    nii_path: Path,
    tumor_label: int = 2,
):
    """
    Return exact 3D 6-connected component voxel counts
    without constructing a complete 3D component image.
    """

    img = nib.load(str(nii_path), mmap="r")

    if len(img.shape) != 3:
        raise RuntimeError(
            f"Expected 3D segmentation, got {img.shape}"
        )

    proxy = img.dataobj

    nx, ny, nz = img.shape

    zooms = tuple(float(x) for x in img.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(zooms))

    dsu = DSU()

    # stores global component IDs only for previous slice
    prev_global = np.zeros(
        (nx, ny),
        dtype=np.int32,
    )

    total_tumor_voxels = 0

    for z in range(nz):

        # Read only one 2D slice.
        sl = np.asanyarray(proxy[:, :, z])

        tumor = sl == tumor_label

        slice_voxels = int(tumor.sum())
        total_tumor_voxels += slice_voxels

        if slice_voxels == 0:
            prev_global.fill(0)
            continue

        local_labels, n_local = label(
            tumor,
            structure=STRUCTURE_2D,
        )

        local_counts = np.bincount(
            local_labels.ravel(),
            minlength=n_local + 1,
        )

        # Each current 2D component initially receives a new
        # global component ID.
        gid_lookup = np.zeros(
            n_local + 1,
            dtype=np.int32,
        )

        for local_id in range(1, n_local + 1):
            gid_lookup[local_id] = dsu.new(
                int(local_counts[local_id])
            )

        current_global = gid_lookup[local_labels]

        # Connections from previous z slice.
        overlap_mask = (
            (local_labels > 0)
            & (prev_global > 0)
        )

        if np.any(overlap_mask):

            current_ids = current_global[overlap_mask]
            previous_ids = prev_global[overlap_mask]

            pairs = np.column_stack(
                (current_ids, previous_ids)
            )

            pairs = np.unique(
                pairs,
                axis=0,
            )

            for current_id, previous_id in pairs:
                dsu.union(
                    int(current_id),
                    int(previous_id),
                )

        prev_global = current_global

        del sl
        del tumor
        del local_labels
        del current_global
        del gid_lookup

    sizes = dsu.component_sizes()

    return (
        sizes,
        total_tumor_voxels,
        voxel_volume_mm3,
        zooms,
    )


def main():

    files = sorted(GTROOT.glob("liver_*.nii.gz"))

    print("GT cases:", len(files), flush=True)

    # Resume support.
    existing_case_rows = []
    existing_lesion_rows = []

    if CASE_OUT.exists():
        old_cases = pd.read_csv(CASE_OUT)
        existing_case_rows = old_cases.to_dict("records")
        done_cases = set(old_cases["case"].astype(str))
    else:
        done_cases = set()

    if LESION_OUT.exists():
        old_lesions = pd.read_csv(LESION_OUT)
        existing_lesion_rows = old_lesions.to_dict("records")

    case_rows = existing_case_rows
    lesion_rows = existing_lesion_rows

    print("Already completed:", len(done_cases), flush=True)

    for i, gz_path in enumerate(files, 1):

        case = gz_path.name.replace(".nii.gz", "")

        if case in done_cases:
            print(
                f"[{i:03d}/{len(files):03d}] "
                f"{case}: SKIP",
                flush=True,
            )
            continue

        print(
            f"[{i:03d}/{len(files):03d}] "
            f"{case}: decompress",
            flush=True,
        )

        tmp_nii = WORKROOT / f"{case}.nii"

        try:
            decompress_nii_gz(
                gz_path,
                tmp_nii,
            )

            print(
                f"[{i:03d}/{len(files):03d}] "
                f"{case}: scan",
                flush=True,
            )

            (
                component_sizes,
                total_tumor_voxels,
                voxel_volume_mm3,
                spacing,
            ) = streaming_component_sizes(
                tmp_nii,
                tumor_label=2,
            )

            component_sizes = sorted(
                component_sizes,
                reverse=True,
            )

            total_volume_mm3 = (
                total_tumor_voxels
                * voxel_volume_mm3
            )

            case_rows.append(
                {
                    "case": case,
                    "n_lesions": len(component_sizes),
                    "total_tumor_voxels": total_tumor_voxels,
                    "voxel_volume_mm3": voxel_volume_mm3,
                    "total_tumor_volume_mm3": total_volume_mm3,
                    "total_tumor_volume_ml": (
                        total_volume_mm3 / 1000.0
                    ),
                    "spacing_x": spacing[0],
                    "spacing_y": spacing[1],
                    "spacing_z": spacing[2],
                    "connectivity": 6,
                }
            )

            for lesion_id, voxels in enumerate(
                component_sizes,
                1,
            ):
                volume_mm3 = (
                    voxels
                    * voxel_volume_mm3
                )

                equiv_diameter_mm = (
                    (
                        6.0
                        * volume_mm3
                        / math.pi
                    )
                    ** (1.0 / 3.0)
                    if volume_mm3 > 0
                    else 0.0
                )

                lesion_rows.append(
                    {
                        "case": case,
                        "lesion_id": lesion_id,
                        "voxels": voxels,
                        "volume_mm3": volume_mm3,
                        "volume_ml": volume_mm3 / 1000.0,
                        "equiv_diameter_mm": equiv_diameter_mm,
                        "connectivity": 6,
                    }
                )

            # Checkpoint after EVERY case.
            pd.DataFrame(case_rows).to_csv(
                CASE_OUT,
                index=False,
            )

            pd.DataFrame(lesion_rows).to_csv(
                LESION_OUT,
                index=False,
            )

            print(
                f"    lesions={len(component_sizes)} "
                f"tumor_voxels={total_tumor_voxels}",
                flush=True,
            )

        finally:
            if tmp_nii.exists():
                tmp_nii.unlink()

            gc.collect()

    cases = pd.read_csv(CASE_OUT)

    if LESION_OUT.exists():
        lesions = pd.read_csv(LESION_OUT)
    else:
        lesions = pd.DataFrame()

    print()
    print("=" * 80)
    print("FINAL GT LESION AUDIT")
    print("=" * 80)

    print("cases:", len(cases))
    print(
        "tumor-positive cases:",
        int((cases.n_lesions > 0).sum()),
    )
    print(
        "tumor-free cases:",
        int((cases.n_lesions == 0).sum()),
    )
    print("total lesions:", len(lesions))

    if len(lesions):

        print()
        print("Lesion physical volume (mm^3)")
        print(
            lesions.volume_mm3.describe(
                percentiles=[
                    .10,
                    .25,
                    1 / 3,
                    .50,
                    2 / 3,
                    .75,
                    .90,
                    .95,
                ]
            )
        )

        print()
        print("Equivalent diameter (mm)")
        print(
            lesions.equiv_diameter_mm.describe(
                percentiles=[
                    .10,
                    .25,
                    1 / 3,
                    .50,
                    2 / 3,
                    .75,
                    .90,
                    .95,
                ]
            )
        )

    print()
    print("Saved:", LESION_OUT)
    print("Saved:", CASE_OUT)


if __name__ == "__main__":
    main()
