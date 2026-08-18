from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


CORE_COMPARISONS = [
    ("A", "B"),
    ("A", "C"),
    ("A", "D"),
    ("B", "D"),
    ("C", "D"),
]


def validate_long_dataframe(
    df: pd.DataFrame,
    models=("A", "B", "C", "D"),
):
    required = {"case", "model"}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"missing required columns: {sorted(missing)}"
        )

    dup = df.duplicated(
        subset=["case", "model"],
        keep=False,
    )

    if dup.any():
        rows = df.loc[
            dup,
            ["case", "model"],
        ].head(20)

        raise ValueError(
            "duplicate case/model rows found:\n"
            + rows.to_string(index=False)
        )

    cases = sorted(df["case"].unique())

    counts = (
        df.groupby("case")["model"]
        .apply(set)
    )

    expected = set(models)

    bad = {
        case: sorted(expected - present)
        for case, present in counts.items()
        if present != expected
    }

    if bad:
        first = list(bad.items())[:20]

        raise ValueError(
            "incomplete A/B/C/D coverage: "
            + repr(first)
        )

    return cases


def bootstrap_ci(
    x,
    statistic="mean",
    n_bootstrap=10000,
    seed=20260818,
):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    if len(x) == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(x),
        size=(n_bootstrap, len(x)),
    )

    samples = x[idx]

    if statistic == "mean":
        values = samples.mean(axis=1)

    elif statistic == "median":
        values = np.median(
            samples,
            axis=1,
        )

    else:
        raise ValueError(
            f"unsupported statistic: {statistic}"
        )

    lo, hi = np.percentile(
        values,
        [2.5, 97.5],
    )

    return float(lo), float(hi)


def describe_vector(
    x,
    n_bootstrap=10000,
    seed=20260818,
):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        raise ValueError(
            "cannot describe an empty vector"
        )

    q25, q75 = np.percentile(
        x,
        [25, 75],
    )

    mean_lo, mean_hi = bootstrap_ci(
        x,
        statistic="mean",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    median_lo, median_hi = bootstrap_ci(
        x,
        statistic="median",
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "sd": (
            float(np.std(x, ddof=1))
            if len(x) > 1
            else 0.0
        ),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),

        "mean_ci95_low": mean_lo,
        "mean_ci95_high": mean_hi,

        "median_ci95_low": median_lo,
        "median_ci95_high": median_hi,
    }


def paired_rank_biserial(
    left,
    right,
):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)

    # Positive effect means right > left.
    d = right - left

    nonzero = d != 0

    if not np.any(nonzero):
        return 0.0

    d = d[nonzero]

    ranks = rankdata(
        np.abs(d),
        method="average",
    )

    pos = float(
        ranks[d > 0].sum()
    )

    neg = float(
        ranks[d < 0].sum()
    )

    denom = pos + neg

    if denom == 0:
        return 0.0

    return float(
        (pos - neg) / denom
    )


def paired_wilcoxon(
    left,
    right,
):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)

    d = right - left

    if np.all(d == 0):
        return 1.0

    result = wilcoxon(
        d,
        zero_method="pratt",
        correction=False,
        alternative="two-sided",
        method="approx",
    )

    return float(result.pvalue)


def paired_delta_summary(
    left,
    right,
    n_bootstrap=10000,
    seed=20260818,
):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)

    if left.shape != right.shape:
        raise ValueError(
            "paired vectors must have identical shape"
        )

    d = right - left

    mean_lo, mean_hi = bootstrap_ci(
        d,
        statistic="mean",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    median_lo, median_hi = bootstrap_ci(
        d,
        statistic="median",
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )

    return {
        "delta_definition": "right_minus_left",

        "mean_delta":
            float(np.mean(d)),

        "median_delta":
            float(np.median(d)),

        "mean_delta_ci95_low":
            mean_lo,

        "mean_delta_ci95_high":
            mean_hi,

        "median_delta_ci95_low":
            median_lo,

        "median_delta_ci95_high":
            median_hi,

        "rank_biserial":
            paired_rank_biserial(
                left,
                right,
            ),

        "p_raw":
            paired_wilcoxon(
                left,
                right,
            ),
    }


def holm_adjust(
    pvalues,
):
    p = np.asarray(
        pvalues,
        dtype=float,
    )

    m = len(p)

    if m == 0:
        return np.array([], dtype=float)

    order = np.argsort(p)

    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )

    running = 0.0

    for i, idx in enumerate(order):
        candidate = (
            (m - i)
            * p[idx]
        )

        running = max(
            running,
            candidate,
        )

        adjusted_sorted[i] = min(
            running,
            1.0,
        )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    for i, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[i]

    return adjusted


def metric_population(
    df,
    metric,
    models=("A", "B", "C", "D"),
):
    if metric not in df.columns:
        raise ValueError(
            f"metric column not found: {metric}"
        )

    pivot = df.pivot(
        index="case",
        columns="model",
        values=metric,
    )

    pivot = pivot.loc[
        :,
        list(models),
    ]

    missing = pivot.isna()

    # Formal protocol requires identical missingness across models.
    # Example: GT-empty tumor cases may be NaN for ALL A/B/C/D.
    inconsistent = (
        missing.nunique(axis=1) > 1
    )

    if inconsistent.any():
        bad = pivot.loc[
            inconsistent
        ].head(20)

        raise ValueError(
            f"model-specific missingness for metric {metric}:\n"
            + bad.to_string()
        )

    valid = ~missing.any(axis=1)

    return pivot.loc[valid]


def analyze_metric(
    df,
    metric,
    models=("A", "B", "C", "D"),
    comparisons=CORE_COMPARISONS,
    n_bootstrap=10000,
    seed=20260818,
):
    pop = metric_population(
        df,
        metric,
        models=models,
    )

    if len(pop) == 0:
        raise ValueError(
            f"no valid cases for metric {metric}"
        )

    descriptive = []

    for i, model in enumerate(models):
        desc = describe_vector(
            pop[model].to_numpy(),
            n_bootstrap=n_bootstrap,
            seed=seed + 100 * i,
        )

        descriptive.append({
            "metric": metric,
            "model": model,
            "n_cases": len(pop),
            **desc,
        })

    pairwise = []

    for i, (left, right) in enumerate(comparisons):
        summary = paired_delta_summary(
            pop[left].to_numpy(),
            pop[right].to_numpy(),
            n_bootstrap=n_bootstrap,
            seed=seed + 1000 + 100 * i,
        )

        pairwise.append({
            "metric": metric,
            "comparison":
                f"{left}_vs_{right}",

            "left_model": left,
            "right_model": right,

            "n_pairs": int(len(pop)),

            **summary,
        })

    p_raw = [
        x["p_raw"]
        for x in pairwise
    ]

    p_holm = holm_adjust(
        p_raw
    )

    for row, adj in zip(
        pairwise,
        p_holm,
    ):
        row["p_holm"] = float(adj)
        row["holm_reject_0_05"] = bool(
            adj < 0.05
        )

    return (
        pd.DataFrame(descriptive),
        pd.DataFrame(pairwise),
    )


def analyze_long_dataframe(
    df,
    metrics,
    models=("A", "B", "C", "D"),
    n_bootstrap=10000,
    seed=20260818,
):
    validate_long_dataframe(
        df,
        models=models,
    )

    all_desc = []
    all_pairs = []

    for i, metric in enumerate(metrics):
        desc, pairs = analyze_metric(
            df,
            metric,
            models=models,
            n_bootstrap=n_bootstrap,
            seed=seed + i * 10000,
        )

        all_desc.append(desc)
        all_pairs.append(pairs)

    return (
        pd.concat(
            all_desc,
            ignore_index=True,
        ),
        pd.concat(
            all_pairs,
            ignore_index=True,
        ),
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        required=True,
        help="Long-format OOF case CSV."
    )

    p.add_argument(
        "--metrics",
        nargs="+",
        required=True,
    )

    p.add_argument(
        "--out-dir",
        required=True,
    )

    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260818,
    )

    args = p.parse_args()

    df = pd.read_csv(
        args.input
    )

    desc, pairs = analyze_long_dataframe(
        df,
        metrics=args.metrics,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    out = Path(
        args.out_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    desc_path = (
        out
        / "descriptive_statistics.csv"
    )

    pair_path = (
        out
        / "paired_statistics.csv"
    )

    desc.to_csv(
        desc_path,
        index=False,
    )

    pairs.to_csv(
        pair_path,
        index=False,
    )

    print(
        "\nDESCRIPTIVE STATISTICS"
    )
    print(
        desc.to_string(index=False)
    )

    print(
        "\nPAIRED STATISTICS"
    )
    print(
        pairs.to_string(index=False)
    )

    print(
        "\nSaved:",
        desc_path
    )

    print(
        "Saved:",
        pair_path
    )


if __name__ == "__main__":
    main()
