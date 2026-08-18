import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from statistics import (  # noqa: E402
    analyze_metric,
    bootstrap_ci,
    holm_adjust,
    metric_population,
    paired_delta_summary,
    paired_rank_biserial,
    paired_wilcoxon,
    validate_long_dataframe,
)


MODELS = ["A", "B", "C", "D"]


def make_long(values):
    rows = []

    for case, model_values in values.items():
        for model in MODELS:
            rows.append({
                "case": case,
                "model": model,
                "tumor_dice":
                    model_values[model],
            })

    return pd.DataFrame(rows)


def test_duplicate_case_model_fails():
    df = make_long({
        "c1": {
            "A": 0.1,
            "B": 0.2,
            "C": 0.3,
            "D": 0.4,
        }
    })

    df = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_long_dataframe(df)


def test_incomplete_model_coverage_fails():
    df = make_long({
        "c1": {
            "A": 0.1,
            "B": 0.2,
            "C": 0.3,
            "D": 0.4,
        }
    })

    df = df[
        ~(
            (df.case == "c1")
            & (df.model == "D")
        )
    ]

    with pytest.raises(
        ValueError,
        match="incomplete",
    ):
        validate_long_dataframe(df)


def test_model_specific_missingness_fails():
    df = make_long({
        "c1": {
            "A": 0.1,
            "B": 0.2,
            "C": 0.3,
            "D": 0.4,
        },
        "c2": {
            "A": 0.5,
            "B": np.nan,
            "C": 0.6,
            "D": 0.7,
        },
    })

    with pytest.raises(
        ValueError,
        match="model-specific missingness",
    ):
        metric_population(
            df,
            "tumor_dice",
        )


def test_identical_protocol_missingness_is_allowed():
    df = make_long({
        "positive": {
            "A": 0.70,
            "B": 0.71,
            "C": 0.72,
            "D": 0.73,
        },
        "gt_empty": {
            "A": np.nan,
            "B": np.nan,
            "C": np.nan,
            "D": np.nan,
        },
    })

    pop = metric_population(
        df,
        "tumor_dice",
    )

    assert len(pop) == 1
    assert pop.index.tolist() == [
        "positive"
    ]


def test_all_zero_difference():
    a = np.array([
        0.2, 0.4, 0.6, 0.8
    ])

    b = a.copy()

    assert paired_wilcoxon(
        a,
        b,
    ) == pytest.approx(1.0)

    assert paired_rank_biserial(
        a,
        b,
    ) == pytest.approx(0.0)

    s = paired_delta_summary(
        a,
        b,
        n_bootstrap=500,
        seed=1,
    )

    assert s["mean_delta"] == pytest.approx(0.0)
    assert s["median_delta"] == pytest.approx(0.0)
    assert s["mean_delta_ci95_low"] == pytest.approx(0.0)
    assert s["mean_delta_ci95_high"] == pytest.approx(0.0)


def test_constant_positive_effect_direction():
    a = np.linspace(
        0.1,
        0.8,
        20,
    )

    b = a + 0.1

    s = paired_delta_summary(
        a,
        b,
        n_bootstrap=500,
        seed=2,
    )

    assert s["mean_delta"] == pytest.approx(
        0.1
    )

    assert s["median_delta"] == pytest.approx(
        0.1
    )

    assert s["rank_biserial"] == pytest.approx(
        1.0
    )

    assert s["p_raw"] < 0.05


def test_holm_adjustment():
    p = np.array([
        0.001,
        0.01,
        0.03,
        0.20,
        0.80,
    ])

    adj = holm_adjust(p)

    assert np.all(adj >= p)
    assert np.all(adj <= 1.0)

    assert adj[0] == pytest.approx(
        0.005
    )


def test_bootstrap_is_deterministic():
    x = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
        100.0,
    ])

    a = bootstrap_ci(
        x,
        n_bootstrap=1000,
        seed=123,
    )

    b = bootstrap_ci(
        x,
        n_bootstrap=1000,
        seed=123,
    )

    assert a == pytest.approx(b)


def test_outlier_keeps_median_robust():
    x = np.array([
        1.0,
        1.0,
        1.0,
        1.0,
        1000.0,
    ])

    assert np.median(x) == pytest.approx(
        1.0
    )

    assert np.mean(x) > 100


def test_full_metric_analysis_has_five_comparisons():
    values = {}

    for i in range(20):
        base = 0.50 + i * 0.005

        values[f"case_{i:02d}"] = {
            "A": base,
            "B": base + 0.01,
            "C": base + 0.02,
            "D": base + 0.015,
        }

    df = make_long(values)

    validate_long_dataframe(df)

    desc, pair = analyze_metric(
        df,
        "tumor_dice",
        n_bootstrap=500,
        seed=99,
    )

    assert len(desc) == 4
    assert len(pair) == 5

    assert set(
        pair["comparison"]
    ) == {
        "A_vs_B",
        "A_vs_C",
        "A_vs_D",
        "B_vs_D",
        "C_vs_D",
    }

    assert pair["p_holm"].notna().all()
