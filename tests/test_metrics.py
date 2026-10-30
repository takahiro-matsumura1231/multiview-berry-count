"""Tests for scoring.

The fidelity metrics exist to catch a model that regresses apparent bulk instead of
counting, so the decisive test is that they actually do: a shortcut predictor that ignores
the thinning series must score well on MAE and badly on beta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grapecount.metrics import (  # noqa: E402
    aggregate_metrics, cluster_bootstrap, evaluate, fold_spread, within_bunch_slope,
)


def _series(bunch: str, gts: list[float], preds: list[float]) -> list[dict]:
    return [{"bunch": bunch, "gt": g, "pred": p} for g, p in zip(gts, preds)]


# ------------------------------------------------------------------ aggregate
def test_perfect_prediction():
    m = aggregate_metrics([10, 20, 30], [10, 20, 30])
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["bias"] == 0
    assert m["within1"] == 1.0


def test_bias_is_signed():
    assert aggregate_metrics([10, 10], [12, 12])["bias"] == pytest.approx(2.0)
    assert aggregate_metrics([10, 10], [8, 8])["bias"] == pytest.approx(-2.0)


# ------------------------------------------------------------------- fidelity
def test_slope_is_one_when_tracking_perfectly():
    assert within_bunch_slope([50, 45, 40], [50, 45, 40]) == pytest.approx(1.0)


def test_slope_is_zero_for_a_constant_prediction():
    assert within_bunch_slope([50, 45, 40], [45, 45, 45]) == pytest.approx(0.0)


def test_slope_is_none_without_spread():
    assert within_bunch_slope([40, 40, 40], [41, 39, 40]) is None


def test_the_shortcut_predictor_is_caught():
    """A model that predicts each bunch's mean count: good MAE, no counting.

    This is the whole reason the fidelity axis exists. MAE alone would rank it as a
    reasonable model; beta_bar reveals it is flat within every bunch.
    """
    records = []
    for b, base in (("A", 50), ("B", 30), ("C", 70)):
        gts = [base + 4, base, base - 4]
        records += _series(b, gts, [base] * 3)      # ignores the thinning series entirely
    metrics, _ = evaluate(records)
    assert metrics["MAE"] < 3.0, "the shortcut should look acceptable on MAE"
    assert metrics["beta_bar"] == pytest.approx(0.0, abs=1e-9), "but must be flat within bunches"
    assert metrics["delta_track_err"] > 3.0


def test_a_true_counter_scores_well_on_both():
    records = []
    for b, base in (("A", 50), ("B", 30), ("C", 70)):
        gts = [base + 4, base, base - 4]
        records += _series(b, gts, [g + 0.5 for g in gts])
    metrics, _ = evaluate(records)
    assert metrics["beta_bar"] == pytest.approx(1.0, abs=1e-9)
    assert metrics["kendall_tau_bar"] == pytest.approx(1.0)
    assert metrics["delta_track_err"] == pytest.approx(0.0, abs=1e-9)


def test_bunches_below_the_minimum_are_skipped_not_zeroed():
    records = _series("A", [50, 45, 40], [50, 45, 40]) + _series("B", [30, 30], [30, 30])
    metrics, per_bunch = evaluate(records, min_distinct_gt=3)
    assert metrics["n_bunches_eval"] == 1
    assert [p["bunch"] for p in per_bunch] == ["A"]


# ------------------------------------------------------------------ intervals
def _mae(records: list[dict]) -> float:
    return sum(abs(r["gt"] - r["pred"]) for r in records) / len(records)


def test_bootstrap_is_deterministic():
    records = [{"bunch": f"b{i // 4}", "gt": 40 + i, "pred": 40 + i + (i % 3)}
               for i in range(80)]
    a = cluster_bootstrap(records, _mae, n_resamples=500, seed=7)
    b = cluster_bootstrap(records, _mae, n_resamples=500, seed=7)
    assert a == b


def test_bootstrap_interval_contains_the_point_estimate():
    records = [{"bunch": f"b{i // 4}", "gt": 40 + i, "pred": 40 + i + (i % 3)}
               for i in range(80)]
    r = cluster_bootstrap(records, _mae, n_resamples=1000, seed=7)
    assert r["ci_low"] <= r["point"] <= r["ci_high"]
    assert r["n_clusters"] == 20


def test_clustering_widens_the_interval_versus_ignoring_it():
    """Videos within a bunch are correlated; pretending otherwise understates the interval.

    Here a bunch's error is a property of the bunch and is shared by all ten of its videos,
    which is the structure real data has -- one physical object photographed repeatedly.
    Resampling bunches must then give a wider interval than resampling videos, because the
    study really only has 20 independent observations, not 200.
    """
    records = []
    for b in range(20):
        error = float(b)                      # the error belongs to the bunch, not the video
        for k in range(10):
            records.append({"bunch": f"b{b}", "row_id": f"b{b}_{k}",
                            "gt": 40 + k, "pred": 40 + k + error})
    clustered = cluster_bootstrap(records, _mae, n_resamples=2000, seed=3)
    naive = cluster_bootstrap(records, _mae, n_resamples=2000, seed=3, cluster_key="row_id")
    clustered_w = clustered["ci_high"] - clustered["ci_low"]
    naive_w = naive["ci_high"] - naive["ci_low"]
    assert clustered["n_clusters"] == 20 and naive["n_clusters"] == 200
    assert clustered_w > 1.5 * naive_w, (clustered_w, naive_w)


def test_bootstrap_needs_at_least_two_clusters():
    with pytest.raises(ValueError):
        cluster_bootstrap([{"bunch": "a", "gt": 1, "pred": 1}], _mae)


def test_fold_spread_uses_sample_sd():
    s = fold_spread([2.0, 4.0])
    assert s["fold_sd"] == pytest.approx(1.4142135, rel=1e-5)   # n-1, not n
