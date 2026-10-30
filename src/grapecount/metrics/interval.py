"""Confidence intervals by resampling physical bunches.

A "±" computed as the standard deviation across cross-validation folds is easily read as
uncertainty on the estimate. They are different quantities and can differ substantially:

    fold SD      how much the five fold estimates differ from each other. Shrinks if the
                 folds happen to agree; says nothing about how many bunches were measured.
    cluster CI   how much the estimate would move if the study were repeated with a
                 different sample of bunches. This is the uncertainty on the number.

Resampling is over **bunches, not videos**. A bunch can contribute many stage-videos of the
same physical object photographed repeatedly; treating those as independent would understate
the interval by roughly the square root of the cluster size. Likewise, spread between models
evaluated on a fixed test set is not an interval on the metric.

Both numbers are reported, side by side and labelled. Neither substitutes for the other.
"""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean, stdev
from typing import Callable

Records = list[dict]
Statistic = Callable[[Records], float]

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 20260802


def group_by_cluster(records: Records, cluster_key: str = "bunch") -> dict[str, Records]:
    out: dict[str, Records] = defaultdict(list)
    for r in records:
        out[r[cluster_key]].append(r)
    return dict(out)


def cluster_bootstrap(
    records: Records,
    statistic: Statistic,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    cluster_key: str = "bunch",
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile CI for `statistic`, resampling whole clusters with replacement.

    Deterministic given (records, seed): the cluster order is sorted before sampling, so the
    interval does not depend on dict iteration order.
    """
    clusters = group_by_cluster(records, cluster_key)
    names = sorted(clusters)
    if len(names) < 2:
        raise ValueError(f"cluster bootstrap needs >=2 clusters, got {len(names)}")

    rng = random.Random(seed)
    point = statistic(records)
    draws: list[float] = []
    for _ in range(n_resamples):
        sample: Records = []
        for _ in names:
            sample.extend(clusters[rng.choice(names)])
        try:
            draws.append(statistic(sample))
        except (ValueError, ZeroDivisionError):
            continue      # a resample may contain too few distinct bunches for a statistic

    draws.sort()
    lo_i = int(alpha / 2 * (len(draws) - 1))
    hi_i = int(round((1 - alpha / 2) * (len(draws) - 1)))
    return {
        "point": point,
        "ci_low": draws[lo_i],
        "ci_high": draws[hi_i],
        "n_clusters": len(names),
        "n_records": len(records),
        "n_resamples": len(draws),
        "seed": seed,
        "alpha": alpha,
    }


def fold_spread(per_fold: list[float]) -> dict[str, float]:
    """Mean and SAMPLE SD (n-1) across folds.

    Labelled `fold_sd`, never `ci`. The n-1 denominator is used consistently; mixing n and
    n-1 conventions gives different "±" values for the same run.
    """
    return {
        "mean": mean(per_fold),
        "fold_sd": stdev(per_fold) if len(per_fold) > 1 else float("nan"),
        "n_folds": len(per_fold),
    }


def report(
    oof_records: Records,
    per_fold_values: list[float],
    statistic: Statistic,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    cluster_key: str = "bunch",
) -> dict:
    """The three quantities reported together, in one object and never merged.

    `pooled` is the headline: computed once over every out-of-fold prediction, so each
    stage-video contributes exactly once.
    """
    spread = fold_spread(per_fold_values)
    boot = cluster_bootstrap(oof_records, statistic, n_resamples, seed, cluster_key)
    return {
        "pooled": boot["point"],
        "interval_95": (boot["ci_low"], boot["ci_high"]),
        "interval_kind": f"{cluster_key}-cluster percentile bootstrap",
        "per_fold": list(per_fold_values),
        "fold_mean": spread["mean"],
        "fold_sd": spread["fold_sd"],
        "fold_sd_is_not_an_interval": True,
        "n_clusters": boot["n_clusters"],
        "n_records": boot["n_records"],
        "n_resamples": boot["n_resamples"],
        "seed": seed,
    }
