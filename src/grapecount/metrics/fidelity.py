"""Counting Fidelity: does the model count, or does it regress apparent bulk?

The diagnostic can be sanity-checked on a synthetic shortcut predictor: a constant output
has a nonzero MAE but beta_bar 0. Every function takes the physical bunch explicitly, because
the bunch is both the leak unit and the bootstrap cluster.

The diagnostic exploits the thinning series. Within one physical bunch appearance is nearly
fixed and only the berry count moves, by a known delta. A model that has learned apparent
bulk rather than count can score a low MAE across bunches while being flat within one.

    beta_bar            mean within-bunch OLS slope of prediction on truth.  ideal 1.0
    beta_std            spread of that slope across bunches
    kendall_tau_bar     mean within-bunch rank agreement.                    ideal 1.0
    delta_track_err     mean |predicted change - true change| between stages. ideal 0.0
    bulk_shortcut_corr  correlation of residual with an apparent-size proxy.  ideal 0.0

Input: one record per stage-video -- the evaluation unit -- with keys
`bunch`, `gt`, `pred`, and optionally `bulk`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

IDEAL = {"beta_bar": 1.0, "kendall_tau_bar": 1.0, "delta_track_err": 0.0,
         "bulk_shortcut_corr": 0.0}


# --------------------------------------------------------------- aggregate
def aggregate_metrics(gt, pred) -> dict[str, float]:
    """"Is it close?" -- reported alongside fidelity, never instead of it."""
    gt, pred = np.asarray(gt, float), np.asarray(pred, float)
    err = pred - gt
    return {
        "n": int(len(gt)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err) / np.clip(gt, 1e-6, None)) * 100),
        "within1": float(np.mean(np.abs(err) <= 1)),
        "within3": float(np.mean(np.abs(err) <= 3)),
        "within5": float(np.mean(np.abs(err) <= 5)),
        "bias": float(np.mean(err)),
    }


# ----------------------------------------------------------------- helpers
def within_bunch_slope(gt, pred) -> float | None:
    """OLS slope of prediction on truth inside one bunch; None if truth has no spread."""
    gt, pred = np.asarray(gt, float), np.asarray(pred, float)
    var = np.var(gt)
    if var < 1e-9:
        return None
    return float(np.cov(gt, pred, bias=True)[0, 1] / var)


def _tau(a, b) -> float | None:
    if len(a) < 2:
        return None
    t = kendalltau(a, b).statistic
    return None if np.isnan(t) else float(t)


# ---------------------------------------------------------------- fidelity
def fidelity_metrics(records: list[dict], min_distinct_gt: int = 3
                     ) -> tuple[dict[str, float | None], list[dict]]:
    """Per-bunch fidelity, averaged over bunches that carry a thinning series.

    A bunch needs at least `min_distinct_gt` distinct ground-truth values before a
    within-bunch slope means anything; bunches below that are skipped, not counted as zero.
    """
    by_bunch: dict[str, list[dict]] = {}
    for r in records:
        by_bunch.setdefault(r["bunch"], []).append(r)

    betas, taus, deltas, per_bunch = [], [], [], []
    for bunch, rows in sorted(by_bunch.items()):
        # thinning order: high count first. Stage numbers are unreliable, the count is not.
        rows = sorted(rows, key=lambda r: -r["gt"])
        gt = [r["gt"] for r in rows]
        pred = [r["pred"] for r in rows]
        if len(set(gt)) < min_distinct_gt:
            continue
        beta = within_bunch_slope(gt, pred)
        tau = _tau(gt, pred)
        d_gt, d_pred = np.diff(gt), np.diff(pred)
        delta = float(np.mean(np.abs(d_pred - d_gt))) if len(d_gt) else None
        for value, bucket in ((beta, betas), (tau, taus), (delta, deltas)):
            if value is not None:
                bucket.append(value)
        per_bunch.append({"bunch": bunch, "n_stage_videos": len(rows),
                          "n_distinct_gt": len(set(gt)), "beta": beta,
                          "kendall_tau": tau, "delta_track_err": delta})

    bulk_corr = None
    if records and records[0].get("bulk") is not None:
        residual = np.array([r["pred"] - r["gt"] for r in records], float)
        bulk = np.array([r["bulk"] for r in records], float)
        if np.var(bulk) > 1e-9 and np.var(residual) > 1e-9:
            bulk_corr = float(np.corrcoef(residual, bulk)[0, 1])

    mean = lambda x: float(np.mean(x)) if x else None      # noqa: E731
    return {
        "n_bunches_eval": len(betas),
        "beta_bar": mean(betas),
        "beta_std": float(np.std(betas, ddof=1)) if len(betas) > 1 else None,
        "kendall_tau_bar": mean(taus),
        "delta_track_err": mean(deltas),
        "bulk_shortcut_corr": bulk_corr,
    }, per_bunch


def evaluate(records: list[dict], min_distinct_gt: int = 3
             ) -> tuple[dict, list[dict]]:
    """Both axes at once. The project's claim is about their divergence, so they travel
    together and a result is never quoted on MAE alone."""
    agg = aggregate_metrics([r["gt"] for r in records], [r["pred"] for r in records])
    fid, per_bunch = fidelity_metrics(records, min_distinct_gt)
    return {**agg, **fid}, per_bunch
