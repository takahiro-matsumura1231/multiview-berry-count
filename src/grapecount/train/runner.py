"""The nested cross-validation runner.

This is the piece that decides whether the protocol is actually obeyed, so it is written so
that the honest path is the only path:

  * `Trainer` is an interface. The runner hands it bunch sets; it never hands it the outer
    test set during selection, because during selection the runner does not have it -- it
    holds a `SealedBunches` that raises on read.
  * Epoch selection is a separate function that takes *only* inner results. It has no
    parameter through which an outer score could arrive.
  * The outer fold is scored exactly once, after an explicit `unseal(reason)`, and a second
    unseal raises.

The failure being designed against is not malice. A training loop that simply has the
reported fold in scope and uses it for early stopping gives an optimistic score, in the same
direction on every fold, and goes unnoticed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from ..splits.access import NestedSplit
from .seal import SealedBunches


@dataclass(frozen=True)
class EpochScore:
    epoch: int
    mae: float
    rmse: float = float("nan")


@dataclass(frozen=True)
class Prediction:
    video_uid: str
    bunch_id: str
    gt: float
    pred: float
    outer_fold: int
    seed: int
    method: str


class Trainer(Protocol):
    """What the runner needs from a model.

    Note what is absent: no method receives the outer-test set except `predict`, and
    `predict` is only ever called after the seal is broken.
    """

    def fit(self, train_bunches: set[str], val_bunches: set[str] | None,
            seed: int, max_epochs: int | None = None) -> list[EpochScore]:
        """Train. If `val_bunches` is given, return the per-epoch validation trace.

        With `val_bunches=None` this is the refit: train for exactly `max_epochs` and
        return an empty trace, because there is nothing legitimate left to select on.
        """

    def predict(self, bunches: set[str], seed: int) -> list[Prediction]:
        """One prediction per stage-video, already aggregated over the M view-sets."""


def select_epoch(inner_traces: list[list[EpochScore]]) -> tuple[int, dict]:
    """Choose the epoch by MEAN inner-validation MAE across the inner folds.

    Takes inner traces and nothing else. There is deliberately no parameter through which an
    outer-fold score could be passed, so this function cannot be misused by accident.

    Tie-break, in the order fixed by the protocol: fewer epochs, then lower inner-val RMSE,
    then lowest inner fold index (implicit in the stable ordering of `inner_traces`).
    """
    if not inner_traces or not all(inner_traces):
        raise ValueError("epoch selection needs a non-empty trace from every inner fold")

    common = min(len(t) for t in inner_traces)
    if common == 0:
        raise ValueError("an inner fold produced no epochs")

    stats = []
    for e in range(common):
        maes = [t[e].mae for t in inner_traces]
        rmses = [t[e].rmse for t in inner_traces]
        stats.append({
            "epoch": inner_traces[0][e].epoch,
            "mean_mae": sum(maes) / len(maes),
            "mean_rmse": sum(rmses) / len(rmses),
        })

    best = min(stats, key=lambda s: (round(s["mean_mae"], 10), s["epoch"], s["mean_rmse"]))
    return best["epoch"], {
        "criterion": "mean inner-val MAE",
        "n_inner_folds": len(inner_traces),
        "epochs_considered": common,
        "chosen_epoch": best["epoch"],
        "chosen_mean_mae": best["mean_mae"],
        "trace": stats,
    }


@dataclass
class OuterResult:
    outer_fold: int
    seed: int
    method: str
    chosen_epoch: int
    selection: dict
    predictions: list[Prediction]
    seal_audit: dict
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def run_outer_fold(outer: int, split: NestedSplit, trainer: Trainer, seed: int,
                   method: str) -> OuterResult:
    """Inner selection, refit on the whole outer-train, then score outer-test once."""
    t0 = time.time()
    seal = SealedBunches(split.outer_test(outer))

    # ---- 1. selection, entirely inside outer-train -------------------------
    inner_traces: list[list[EpochScore]] = []
    for inner in range(split.n_inner):
        tr, va = split.inner_split(outer, inner)
        inner_traces.append(trainer.fit(train_bunches=tr, val_bunches=va, seed=seed))
    chosen_epoch, selection = select_epoch(inner_traces)

    # ---- 2. refit at the chosen epoch, no validation ------------------------
    outer_train = split.outer_train(outer)
    trainer.fit(train_bunches=outer_train, val_bunches=None, seed=seed,
                max_epochs=chosen_epoch)

    # ---- 3. score once -----------------------------------------------------
    seal.unseal(f"selection complete for outer fold {outer}: epoch {chosen_epoch}")
    test_bunches = set(seal.get())
    raw = trainer.predict(bunches=test_bunches, seed=seed)

    # The RUNNER stamps fold identity, not the trainer. A trainer does not know which outer
    # fold it is serving, and asking it to report one invites exactly the defect this caught
    # in an earlier version: every prediction carried outer_fold=-1, which collapsed the five folds into
    # one and would have made the per-fold breakdown and fold SD silently meaningless.
    predictions = [replace(p, outer_fold=outer, seed=seed, method=method) for p in raw]

    stray = {p.bunch_id for p in predictions} - test_bunches
    if stray:
        raise RuntimeError(
            f"outer {outer}: trainer returned predictions for {len(stray)} bunches that are "
            f"not in the outer-test set, e.g. {sorted(stray)[:5]}"
        )

    return OuterResult(
        outer_fold=outer, seed=seed, method=method, chosen_epoch=chosen_epoch,
        selection=selection, predictions=predictions, seal_audit=seal.audit(),
        seconds=time.time() - t0,
    )


def write_predictions(path: Path, results: list[OuterResult]) -> None:
    """One row per (method, seed, outer fold, video). The only authoritative numbers."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "seed", "outer_fold", "video_uid", "bunch_id", "gt", "pred"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in sorted(results, key=lambda r: (r.method, r.seed, r.outer_fold)):
            for p in sorted(r.predictions, key=lambda p: int(p.video_uid)):
                w.writerow({"method": p.method, "seed": p.seed, "outer_fold": p.outer_fold,
                            "video_uid": p.video_uid, "bunch_id": p.bunch_id,
                            "gt": p.gt, "pred": p.pred})


def write_run_manifest(path: Path, results: list[OuterResult], extra: dict) -> None:
    """Provenance for every run. Without the script and configuration that produced a set of
    predictions they cannot be audited; a run without a manifest is not a result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
        "outer_runs": [
            {"outer_fold": r.outer_fold, "seed": r.seed, "method": r.method,
             "chosen_epoch": r.chosen_epoch, "selection": r.selection,
             "seal": r.seal_audit, "n_predictions": len(r.predictions),
             "seconds": round(r.seconds, 2), **r.extra}
            for r in sorted(results, key=lambda r: (r.method, r.seed, r.outer_fold))
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
