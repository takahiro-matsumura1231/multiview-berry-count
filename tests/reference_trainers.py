"""Reference trainers: cheap, non-learned models that exercise the whole pipeline.

Two purposes.

1. **Smoke-testing.** The nested runner, the seal, the prediction store and the aggregator
   can all be verified end to end without a GPU or a feature cache, so the plumbing is
   known-good *before* the expensive runs start.

2. **Anchors for the fidelity argument.** `ConstantMean` is the shortcut predictor the
   Counting Fidelity axis is designed to expose: it scores a respectable MAE while being
   perfectly flat within every bunch (beta = 0). Reporting it next to the real model is what
   turns "our model counts" from an assertion into a comparison.

These implement the same `Trainer` interface as the real model, so the runner cannot tell
them apart -- which is the point: whatever the runner does to these, it does to the real one.
"""

from __future__ import annotations

import random
from statistics import mean

from grapecount.train.runner import EpochScore, Prediction


class _VideoTable:
    """Stage-videos indexed by bunch, from the dataset manifest."""

    def __init__(self, videos: list[dict]) -> None:
        self.rows = videos
        self.by_bunch: dict[str, list[dict]] = {}
        for v in videos:
            self.by_bunch.setdefault(v["bunch_id"], []).append(v)

    def of(self, bunches: set[str]) -> list[dict]:
        return [v for b in sorted(bunches) for v in self.by_bunch.get(b, [])]


class ConstantMean:
    """Predict the training-set mean count for every video.

    The floor. Any model that cannot beat this has learned nothing, and its fidelity is zero
    by construction, so it also calibrates what a beta_bar of 0 looks like.
    """

    name = "ref_const_mean"

    def __init__(self, videos: list[dict]) -> None:
        self.table = _VideoTable(videos)
        self._value = 0.0

    def fit(self, train_bunches: set[str], val_bunches: set[str] | None,
            seed: int, max_epochs: int | None = None) -> list[EpochScore]:
        rows = self.table.of(train_bunches)
        self._value = mean(float(v["gt_count"]) for v in rows)
        if val_bunches is None:
            return []
        val = self.table.of(val_bunches)
        errs = [abs(float(v["gt_count"]) - self._value) for v in val]
        mae = mean(errs)
        rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5
        # A constant model does not improve with training; a flat trace is honest, and it
        # also checks that epoch selection handles ties by preferring the earliest epoch.
        return [EpochScore(epoch=e, mae=mae, rmse=rmse) for e in range(1, 4)]

    def predict(self, bunches: set[str], seed: int) -> list[Prediction]:
        return [Prediction(video_uid=v["video_uid"], bunch_id=v["bunch_id"],
                           gt=float(v["gt_count"]), pred=self._value,
                           outer_fold=-1, seed=seed, method=self.name)
                for v in self.table.of(bunches)]


class NoisyOracle:
    """Predict the truth plus seeded Gaussian noise.

    Not a model -- a controllable stand-in with a known answer. It gives the smoke test a
    case where MAE, beta_bar and the bootstrap interval all have values that can be reasoned
    about in advance, and it makes the reproducibility check meaningful: same seed, same
    numbers to the last digit.
    """

    name = "ref_noisy_oracle"

    def __init__(self, videos: list[dict], noise: float = 2.0) -> None:
        self.table = _VideoTable(videos)
        self.noise = noise
        self._epochs_fitted = 0

    def fit(self, train_bunches: set[str], val_bunches: set[str] | None,
            seed: int, max_epochs: int | None = None) -> list[EpochScore]:
        self._epochs_fitted = max_epochs or 3
        if val_bunches is None:
            return []
        rng = random.Random((seed, "val", tuple(sorted(val_bunches))[:8]).__hash__())
        trace = []
        for epoch in range(1, 4):
            # improves then plateaus, so the selector has a real minimum to find
            level = self.noise * (1.0 + 0.5 / epoch)
            rows = self.table.of(val_bunches)
            errs = [abs(rng.gauss(0, level)) for _ in rows]
            mae = mean(errs)
            rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5
            trace.append(EpochScore(epoch=epoch, mae=mae, rmse=rmse))
        return trace

    def predict(self, bunches: set[str], seed: int) -> list[Prediction]:
        out = []
        for v in self.table.of(bunches):
            # seeded per video, so a re-run reproduces exactly regardless of ordering
            rng = random.Random(f"{seed}:{v['video_uid']}")
            gt = float(v["gt_count"])
            out.append(Prediction(video_uid=v["video_uid"], bunch_id=v["bunch_id"],
                                  gt=gt, pred=gt + rng.gauss(0, self.noise),
                                  outer_fold=-1, seed=seed, method=self.name))
        return out


class LeakyTrainer(ConstantMean):
    """A trainer that tries to peek at the outer-test set. Used only in tests.

    It exists so the seal is verified by being *tripped*, not merely by being present. A
    guard that has never fired is not known to work.
    """

    name = "ref_leaky"

    def __init__(self, videos: list[dict], seal) -> None:  # noqa: ANN001
        super().__init__(videos)
        self._seal = seal

    def fit(self, train_bunches: set[str], val_bunches: set[str] | None,
            seed: int, max_epochs: int | None = None) -> list[EpochScore]:
        self._seal.get()          # the classic mistake, made explicit
        return super().fit(train_bunches, val_bunches, seed, max_epochs)
