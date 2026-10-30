"""View-set selection.

Training, evaluation and augmentation share this single implementation so the sampling
logic cannot drift apart between them.

A capture is a handheld orbit around the bunch, so time is a proxy for viewing angle. `tau`
normalises time to [0,1] within a video, and a K-view set is K positions spaced 1/K apart
starting from a phase, i.e. K roughly evenly spaced viewpoints.

    training     random phase each epoch, plus the partial-orbit augmentation below
    evaluation   fixed phases i/M for i in 0..M-1, nothing sampled
"""

from __future__ import annotations

import random


def tau_of(timestamps_ms: list[int]) -> list[float]:
    """Position within the orbit, normalised to [0,1] inside this video."""
    lo, hi = min(timestamps_ms), max(timestamps_ms)
    span = (hi - lo) or 1
    return [(t - lo) / span for t in timestamps_ms]


def phase_indices(taus: list[float], k: int, phase: float) -> list[int]:
    """Indices of k phase-spaced frames, targets at (phase + j/k) mod 1.

    Distance is circular, because the orbit wraps: tau 0.99 and tau 0.01 are adjacent
    viewpoints, not opposite ones. Each frame is used at most once until the pool runs out.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    targets = [((phase + j / k) % 1.0) for j in range(k)]
    chosen: list[int] = []
    used: set[int] = set()
    for target in targets:
        pool = [i for i in range(len(taus)) if i not in used] or list(range(len(taus)))
        best = min(pool, key=lambda i: min(abs(taus[i] - target), 1 - abs(taus[i] - target)))
        chosen.append(best)
        used.add(best)
    return chosen


def phase_indices_window(taus: list[float], k: int, phase: float,
                         width: float = 1.0, speed_jitter: float = 0.0,
                         rng: random.Random | None = None) -> list[int]:
    """k views spread over an ARC of the orbit rather than the whole circle.

    `width=1.0` is exactly `phase_indices`. Smaller widths model the capture this system
    actually receives in the field: a farmer rotates the bunch part of a turn, unevenly.
    Training on full orbits only was measurably fragile on real handheld clips, and the
    partial-orbit augmentation improved both cross-validation error and the external
    clips (the `PARTIAL_ORBIT_MIXTURE` mixture below).

    `speed_jitter` perturbs the spacing inside the arc, modelling an uneven rotation speed.
    """
    if not 0 < width <= 1.0:
        raise ValueError("width must be in (0, 1]")
    offsets = [j / k for j in range(k)]
    if speed_jitter and rng is not None:
        offsets = [min(max(o + rng.uniform(-speed_jitter, speed_jitter) / k, 0.0), 1.0)
                   for o in offsets]
        offsets.sort()
    targets = [((phase + o * width) % 1.0) for o in offsets]

    chosen: list[int] = []
    used: set[int] = set()
    for target in targets:
        pool = [i for i in range(len(taus)) if i not in used] or list(range(len(taus)))
        best = min(pool, key=lambda i: min(abs(taus[i] - target), 1 - abs(taus[i] - target)))
        chosen.append(best)
        used.add(best)
    return chosen


# The adopted training mixture: mostly partial arcs, since that is what field capture looks
# like. Widths and probabilities are listed below.
PARTIAL_ORBIT_MIXTURE = [(1.00, 0.1), (0.60, 0.2), (0.40, 0.3), (0.20, 0.4)]   # (width, probability)


def sample_training_views(taus: list[float], k: int, rng: random.Random,
                          mixture: list[tuple[float, float]] | None = None,
                          speed_jitter: float = 0.15) -> list[int]:
    """One training view-set: random phase, arc width drawn from the mixture."""
    mixture = mixture or PARTIAL_ORBIT_MIXTURE
    r = rng.random()
    cumulative = 0.0
    width = mixture[-1][0]
    for w, p in mixture:
        cumulative += p
        if r <= cumulative:
            width = w
            break
    return phase_indices_window(taus, k, rng.random(), width, speed_jitter, rng)


def evaluation_view_sets(taus: list[float], k: int, m: int) -> list[list[int]]:
    """The M deterministic view-sets a video is scored on. No randomness at all."""
    return [phase_indices(taus, k, i / m) for i in range(m)]
