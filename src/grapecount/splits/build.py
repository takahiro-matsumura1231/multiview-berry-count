"""Build the nested grouped cross-validation split.

Structure
---------
    outer:  5 grouped folds over all bunch labels.  Fold o's bunches are the
            OUTER TEST for run o and are scored exactly once.
    inner:  4 grouped folds over each outer-train separately.  All model selection --
            epoch, hyper-parameters, anything -- happens here and only here.

Without an inner loop, picking the best epoch by the MAE of the fold that is then reported
gives an optimistic estimate in the same direction on every fold, and averaging over folds
does not remove that bias. The inner loop exists to make that
mistake structurally impossible rather than a matter of discipline.

Grouping
--------
The leak unit is the bunch label. Every video, stage, session and frame derived from one
bunch lands in the same fold, at both levels. Inner folds are computed **within each
outer-train independently**, so a bunch's inner assignment differs across outer runs -- and
the outer-test bunches have no inner assignment at all, which is what makes it impossible
to select on them by accident.

Determinism
-----------
Assignment uses md5(seed : key) as a per-bunch rank rather than a global RNG, so a fold is a
pure function of (seed, bunch label) and does not depend on iteration order, dict ordering,
numpy version or how many bunches came before it. Each stratum's starting fold is rotated by
a hash of the stratum, so small strata do not all pile onto fold 0.

Outputs, under <dataset>/splits/:
    bunch_outer.tsv   bunch_id, outer_fold, stratum, n_videos, n_distinct_gt, median_gt, sessions
    bunch_inner.tsv   outer_fold, bunch_id, inner_fold
    composition.tsv   per-fold bunch/video/GT/session composition, for auditing
"""

from __future__ import annotations

import bisect
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def _rank(seed: int, key: str) -> float:
    """Stable pseudo-random rank in [0,1). A pure function of (seed, key)."""
    return int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _rotation(key: str, n_folds: int) -> int:
    return int(hashlib.md5(f"fold:{key}".encode()).hexdigest()[:8], 16) % n_folds


def _band(value: float, edges: list[int]) -> int:
    return bisect.bisect_right(edges, value)


def bunch_metadata(videos: list[dict], count_bands: list[int],
                   stage_classes: list[int]) -> dict[str, dict]:
    """One record per bunch label, with the stratification key.

    `session` for stratification is the session contributing the most videos, so a label that
    spans several sessions still lands in exactly one stratum.
    """
    by_bunch: dict[str, list[dict]] = defaultdict(list)
    for v in videos:
        by_bunch[v["bunch_id"]].append(v)

    meta: dict[str, dict] = {}
    for bunch, rows in by_bunch.items():
        counts = [int(r["gt_count"]) for r in rows]
        n_distinct_gt = len(set(counts))
        sessions = sorted({r["session_id"] for r in rows})
        main_session = Counter(r["session_id"] for r in rows).most_common(1)[0][0]
        meta[bunch] = {
            "stratum": (main_session,
                        _band(median(counts), count_bands),
                        _band(n_distinct_gt, stage_classes)),
            "n_videos": len(rows),
            "n_distinct_gt": n_distinct_gt,
            "median_gt": int(median(counts)),
            "sessions": "|".join(sessions),
        }
    return meta


def assign_folds(bunches: list[str], meta: dict[str, dict], n_folds: int,
                 seed: int, salt: str) -> dict[str, int]:
    """Stratified grouped assignment of `bunches` into `n_folds`.

    `salt` separates the outer assignment from each outer-train's inner assignment, so a
    bunch does not receive correlated positions at the two levels.

    Balancing is on what is actually scored, not on bunch counts. Bunches carry very
    different numbers of stage-videos, so equalising bunch counts alone leaves the number
    of videos per fold uneven.

    Four loads are balanced jointly, because each governs a different reported number:

        videos           the evaluation unit; sets each fold's MAE precision
        bunches          the bootstrap cluster; sets each fold's effective sample size
        fidelity bunches only bunches with >=3 distinct GT carry beta/tau/delta-track, and
                         they can be a minority, so they must not pile up
        total GT         guards against one fold being systematically harder

    Within each stratum, bunches are visited in stable pseudo-random order and each goes to
    the fold minimising the sum of its four loads normalised by their running means. The
    assignment stays a pure function of (seed, bunch label): no randomness beyond the rank,
    and ties broken by fold index.
    """
    strata: dict[tuple, list[str]] = defaultdict(list)
    for b in bunches:
        strata[meta[b]["stratum"]].append(b)

    videos = [0] * n_folds
    bunches_n = [0] * n_folds
    fidelity = [0] * n_folds
    gt_total = [0] * n_folds

    def cost(f: int) -> float:
        # normalise by the running mean so the loads are comparable regardless of scale
        mv = sum(videos) / n_folds or 1.0
        mb = sum(bunches_n) / n_folds or 1.0
        mf = sum(fidelity) / n_folds or 1.0
        mg = sum(gt_total) / n_folds or 1.0
        return videos[f] / mv + bunches_n[f] / mb + fidelity[f] / mf + gt_total[f] / mg

    fold_of: dict[str, int] = {}
    for stratum, keys in sorted(strata.items()):
        # rotation keeps small strata from all starting at fold 0 when loads are tied
        offset = _rotation(f"{salt}:{stratum}", n_folds)
        keys.sort(key=lambda k: _rank(seed, f"{salt}:{k}"))
        for k in keys:
            target = min(range(n_folds),
                         key=lambda j: (cost((j + offset) % n_folds), j))
            target = (target + offset) % n_folds
            fold_of[k] = target
            videos[target] += meta[k]["n_videos"]
            bunches_n[target] += 1
            fidelity[target] += int(meta[k]["n_distinct_gt"] >= 3)
            gt_total[target] += meta[k]["median_gt"] * meta[k]["n_videos"]
    return fold_of


def build(videos: list[dict], n_outer: int, n_inner: int, seed: int,
          count_bands: list[int], stage_classes: list[int]) -> tuple[dict, dict, dict]:
    """Return (outer_fold_of, inner_fold_of, meta).

    inner_fold_of maps (outer_fold, bunch) -> inner fold, and contains an entry only for
    bunches that are in that outer fold's TRAIN set. A missing key is the invariant that
    keeps outer-test out of selection.
    """
    meta = bunch_metadata(videos, count_bands, stage_classes)
    all_bunches = sorted(meta)
    outer_fold_of = assign_folds(all_bunches, meta, n_outer, seed, salt="outer")

    inner_fold_of: dict[tuple[int, str], int] = {}
    for o in range(n_outer):
        train = [b for b in all_bunches if outer_fold_of[b] != o]
        for b, i in assign_folds(train, meta, n_inner, seed, salt=f"inner{o}").items():
            inner_fold_of[(o, b)] = i
    return outer_fold_of, inner_fold_of, meta


def write(out_dir: Path, videos: list[dict], outer_fold_of: dict, inner_fold_of: dict,
          meta: dict, n_outer: int, n_inner: int) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    def _write(name: str, rows: list[dict], fields: list[str]) -> None:
        path = out_dir / name
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    _write("bunch_outer.tsv",
           [{"bunch_id": b, "outer_fold": outer_fold_of[b],
             "stratum": "|".join(str(x) for x in meta[b]["stratum"]),
             "n_videos": meta[b]["n_videos"], "n_distinct_gt": meta[b]["n_distinct_gt"],
             "median_gt": meta[b]["median_gt"], "sessions": meta[b]["sessions"]}
            for b in sorted(meta)],
           ["bunch_id", "outer_fold", "stratum", "n_videos", "n_distinct_gt",
            "median_gt", "sessions"])

    _write("bunch_inner.tsv",
           [{"outer_fold": o, "bunch_id": b, "inner_fold": i}
            for (o, b), i in sorted(inner_fold_of.items())],
           ["outer_fold", "bunch_id", "inner_fold"])

    # composition, for auditing that the folds are actually comparable
    by_bunch: dict[str, list[dict]] = defaultdict(list)
    for v in videos:
        by_bunch[v["bunch_id"]].append(v)

    rows = []
    for o in range(n_outer):
        for role, bunches in (
            ("outer_test", [b for b in meta if outer_fold_of[b] == o]),
            ("outer_train", [b for b in meta if outer_fold_of[b] != o]),
        ):
            rows.append(_composition(o, role, "", bunches, by_bunch, meta))
        for i in range(n_inner):
            inner_val = [b for b in meta if inner_fold_of.get((o, b)) == i]
            inner_tr = [b for b in meta
                        if (o, b) in inner_fold_of and inner_fold_of[(o, b)] != i]
            rows.append(_composition(o, "inner_val", str(i), inner_val, by_bunch, meta))
            rows.append(_composition(o, "inner_train", str(i), inner_tr, by_bunch, meta))
    _write("composition.tsv", rows, list(rows[0]))
    return hashes


def _composition(outer: int, role: str, inner: str, bunches: list[str],
                 by_bunch: dict, meta: dict) -> dict:
    vids = [v for b in bunches for v in by_bunch[b]]
    gts = [int(v["gt_count"]) for v in vids]
    return {
        "outer_fold": outer, "role": role, "inner_fold": inner,
        "n_bunches": len(bunches), "n_videos": len(vids),
        "n_fidelity_bunches": sum(1 for b in bunches if meta[b]["n_distinct_gt"] >= 3),
        "gt_min": min(gts) if gts else 0,
        "gt_median": int(median(gts)) if gts else 0,
        "gt_max": max(gts) if gts else 0,
        "gt_mean": round(sum(gts) / len(gts), 2) if gts else 0,
        "n_sessions": len({v["session_id"] for v in vids}),
    }
