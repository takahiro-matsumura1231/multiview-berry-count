"""Tests for the nested split.

These encode the key invariant: the fold a number is reported on must be unreachable from
the code that chooses anything. If the evaluation fold is reachable from the training loop,
every trainer will select its best epoch on it and the reported score is optimistic.

    python -m pytest tests/test_splits.py -q
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grapecount.splits.access import load  # noqa: E402
from grapecount.splits.build import assign_folds, bunch_metadata  # noqa: E402

DATASET = REPO / "data" / "dataset_v2"
pytestmark = pytest.mark.skipif(
    not (DATASET / "splits" / "bunch_outer.tsv").exists(),
    reason="dataset v2 splits not generated in this checkout",
)


@pytest.fixture(scope="module")
def split():
    return load(DATASET)


@pytest.fixture(scope="module")
def videos():
    with (DATASET / "manifests" / "videos.tsv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ------------------------------------------------------------------ invariants
def test_no_invariant_violations(split):
    assert split.check() == []


def test_every_bunch_has_exactly_one_outer_fold(split, videos):
    bunches = {v["bunch_id"] for v in videos}
    assert set(split.outer_fold_of) == bunches


def test_outer_folds_partition_the_bunches(split):
    seen: set[str] = set()
    for o in range(split.n_outer):
        test = split.outer_test(o)
        assert not (seen & test), "a bunch appears in two outer test sets"
        seen |= test
    assert seen == set(split.outer_fold_of), "some bunch is never an outer test"


def test_every_video_is_predicted_exactly_once_out_of_fold(split, videos):
    """The pooled OOF estimate requires each stage-video to be scored once and only once."""
    counted: dict[str, int] = defaultdict(int)
    for o in range(split.n_outer):
        test = split.outer_test(o)
        for v in videos:
            if v["bunch_id"] in test:
                counted[v["video_uid"]] += 1
    assert len(counted) == len(videos)
    assert set(counted.values()) == {1}


# --------------------------------------------------- selection cannot see outer-test
def test_outer_test_has_no_inner_assignment(split):
    """The structural guard: selecting on outer-test requires calling the wrong function."""
    for o in range(split.n_outer):
        test = split.outer_test(o)
        inner = {b for (oo, b) in split.inner_fold_of if oo == o}
        assert not (inner & test)


def test_inner_splits_never_touch_outer_test(split):
    for o in range(split.n_outer):
        test = split.outer_test(o)
        for i in range(split.n_inner):
            train, val = split.inner_split(o, i)
            assert not (train & test)
            assert not (val & test)
            assert not (train & val)


def test_inner_folds_partition_the_outer_train(split):
    for o in range(split.n_outer):
        train = split.outer_train(o)
        seen: set[str] = set()
        for i in range(split.n_inner):
            _, val = split.inner_split(o, i)
            assert not (seen & val)
            seen |= val
        assert seen == train


def test_inner_assignment_differs_across_outer_folds(split):
    """Inner folds are rebuilt inside each outer-train, so they must not be a fixed labelling."""
    shared = split.outer_train(0) & split.outer_train(1)
    same = sum(1 for b in shared
               if split.inner_fold_of[(0, b)] == split.inner_fold_of[(1, b)])
    assert same < 0.9 * len(shared)


# ------------------------------------------------------------------- determinism
def test_assignment_is_a_pure_function_of_seed_and_label(videos):
    meta = bunch_metadata(videos, [30, 40, 50, 60], [2, 5, 10])
    a = assign_folds(sorted(meta), meta, 5, 42, "outer")
    b = assign_folds(sorted(meta, reverse=True), meta, 5, 42, "outer")
    assert a == b, "fold assignment depends on input ordering"


def test_seed_changes_the_assignment(videos):
    meta = bunch_metadata(videos, [30, 40, 50, 60], [2, 5, 10])
    a = assign_folds(sorted(meta), meta, 5, 42, "outer")
    b = assign_folds(sorted(meta), meta, 5, 43, "outer")
    assert a != b


def test_matches_the_generated_manifest(split, videos):
    """The committed split files must equal what the builder produces today."""
    meta = bunch_metadata(videos, [30, 40, 50, 60], [2, 5, 10])
    assert assign_folds(sorted(meta), meta, 5, 42, "outer") == split.outer_fold_of


# ------------------------------------------------------------------- balance
def test_folds_are_comparable(split, videos):
    by_bunch: dict[str, list[dict]] = defaultdict(list)
    for v in videos:
        by_bunch[v["bunch_id"]].append(v)
    n_videos, n_fid = [], []
    for o in range(split.n_outer):
        test = split.outer_test(o)
        n_videos.append(sum(len(by_bunch[b]) for b in test))
        n_fid.append(sum(1 for b in test if len({r["gt_count"] for r in by_bunch[b]}) >= 3))
    assert max(n_videos) / min(n_videos) < 1.15, f"video counts unbalanced: {n_videos}"
    assert max(n_fid) / min(n_fid) < 1.35, f"fidelity bunches unbalanced: {n_fid}"


def test_every_fold_sees_every_session(split, videos):
    by_bunch: dict[str, set[str]] = defaultdict(set)
    for v in videos:
        by_bunch[v["bunch_id"]].add(v["session_id"])
    all_sessions = {s for ss in by_bunch.values() for s in ss}
    for o in range(split.n_outer):
        got = {s for b in split.outer_test(o) for s in by_bunch[b]}
        assert got == all_sessions, f"outer {o} is missing {all_sessions - got}"
