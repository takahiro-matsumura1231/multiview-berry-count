"""The stage-2 view-set cache must return exactly what recomputing would return.

`ViewSetCache` memoises `sample_training_views` and `evaluation_view_sets` across the seven
arms of one `(seed, outer fold)` block. That is only sound because the selection is a pure
function of `(k, seed, epoch, uid)` -- the RNG is constructed fresh from that string on every
call and `taus` is fixed per video.

"It is obviously a pure function" is exactly how a cache that is not one gets in, and this one
would fail silently: every arm would train on the FIRST arm's view-sets, which is not wrong in
any way a metric would show -- every arm would still see identical, valid, differently-
sampled-per-epoch views. It would simply no longer be the sampler the protocol specifies.

So the property is tested rather than argued: memoised and unmemoised selections are compared
element by element, over many videos, several epochs, both seeds' shapes and every configured K.

    python -m pytest tests/test_view_cache.py -q
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grapecount.views import evaluation_view_sets, sample_training_views  # noqa: E402

torch = pytest.importorskip("torch", reason="the view cache stores index tensors")
pytest.importorskip("torchvision", reason="the trainer module imports torchvision")

from grapecount.train.baselines import ViewSetCache  # noqa: E402


def taus_for(n: int, jitter: int) -> list[float]:
    rng = random.Random(jitter)
    raw = sorted(rng.uniform(0, 1) for _ in range(n))
    lo, hi = raw[0], raw[-1]
    return [(t - lo) / (hi - lo or 1) for t in raw]


VIDEOS = {f"v{i}": taus_for(16 + 3 * i, i) for i in range(12)}


@pytest.mark.parametrize("k", [1, 3, 9])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cached_training_rows_equal_a_fresh_computation(k, seed):
    cache = ViewSetCache()
    for epoch in (1, 2, 7, 60, 119, 120):
        for uid, taus in VIDEOS.items():
            expected = sample_training_views(taus, k, random.Random(f"{seed}:{epoch}:{uid}"))
            got = cache.training_rows(taus, k, seed, epoch, uid)
            assert got.tolist() == expected, (k, seed, epoch, uid)
            assert got.dtype == torch.long


@pytest.mark.parametrize("k", [1, 3, 9])
def test_cached_evaluation_rows_equal_a_fresh_computation(k):
    cache = ViewSetCache()
    for uid, taus in VIDEOS.items():
        expected = evaluation_view_sets(taus, k, m=1)[0]
        assert cache.evaluation_rows(taus, k, uid).tolist() == expected


def test_a_second_read_returns_the_same_selection_and_counts_as_a_hit():
    """The whole point: the arms after the first must get the identical selection."""
    cache = ViewSetCache()
    taus = VIDEOS["v3"]
    first = cache.training_rows(taus, 9, 0, 5, "v3")
    second = cache.training_rows(taus, 9, 0, 5, "v3")
    assert first.tolist() == second.tolist()
    assert cache.stats() == {"hits": 1, "misses": 1, "hit_rate": 0.5, "held": 1}


@pytest.mark.parametrize("differing", ["k", "seed", "epoch", "uid"])
def test_every_part_of_the_key_separates_entries(differing):
    """A key missing any of these would hand one arm another arm's -- or another epoch's --
    views, and the run would complete with a full, plausible results table."""
    cache = ViewSetCache()
    base = {"taus": VIDEOS["v5"], "k": 9, "seed": 0, "epoch": 4, "uid": "v5"}
    a = cache.training_rows(base["taus"], base["k"], base["seed"], base["epoch"], base["uid"])
    other = dict(base)
    other[differing] = {"k": 3, "seed": 1, "epoch": 5, "uid": "v6"}[differing]
    if differing == "uid":
        other["taus"] = VIDEOS["v6"]
    b = cache.training_rows(other["taus"], other["k"], other["seed"], other["epoch"],
                            other["uid"])
    expected = sample_training_views(
        other["taus"], other["k"],
        random.Random(f"{other['seed']}:{other['epoch']}:{other['uid']}"))
    assert b.tolist() == expected
    if differing != "uid" or len(a) == len(b):
        assert cache.stats()["misses"] == 2, "the two keys collided"


def test_clearing_between_blocks_empties_it():
    cache = ViewSetCache()
    cache.training_rows(VIDEOS["v1"], 9, 0, 1, "v1")
    cache.evaluation_rows(VIDEOS["v1"], 9, "v1")
    assert cache.stats()["held"] == 2
    cache.clear()
    assert cache.stats()["held"] == 0


def test_an_index_tensor_gathers_the_same_rows_as_a_python_list():
    """The cache stores a LongTensor so the gather skips a per-batch list conversion. That is
    only safe if `t[tensor]` and `t[list]` are the same gather -- they are, and a regression
    here would silently reorder or reshape the tokens the aggregator sees."""
    tokens = torch.arange(60, dtype=torch.float32).reshape(12, 5)
    rows = [7, 1, 11, 3]
    assert torch.equal(tokens[rows], tokens[torch.tensor(rows, dtype=torch.long)])


# ============================================ the same property, on the real frame manifest
DATASET = REPO / "data" / "dataset_v2"


@pytest.mark.skipif(not (DATASET / "manifests" / "frames.tsv").exists(),
                    reason="dataset v2 manifests not generated in this checkout")
@pytest.mark.parametrize("k", [1, 9])
def test_real_videos_select_identically_with_and_without_the_cache(k):
    """The synthetic case proves the property; this proves it on the data the run uses.

    Real `taus` are not evenly spaced -- videos differ in length and timestamps are irregular --
    and `phase_indices_window` breaks ties by a circular minimum over that spacing. If a cache
    key were wrong, or if `taus` were not in fact fixed per video, this is where it shows.
    """
    from grapecount.data.views import VideoIndex

    index = VideoIndex(DATASET)
    uids = index.uids_for(set(sorted(index.by_bunch)[:60]))[:200]
    assert uids, "no videos to check"
    cache = ViewSetCache()
    for seed in (0, 2):
        for epoch in (1, 3, 40, 120):
            for uid in uids:
                taus = index.taus(uid)
                expected = sample_training_views(
                    taus, k, random.Random(f"{seed}:{epoch}:{uid}"))
                assert cache.training_rows(taus, k, seed, epoch, uid).tolist() == expected
    for uid in uids:
        taus = index.taus(uid)
        assert (cache.evaluation_rows(taus, k, uid).tolist()
                == evaluation_view_sets(taus, k, m=1)[0])
