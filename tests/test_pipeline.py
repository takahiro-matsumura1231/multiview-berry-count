"""End-to-end verification of the run path, before any real run happens.

Split, selection, evaluation, storage and aggregation are all verified *before* formal data
is pushed through them. These tests exercise the whole
chain on the real dataset split with cheap reference models, so a defect in the plumbing
surfaces here rather than after expensive fine-tunes.

The seal is verified by being **tripped**, not merely by existing. A guard that has never
fired is not known to work.

    python -m pytest tests/test_pipeline.py -q
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grapecount.splits.access import load  # noqa: E402
from grapecount.train import (  # noqa: E402
    EpochScore, SealedBunches, SealViolation, run_outer_fold, select_epoch,
    write_predictions, write_run_manifest,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_trainers import ConstantMean, LeakyTrainer, NoisyOracle  # noqa: E402
from grapecount.views import evaluation_view_sets, phase_indices, tau_of  # noqa: E402

DATASET = REPO / "data" / "dataset_v2"
pytestmark = pytest.mark.skipif(
    not (DATASET / "splits" / "bunch_outer.tsv").exists(),
    reason="dataset v2 splits not generated in this checkout",
)


@pytest.fixture(scope="module")
def videos():
    with (DATASET / "manifests" / "videos.tsv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


@pytest.fixture(scope="module")
def split():
    return load(DATASET)


# ===================================================================== the seal
def test_reading_the_seal_before_selection_raises():
    seal = SealedBunches({"A0", "A1"})
    with pytest.raises(SealViolation):
        seal.get()


@pytest.mark.parametrize("attempt", [len, lambda s: "A0" in s, lambda s: list(iter(s))])
def test_the_sneaky_routes_are_closed(attempt):
    """len(), membership and iteration would each leak the test set without calling get()."""
    seal = SealedBunches({"A0", "A1"})
    with pytest.raises(SealViolation):
        attempt(seal)


def test_unsealing_requires_a_reason_and_works_once():
    seal = SealedBunches({"A0"})
    with pytest.raises(ValueError):
        seal.unseal("")
    seal.unseal("selection complete")
    assert seal.get() == {"A0"}
    with pytest.raises(SealViolation):
        seal.unseal("again")          # a fold is scored once, not twice


def test_a_trainer_that_peeks_is_caught(videos, split):
    """The test-fold peeking failure, reproduced deliberately: it must not be silent."""
    seal = SealedBunches(split.outer_test(0))
    leaky = LeakyTrainer(videos, seal)
    with pytest.raises(SealViolation):
        leaky.fit(train_bunches={"A0"}, val_bunches={"A1"}, seed=0)


# =========================================================== selection isolation
def test_select_epoch_cannot_receive_an_outer_score():
    """Structural, not behavioural: there is no parameter for it."""
    import inspect

    params = list(inspect.signature(select_epoch).parameters)
    assert params == ["inner_traces"]


def test_select_epoch_picks_the_mean_minimum():
    traces = [
        [EpochScore(1, 5.0), EpochScore(2, 3.0), EpochScore(3, 4.0)],
        [EpochScore(1, 5.0), EpochScore(2, 3.5), EpochScore(3, 4.0)],
    ]
    epoch, info = select_epoch(traces)
    assert epoch == 2
    assert info["chosen_mean_mae"] == pytest.approx(3.25)


def test_select_epoch_breaks_ties_toward_fewer_epochs():
    traces = [[EpochScore(1, 3.0), EpochScore(2, 3.0)]] * 2
    assert select_epoch(traces)[0] == 1


def test_select_epoch_rejects_an_empty_inner_fold():
    with pytest.raises(ValueError):
        select_epoch([[EpochScore(1, 1.0)], []])


# ================================================================ view sampling
def test_evaluation_view_sets_are_deterministic():
    taus = [i / 40 for i in range(41)]
    a = evaluation_view_sets(taus, k=9, m=5)
    b = evaluation_view_sets(taus, k=9, m=5)
    assert a == b
    assert len(a) == 5 and all(len(s) == 9 for s in a)


def test_view_sets_do_not_repeat_a_frame_when_the_pool_allows():
    taus = [i / 40 for i in range(41)]
    for s in evaluation_view_sets(taus, k=9, m=5):
        assert len(set(s)) == 9


def test_phase_distance_wraps_around_the_orbit():
    """tau 0.99 and 0.01 are adjacent viewpoints, not opposite ones."""
    taus = [0.0, 0.25, 0.5, 0.75, 0.99]
    assert phase_indices(taus, k=1, phase=0.01)[0] in (0, 4)


def test_tau_is_normalised_within_the_video():
    assert tau_of([2000, 3000, 4000]) == [0.0, 0.5, 1.0]


# =========================================================== end to end, real split
@pytest.fixture(scope="module")
def oracle_results(videos, split):
    trainer = NoisyOracle(videos, noise=2.0)
    return [run_outer_fold(o, split, trainer, seed=0, method=trainer.name)
            for o in range(split.n_outer)]


def test_every_outer_fold_is_scored_exactly_once(oracle_results):
    for r in oracle_results:
        assert r.seal_audit["sealed"] is False
        assert r.seal_audit["n_accesses"] == 1


def test_predictions_cover_every_video_exactly_once(oracle_results, videos):
    records = [{"method": p.method, "seed": p.seed, "outer_fold": r.outer_fold,
                "video_uid": p.video_uid, "bunch": p.bunch_id, "gt": p.gt, "pred": p.pred}
               for r in oracle_results for p in r.predictions]
    uids = [r["video_uid"] for r in records]
    assert len(uids) == len(set(uids)) == len(videos)
    assert set(uids) == {v["video_uid"] for v in videos}


def test_the_runner_stamps_fold_identity_not_the_trainer(oracle_results, split):
    """Regression: reference trainers return outer_fold=-1 and the runner must overwrite it.

    Left unstamped, every prediction landed in a single fold. Coverage still passed, the
    pooled MAE was still correct, and only the per-fold breakdown and fold SD were quietly
    destroyed -- the kind of defect that survives to publication.
    """
    for r in oracle_results:
        assert all(p.outer_fold == r.outer_fold for p in r.predictions)
    assert {r.outer_fold for r in oracle_results} == set(range(split.n_outer))
    assert -1 not in {p.outer_fold for r in oracle_results for p in r.predictions}


def test_predictions_for_foreign_bunches_are_rejected(videos, split):
    """A trainer that returns rows outside outer-test must fail loudly, not pollute the pool."""
    class Overreaching(ConstantMean):
        name = "ref_overreaching"

        def predict(self, bunches, seed):
            return super().predict(set(self.table.by_bunch), seed)   # everything

    with pytest.raises(RuntimeError, match="not in the outer-test set"):
        run_outer_fold(0, split, Overreaching(videos), seed=0, method="x")


def test_predictions_are_out_of_fold(oracle_results, split, videos):
    """A prediction must come from the fold that held its bunch out."""
    for r in oracle_results:
        test_bunches = split.outer_test(r.outer_fold)
        assert {p.bunch_id for p in r.predictions} <= test_bunches


def test_rerunning_with_the_same_seed_reproduces_exactly(videos, split):
    a = run_outer_fold(0, split, NoisyOracle(videos, 2.0), seed=0, method="x")
    b = run_outer_fold(0, split, NoisyOracle(videos, 2.0), seed=0, method="x")
    assert a.chosen_epoch == b.chosen_epoch
    assert [(p.video_uid, p.pred) for p in a.predictions] == \
           [(p.video_uid, p.pred) for p in b.predictions]


def test_a_different_seed_changes_the_predictions(videos, split):
    a = run_outer_fold(0, split, NoisyOracle(videos, 2.0), seed=0, method="x")
    b = run_outer_fold(0, split, NoisyOracle(videos, 2.0), seed=1, method="x")
    assert [p.pred for p in a.predictions] != [p.pred for p in b.predictions]


# ============================================================ storage and report
def test_predictions_file_is_lf_and_has_the_required_columns(tmp_path, videos, split):
    results = [run_outer_fold(o, split, ConstantMean(videos), seed=0, method="c")
               for o in range(split.n_outer)]
    path = tmp_path / "p.tsv"
    write_predictions(path, results)
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    header = raw.decode().splitlines()[0].split("\t")
    assert header == ["method", "seed", "outer_fold", "video_uid", "bunch_id", "gt", "pred"]


def test_the_run_manifest_records_the_seal_and_the_selection(tmp_path, videos, split):
    import json

    results = [run_outer_fold(0, split, ConstantMean(videos), seed=0, method="c")]
    path = tmp_path / "m.json"
    write_run_manifest(path, results, {"protocol": "test-run"})
    payload = json.loads(path.read_text())
    run = payload["outer_runs"][0]
    assert run["seal"]["sealed"] is False and run["seal"]["n_accesses"] == 1
    assert run["selection"]["criterion"] == "mean inner-val MAE"
    assert run["chosen_epoch"] >= 1
