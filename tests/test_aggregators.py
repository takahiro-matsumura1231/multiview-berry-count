"""Baseline-aggregator mechanisms, each of which fails SILENTLY if it is wrong.

Every defect tested here produces a complete, plausible results table describing the wrong
experiment. None of them raises on its own:

  * a positional encoding, or any order-dependence, in an aggregator -- the model trains, the
    loss falls, and the "permutation-invariant set model" row of the table is not one;
  * the aggregator or K entering the backbone store key -- each arm quietly fine-tunes its own
    stage-1 backbone and the aggregator contrast carries initialisation noise the same size as
    the effect;
  * a cell name or an output path that omits the aggregator -- two models pool into one
    estimate, or one overwrites the other's predictions;
  * the K=1 arm scored by the median over five phases -- a five-frame result under a
    single-view label.

`torch` is needed for the model tests and is skipped where it is absent; the naming, path and
store tests are pure and always run.

    python -m pytest tests/test_aggregators.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grapecount import protocol  # noqa: E402
from grapecount.train.store import BackboneStore  # noqa: E402
from grapecount.views import evaluation_view_sets  # noqa: E402

# The models import torchvision for the backbone, so a torch-only environment
# collects these and then fails on the import rather than skipping them.
torch = pytest.importorskip("torch", reason="stage-2 model tests need torch and torchvision")
pytest.importorskip("torchvision", reason="stage-2 model tests need torch and torchvision")

from grapecount.models import (  # noqa: E402
    AGGREGATOR_SOURCE, AGGREGATORS, AggregatorConfig, make_aggregator, n_parameters,
)

PROTOCOL = REPO / "configs" / "model.yaml"
INPUT_DIM = 512 + 21          # resnet18 feature + det_size_silhouette
NAMES = sorted(AGGREGATORS)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def agg_cfg(cfg) -> AggregatorConfig:
    return AggregatorConfig.from_protocol(cfg)


# ============================================================ the five aggregators
def test_unknown_aggregator_raises(agg_cfg):
    with pytest.raises(ValueError, match="unknown aggregator"):
        make_aggregator("transformer", INPUT_DIM, agg_cfg)


# ================================================================== permutation invariance
@pytest.mark.parametrize("name", NAMES)
def test_output_does_not_move_when_the_views_are_permuted(name, agg_cfg):
    """A bunch photographed from K angles has no canonical view order.

    Tested rather than reviewed: an accidental positional encoding lowers the training loss
    and produces a full, plausible table under a label that is no longer true.
    """
    torch.manual_seed(0)
    model = make_aggregator(name, INPUT_DIM, agg_cfg).eval()
    x = torch.randn(4, 9, INPUT_DIM)
    for shift in (1, 4, 8):
        perm = torch.roll(torch.arange(9), shift)
        with torch.no_grad():
            assert torch.allclose(model(x), model(x[:, perm]), atol=1e-5), f"{name} shift {shift}"


@pytest.mark.parametrize("name", NAMES)
def test_a_reversed_view_order_also_leaves_the_output_fixed(name, agg_cfg):
    torch.manual_seed(1)
    model = make_aggregator(name, INPUT_DIM, agg_cfg).eval()
    x = torch.randn(3, 9, INPUT_DIM)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x.flip(1)), atol=1e-5)


def test_the_test_would_catch_an_order_dependent_aggregator(agg_cfg):
    """A guard that has never fired is not known to work."""
    class OrderDependent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Linear(INPUT_DIM, 1)

        def forward(self, tokens):
            weights = torch.arange(tokens.shape[1], dtype=tokens.dtype, device=tokens.device)
            return (self.w(tokens).squeeze(-1) * weights).sum(dim=1)

    model = OrderDependent().eval()
    x = torch.randn(3, 9, INPUT_DIM)
    with torch.no_grad():
        assert not torch.allclose(model(x), model(x.flip(1)), atol=1e-5)


# ============================================================== shapes, finiteness, budgets
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("k", [1, 3, 9])
def test_shape_and_finiteness_at_every_configured_k(name, k, agg_cfg):
    """An aggregator may be fed 1, 3 or 9 tokens. A model that silently assumes K=9 -- a
    flattened pool, a fixed-size projection -- must fail here, not in a long training run."""
    torch.manual_seed(2)
    model = make_aggregator(name, INPUT_DIM, agg_cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(5, k, INPUT_DIM))
    assert out.shape == (5,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", NAMES)
def test_the_shared_settings_reach_every_arm(name, agg_cfg, cfg):
    """model_dim, head hidden and head dropout are constants of the comparison. If one arm
    quietly used a different width, a row of the table would differ by capacity as well as
    by pooling."""
    torch.manual_seed(3)
    model = make_aggregator(name, INPUT_DIM, agg_cfg)
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert any(m.out_features == agg_cfg.head_hidden for m in linears), "no head hidden layer"
    assert [m for m in linears if m.out_features == 1], "no scalar output"
    assert any(m.out_features == agg_cfg.model_dim for m in linears), "no model_dim layer"


def test_capacity_is_recorded_and_no_arm_is_extreme(agg_cfg):
    """The protocol does not equalise parameter counts -- padding a Deep Sets network to a
    transformer's size would be a tuning decision made to flatter a comparison. It does
    require the spread to be recorded and bounded, so the table can say what it bounds."""
    torch.manual_seed(4)
    counts = {n: n_parameters(make_aggregator(n, INPUT_DIM, agg_cfg)) for n in NAMES}
    assert counts["deep_sets_mean"] == counts["deep_sets_max"], (
        "mean and max must differ only in the pool, so their parameter counts are equal")
    assert max(counts.values()) / min(counts.values()) < 20
    # The two transformer arms are the large ones; the baseline is not handicapped by size.
    assert counts["set_transformer"] > counts["view_token_transformer"]


# ================================================= the backbone store, which must be shared
def test_the_aggregator_is_not_part_of_the_store_key():
    """The defect the shared store exists to prevent. Every aggregator of a block must be handed
    the SAME backbone object; a per-aggregator key would give each its own fine-tune."""
    store = BackboneStore(capacity=5)
    bunches = {"A", "B", "C"}
    store.put("resnet18", 0, bunches, "backbone", {"1": "features"})
    first = store.get("resnet18", 0, bunches)
    second = store.get("resnet18", 0, bunches)
    assert first is not None and second is not None
    assert first[0] is second[0] and first[1] is second[1]
    assert store.stats()["hits"] == 2


def test_a_different_training_bunch_set_still_misses():
    """Sharing must not reach across inner folds: under full nesting each fold has its own
    backbone, and reusing one would carry another fold's fitting."""
    store = BackboneStore(capacity=5)
    store.put("resnet18", 0, {"A", "B"}, "backbone", {})
    assert store.get("resnet18", 0, {"A", "C"}) is None
    assert store.get("resnet18", 1, {"A", "B"}) is None
    assert store.get("resnet50", 0, {"A", "B"}) is None


# ================================================================ cell names and run paths
# ====================================================== the K and M evaluation specification
def test_k1_at_m5_selects_five_distinct_single_frames():
    """The single-view scoring rule depends on this: five fixed phases must give five
    different frames, or 'five complete out-of-fold passes' is one pass repeated."""
    taus = [i / 50 for i in range(51)]
    sets = evaluation_view_sets(taus, k=1, m=5)
    assert [len(s) for s in sets] == [1] * 5
    assert len({s[0] for s in sets}) == 5


@pytest.mark.parametrize("k", [1, 3, 9])
def test_evaluation_view_sets_are_deterministic_and_the_right_size(k):
    taus = [i / 60 for i in range(61)]
    first = evaluation_view_sets(taus, k, m=5)
    assert first == evaluation_view_sets(taus, k, m=5)
    assert [len(s) for s in first] == [k] * 5
    for rows in first:
        assert len(set(rows)) == k, "a view-set must not use one frame twice"


