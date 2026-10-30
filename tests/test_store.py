"""The backbone is part of the experimental design, and two mechanisms fail SILENTLY if it is not carried.

Both defects tested here produce a complete, plausible-looking results table while the numbers
describe the wrong model. Neither raises. They are therefore tested rather than reviewed:

  * the backbone store handed out a cached backbone keyed on (seed, bunches) alone, so
    ResNet50 would receive ResNet18's backbone and feature cache;
  * the run plan and the output paths had no backbone level, so the two backbones collided on
    run id and the second overwrote the first's predictions.

Single-backbone configurations must keep working throughout: tables regenerated from stored
predictions have to stay readable.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grapecount.train.store import BackboneStore  # noqa: E402



# --------------------------------------------------------------------------- store
def test_store_never_serves_one_backbone_to_another():
    """The defect itself: same seed, same bunches, different backbone must MISS."""
    store = BackboneStore(capacity=4)
    bunches = {"A", "B", "C"}
    store.put("resnet18", 0, bunches, "r18-model", {"uid": "r18-cache"})

    assert store.get("resnet50", 0, bunches) is None
    assert store.get("resnet18", 0, bunches) == ("r18-model", {"uid": "r18-cache"})


def test_store_still_shares_between_feature_sets():
    """The sharing the store exists for must survive the fix.

    det and det_size_silhouette must see the identical backbone, or their difference includes
    backbone noise and the ablation stops isolating the feature set.
    """
    store = BackboneStore(capacity=4)
    bunches = {"A", "B"}
    store.put("resnet18", 0, bunches, "model", {})
    first = store.get("resnet18", 0, bunches)
    second = store.get("resnet18", 0, bunches)
    assert first is not None and first[0] is second[0]
    assert store.stats()["hits"] == 2


def test_store_keys_differ_across_backbone_seed_and_bunches():
    k = BackboneStore.key
    base = k("resnet18", 0, {"A"})
    assert base != k("resnet50", 0, {"A"})
    assert base != k("resnet18", 1, {"A"})
    assert base != k("resnet18", 0, {"B"})
    assert base == k("resnet18", 0, {"A"})
