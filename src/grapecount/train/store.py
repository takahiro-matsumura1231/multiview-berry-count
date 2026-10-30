"""Share the stage-1 backbone and its feature cache between arms that must have it identical.

`det` and `det_size_silhouette` differ only in which detection columns are concatenated onto
the view token. They train on identical bunches with an identical seed, so they must get the
**identical** backbone: fine-tuning twice would make the arms differ by backbone noise as well
as by feature set, and the ablation would no longer isolate what it claims to.

That it also halves the cost is secondary. The correctness argument is the reason.

Keyed by (backbone, seed, exact training bunch set, image geometry) -- everything that determines the stored
object. Two feature arms hit the same key; a different inner fold misses, as it must, because
under full nesting each fold gets its own backbone.

**The backbone belongs in the key.** Omitting it is correct only while a single backbone is
used. With ResNet18 and ResNet50 as two scored arms it becomes
a silent-corruption bug: the two share a seed and a training bunch set, so ResNet50 would hit
ResNet18's entry and be handed its backbone and feature cache. Nothing raises -- the
aggregator derives `input_dim` from the cache -- and the run completes with ResNet18 features
recorded under the ResNet50 label. `get` therefore takes the backbone name and a cross-backbone
hit is impossible by construction rather than by discipline.
"""

from __future__ import annotations

from collections import OrderedDict


class BackboneStore:
    """Small LRU of (backbone, feature cache), keyed by what determines them.

    Capacity must span one arm pass: the runner walks every inner fold plus the refit for arm
    A before arm B starts, so anything smaller evicts A's backbones before B can reuse them.
    """

    def __init__(self, capacity: int = 2) -> None:
        self.capacity = capacity
        self._items: OrderedDict[tuple, tuple] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(backbone: str, seed: int, bunches: set[str],
            image_shape_hw: tuple[int, int] | None = None) -> tuple:
        return (backbone, seed, frozenset(bunches),
                tuple(image_shape_hw) if image_shape_hw is not None else None)

    def get(self, backbone: str, seed: int, bunches: set[str],
            image_shape_hw: tuple[int, int] | None = None):
        k = self.key(backbone, seed, bunches, image_shape_hw)
        if k in self._items:
            self._items.move_to_end(k)
            self.hits += 1
            return self._items[k]
        self.misses += 1
        return None

    def put(self, backbone: str, seed: int, bunches: set[str], model, cache,
            image_shape_hw: tuple[int, int] | None = None) -> None:
        k = self.key(backbone, seed, bunches, image_shape_hw)
        self._items[k] = (model, cache)
        self._items.move_to_end(k)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "held": len(self._items)}

    def clear(self) -> None:
        self._items.clear()
