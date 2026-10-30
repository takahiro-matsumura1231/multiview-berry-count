"""Read the nested split.

The API is deliberately narrow. `outer_split` is the only way to obtain the outer-test
bunches, and it is the only function a training routine must never call for selection;
`inner_split` cannot return them at all, because the inner table has no row for a bunch that
is in that outer fold's test set.

That asymmetry is the point. A common failure is making the evaluation fold reachable from
the training loop, so that the best epoch is picked on it. Here, selecting on
outer-test requires deliberately calling the wrong function.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


@dataclass(frozen=True)
class NestedSplit:
    outer_fold_of: dict[str, int]
    inner_fold_of: dict[tuple[int, str], int]
    n_outer: int
    n_inner: int

    # ------------------------------------------------------------------ outer
    def outer_test(self, outer: int) -> set[str]:
        """Scored exactly once, at the end of outer run `outer`. Never used to select."""
        return {b for b, o in self.outer_fold_of.items() if o == outer}

    def outer_train(self, outer: int) -> set[str]:
        return {b for b, o in self.outer_fold_of.items() if o != outer}

    def outer_split(self, outer: int) -> tuple[set[str], set[str]]:
        return self.outer_train(outer), self.outer_test(outer)

    # ------------------------------------------------------------------ inner
    def inner_split(self, outer: int, inner: int) -> tuple[set[str], set[str]]:
        """(train, val) inside outer-train. Everything selectable is decided here."""
        train, val = set(), set()
        for (o, b), i in self.inner_fold_of.items():
            if o != outer:
                continue
            (val if i == inner else train).add(b)
        return train, val

    # ------------------------------------------------------------ invariants
    def check(self) -> list[str]:
        """Return a list of violated invariants; empty means the split is sound."""
        problems: list[str] = []

        for o in range(self.n_outer):
            test = self.outer_test(o)
            train = self.outer_train(o)
            if test & train:
                problems.append(f"outer {o}: {len(test & train)} bunches in both train and test")
            if not test:
                problems.append(f"outer {o}: empty test set")

            inner_bunches = {b for (oo, b) in self.inner_fold_of if oo == o}
            leaked = inner_bunches & test
            if leaked:
                problems.append(
                    f"outer {o}: {len(leaked)} outer-test bunches have an inner assignment "
                    f"and are therefore reachable from model selection: {sorted(leaked)[:5]}")
            missing = train - inner_bunches
            if missing:
                problems.append(f"outer {o}: {len(missing)} train bunches have no inner fold")

            for i in range(self.n_inner):
                tr, va = self.inner_split(o, i)
                if tr & va:
                    problems.append(f"outer {o} inner {i}: train and val overlap")
                if not va:
                    problems.append(f"outer {o} inner {i}: empty val set")
                if (tr | va) & test:
                    problems.append(f"outer {o} inner {i}: touches outer-test")
        return problems


def load(dataset_root: Path) -> NestedSplit:
    splits = dataset_root / "splits"
    outer_rows = _read(splits / "bunch_outer.tsv")
    inner_rows = _read(splits / "bunch_inner.tsv")
    outer_fold_of = {r["bunch_id"]: int(r["outer_fold"]) for r in outer_rows}
    inner_fold_of = {(int(r["outer_fold"]), r["bunch_id"]): int(r["inner_fold"])
                     for r in inner_rows}
    return NestedSplit(
        outer_fold_of=outer_fold_of,
        inner_fold_of=inner_fold_of,
        n_outer=len(set(outer_fold_of.values())),
        n_inner=len({i for i in inner_fold_of.values()}),
    )


def videos_of(videos: list[dict], bunches: set[str]) -> list[dict]:
    """Stage-videos belonging to a set of bunches. The evaluation unit is the video."""
    return [v for v in videos if v["bunch_id"] in bunches]


def group_by_bunch(rows: list[dict], key: str = "bunch_id") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return dict(out)
