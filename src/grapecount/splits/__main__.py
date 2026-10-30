"""Generate the nested split from the split config.

    python -m grapecount.splits --config configs/split.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import yaml

from .access import load
from .build import build, write

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "split.yaml")
    ap.add_argument("--dataset-root", type=Path, default=REPO / "data" / "dataset_v2")
    args = ap.parse_args()

    text = args.config.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    sp = cfg["split"]

    with (args.dataset_root / "manifests" / "videos.tsv").open(newline="", encoding="utf-8") as fh:
        videos = list(csv.DictReader(fh, delimiter="\t"))

    outer_fold_of, inner_fold_of, meta = build(
        videos, n_outer=sp["outer_folds"], n_inner=sp["inner_folds"], seed=sp["seed"],
        count_bands=sp["stratify"]["count_bands"],
        stage_classes=sp["stratify"]["distinct_gt_classes"],
    )
    hashes = write(args.dataset_root / "splits", videos, outer_fold_of, inner_fold_of,
                   meta, sp["outer_folds"], sp["inner_folds"])

    split = load(args.dataset_root)
    problems = split.check()
    audit = {
        "protocol_version": cfg["version"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "protocol_semantic_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")).encode()).hexdigest(),
        "split_sha256": hashes,
        "n_bunches": len(outer_fold_of),
        "n_videos": len(videos),
        "outer_folds": sp["outer_folds"],
        "inner_folds": sp["inner_folds"],
        "seed": sp["seed"],
        "invariant_violations": problems,
    }
    out = args.dataset_root / "splits" / "audit.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[splits] {args.dataset_root / 'splits'}")
    print(f"  bunches={len(outer_fold_of)} videos={len(videos)} "
          f"outer={sp['outer_folds']} inner={sp['inner_folds']} seed={sp['seed']}")
    for o in range(sp["outer_folds"]):
        tr, te = split.outer_split(o)
        print(f"  outer {o}: train {len(tr):4d} bunches / test {len(te):4d} bunches")
    print(f"  invariant violations: {len(problems)}")
    for p in problems:
        print(f"    !! {p}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
