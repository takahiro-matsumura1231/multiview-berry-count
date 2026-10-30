#!/usr/bin/env python3
"""Count berries in your own videos with trained checkpoints.

Put the videos (.mp4 / .mov) in one directory, any depth. Two naming modes:

  labeled     <bunch>_<stage>_gt_<count>.mp4, e.g. B1_0_gt_60.mp4. The count is read from
              the file name and MAE / RMSE are reported. This is the training format.
  unlabeled   any file name (--unlabeled). Only predictions are written.

The videos go through the same pipeline as training -- frame extraction (configs/dataset.yaml),
berry detection, detection features -- into <work>/dataset, and every checkpoint predicts
every video. With several checkpoints (e.g. all seeds x outer folds of one training run) the
reported prediction is their mean; each per-checkpoint prediction is the median over the
5 fixed view-sets of the video.

Example:
    PYTHONPATH=src python scripts/predict.py --videos my_videos/ \\
        --checkpoints results/model/runs --out predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grapecount.data.build import build_manifests, extract_frames  # noqa: E402
from grapecount.data.spec import DEFAULT_SPEC, load_spec  # noqa: E402
from grapecount.data.views import VideoIndex  # noqa: E402
from grapecount.features import det_features  # noqa: E402
from grapecount.features.detection import DEFAULT_MODEL, detect  # noqa: E402
from grapecount.train.multiview import load_checkpoint  # noqa: E402


def find_checkpoints(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for p in paths:
        found += [p] if p.is_file() else sorted(p.rglob("checkpoint.pt"))
    if not found:
        raise SystemExit(f"no checkpoint.pt under {', '.join(map(str, paths))}")
    return found


def input_stamp(args) -> dict:
    """What the prepared dataset was built from; any change forces a rebuild."""
    videos = args.videos.resolve()
    files = sorted((str(p.relative_to(videos)), p.stat().st_size)
                   for p in videos.rglob("*") if p.is_file())
    return {"videos": str(videos), "unlabeled": args.unlabeled,
            "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
            "files_sha256": hashlib.sha256(json.dumps(files).encode()).hexdigest()}


def prepare_dataset(args) -> Path:
    root = args.work / "dataset"
    stamp_path = args.work / "input_stamp.json"
    stamp = input_stamp(args)
    ready = (root / "features" / "det_features.pt").is_file() and stamp_path.is_file()
    if ready and not args.rebuild and json.loads(stamp_path.read_text()) == stamp:
        print(f"[predict] reusing prepared dataset {root}", flush=True)
        return root
    if root.exists():
        shutil.rmtree(root)           # never mix frames or features of different inputs
    stamp_path.unlink(missing_ok=True)
    spec = load_spec(args.spec, raw_root=args.videos, output_root=root,
                     unlabeled=args.unlabeled)
    if not spec.raw_root.is_dir():
        raise SystemExit(f"video directory not found: {spec.raw_root}")
    admitted, res, _audit = build_manifests(spec, args.workers)
    for reason, paths in sorted(res.excluded.items()):
        for rel in paths:
            print(f"[predict] skipped ({reason}): {rel}", flush=True)
    if not admitted:
        raise SystemExit("no video was admitted; check the file names "
                         "(use --unlabeled for names without _gt_<count>)")
    extract_frames(spec, args.workers)
    detect(root, args.detector, device=args.device)
    det_features.build(root, root / "features" / "det_features.pt")
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, required=True, help="directory of input videos")
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                    help="checkpoint.pt files or directories searched for them")
    ap.add_argument("--out", type=Path, default=Path("predictions.csv"))
    ap.add_argument("--work", type=Path, default=None,
                    help="working directory for frames and features (default: <out>.work)")
    ap.add_argument("--unlabeled", action="store_true",
                    help="accept any file name; no ground truth")
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--detector", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--rebuild", action="store_true", help="re-extract frames and features")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.work = args.work or args.out.with_suffix(".work")

    checkpoints = find_checkpoints(args.checkpoints)
    root = prepare_dataset(args)
    index = VideoIndex(root, det_features=root / "features" / "det_features.pt")
    bunches = set(index.by_bunch)

    per_ckpt: dict[str, list[float]] = {u: [] for u in index.videos}
    for n, path in enumerate(checkpoints):
        trainer = load_checkpoint(path, index, device=args.device)
        for p in trainer.predict(bunches, seed=0):
            per_ckpt[p.video_uid].append(p.pred)
        print(f"[predict] checkpoint {n + 1}/{len(checkpoints)}: {path}", flush=True)
        del trainer

    rows, errors = [], []
    for uid in sorted(index.videos, key=int):
        v = index.videos[uid]
        preds = per_ckpt[uid]
        pred = sum(preds) / len(preds)
        gt = float(v["gt_count"]) if v["gt_count"] not in ("", "nan") else math.nan
        if not math.isnan(gt):
            errors.append(pred - gt)
        rows.append({"video": v["raw_video_path"], "gt": "" if math.isnan(gt) else int(gt),
                     "pred": round(pred, 3), "n_checkpoints": len(preds),
                     "pred_min": round(min(preds), 3), "pred_max": round(max(preds), 3)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[predict] {len(rows)} videos -> {args.out}")
    if errors:
        mae = sum(abs(e) for e in errors) / len(errors)
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        print(f"[predict] labeled videos {len(errors)}: MAE {mae:.3f}  RMSE {rmse:.3f}")


if __name__ == "__main__":
    main()
