#!/usr/bin/env python3
"""Train the multi-view counter with nested bunch-level cross-validation.

For every (seed, outer fold): the epoch is chosen on the inner folds, the model is refit on
the whole outer-train set at that epoch, and the outer-test videos are predicted once. Each
scored unit is written as

    <out>/runs/seed{s}/outer{o}/checkpoint.pt         every inference component
    <out>/runs/seed{s}/outer{o}/predictions.tsv       outer-test predictions (+ 5 phases)
    <out>/runs/seed{s}/outer{o}/inner_predictions.tsv inner-validation predictions per epoch
    <out>/runs/seed{s}/outer{o}/run_manifest.json     provenance; written last

The checkpoint is re-read in a separate process and one test video is re-predicted before
the unit is published. A unit with run_manifest.json is complete and is skipped on resume;
Stage 1 backbones are kept under <out>/stage1/ so a resumed unit does not fine-tune again.
A resumed run must use the same config and dataset; this is checked by hash.

Example:
    PYTHONPATH=src python scripts/train.py --dataset-root data/dataset_v2 --out results/model
    PYTHONPATH=src python scripts/train.py --seeds 0 --outer 0      # one unit only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grapecount.data.views import VideoIndex  # noqa: E402
from grapecount.splits.access import load as load_split  # noqa: E402
from grapecount.train.baselines import ViewSetCache  # noqa: E402
from grapecount.train.multiview import (CHECKPOINT_FORMAT, MultiViewTrainer,  # noqa: E402
                                        atomic_new, load_checkpoint, sha256_file)
from grapecount.train.runner import run_outer_fold  # noqa: E402
from grapecount.train.store import BackboneStore  # noqa: E402

UNIT_FILES = ("checkpoint.pt", "predictions.tsv", "inner_predictions.tsv", "run_manifest.json")


INPUT_FILES = ("manifests/videos.tsv", "manifests/frames.tsv",
               "splits/bunch_outer.tsv", "splits/bunch_inner.tsv")


def inputs_fingerprint(args) -> dict:
    """Hashes of everything a unit depends on; a resumed run must match them exactly."""
    files = {name: sha256_file(args.dataset_root / name) for name in INPUT_FILES}
    files["det_features"] = sha256_file(args.det_features)
    files["config"] = sha256_file(args.config)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def unit_done(unit: Path, fingerprint: str) -> bool:
    present = [(unit / f).is_file() for f in UNIT_FILES]
    if all(present):
        rec = json.loads((unit / "run_manifest.json").read_text())
        if rec.get("inputs_sha256") != fingerprint:
            raise RuntimeError(f"{unit} was trained with a different config or dataset; "
                               "use a new --out")
        for name, digest in rec["artifacts"].items():
            if sha256_file(unit / name) != digest:
                raise RuntimeError(f"artifact changed after publication: {unit / name}")
        return True
    if any(present):
        raise RuntimeError(f"incomplete unit (move it aside and re-run): {unit}")
    return False


def write_predictions(path: Path, predictions, m: int) -> None:
    def writer(stream):
        fields = (["method", "seed", "outer_fold", "video_uid", "bunch_id", "gt", "pred"]
                  + [f"pred_phase{i}" for i in range(m)])
        lines = ["\t".join(fields)]
        for p in sorted(predictions, key=lambda p: int(p.video_uid)):
            lines.append("\t".join(str(x) for x in (
                p.method, p.seed, p.outer_fold, p.video_uid, p.bunch_id, p.gt, p.pred,
                *p.phase_preds)))
        stream.write(("\n".join(lines) + "\n").encode("utf-8"))
    atomic_new(path, writer)


def verify_main(args) -> None:
    """Child process: reload a checkpoint and re-predict one video."""
    index = VideoIndex(args.dataset_root, det_features=args.det_features)
    trainer = load_checkpoint(args.verify_checkpoint, index, device=args.device)
    bunch = index.videos[args.video]["bunch_id"]
    got = [p for p in trainer.predict({bunch}, seed=0) if p.video_uid == args.video]
    if len(got) != 1:
        raise SystemExit(f"video {args.video} not reproduced")
    print(json.dumps({"video_uid": args.video, "reload_pred": got[0].pred,
                      "absolute_delta": abs(got[0].pred - args.expected)}))


def reload_in_subprocess(args, checkpoint: Path, video_uid: str, expected: float) -> dict:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    cmd = [sys.executable, str(Path(__file__).resolve()), "--verify-checkpoint", str(checkpoint),
           "--video", video_uid, "--expected", repr(expected),
           "--dataset-root", str(args.dataset_root), "--det-features", str(args.det_features),
           "--device", args.device]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if out.returncode != 0 or not lines:
        raise RuntimeError(f"checkpoint reload failed for {checkpoint}: {out.stderr[-1500:]}")
    result = json.loads(lines[-1])
    if not (math.isfinite(result["absolute_delta"]) and result["absolute_delta"] <= 0.01):
        raise RuntimeError(f"reloaded checkpoint differs by {result['absolute_delta']}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/model.yaml")
    ap.add_argument("--dataset-root", type=Path, default=REPO / "data/dataset_v2")
    ap.add_argument("--det-features", type=Path, default=None,
                    help="default: <dataset-root>/features/det_features.pt")
    ap.add_argument("--out", type=Path, default=REPO / "results/model")
    ap.add_argument("--seeds", nargs="+", type=int, help="default: training.seeds of the config")
    ap.add_argument("--outer", nargs="+", type=int, help="default: every outer fold")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verify-checkpoint", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--video", help=argparse.SUPPRESS)
    ap.add_argument("--expected", type=float, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.det_features = args.det_features or args.dataset_root / "features/det_features.pt"
    if args.verify_checkpoint:
        verify_main(args)
        return

    cfg = yaml.safe_load(args.config.read_text())
    index = VideoIndex(args.dataset_root, det_features=args.det_features)
    split = load_split(args.dataset_root)
    seeds = args.seeds if args.seeds is not None else cfg["training"]["seeds"]
    outers = args.outer if args.outer is not None else list(range(split.n_outer))
    inputs = inputs_fingerprint(args)
    print(f"[train] {len(index.videos)} videos, seeds {seeds}, outer folds {outers}, "
          f"inputs {inputs['sha256'][:12]}", flush=True)

    for seed in seeds:
        for outer in outers:
            unit = args.out / f"runs/seed{seed}/outer{outer}"
            if unit_done(unit, inputs["sha256"]):
                print(f"[train] seed {seed} outer {outer}: complete, skipped", flush=True)
                continue
            trainer = MultiViewTrainer(
                index, cfg, device=args.device, workers=args.workers,
                store=BackboneStore(capacity=split.n_inner + 1), view_cache=ViewSetCache(),
                # Keyed by the input fingerprint, so a changed config or dataset never
                # re-reads a backbone fine-tuned under the old one.
                stage1_dir=args.out / f"stage1/{inputs['sha256'][:16]}/seed{seed}/outer{outer}")
            result = run_outer_fold(outer, split, trainer, seed=seed, method=trainer.name)

            provenance = {"seed": seed, "outer_fold": outer,
                          "chosen_epoch": result.chosen_epoch,
                          "inputs_sha256": inputs["sha256"]}
            blob = trainer.checkpoint_blob(cfg, provenance)
            atomic_new(unit / "checkpoint.pt", lambda s: torch.save(blob, s))
            first = min(result.predictions, key=lambda p: int(p.video_uid))
            reload = reload_in_subprocess(args, unit / "checkpoint.pt",
                                          first.video_uid, first.pred)
            write_predictions(unit / "predictions.tsv", result.predictions,
                              trainer.m_view_sets)
            n_inner = trainer.write_inner_predictions(unit / "inner_predictions.tsv")
            manifest = {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "model": trainer.name, "seed": seed, "outer_fold": outer,
                "chosen_epoch": result.chosen_epoch, "selection": result.selection,
                "seal": result.seal_audit, "n_predictions": len(result.predictions),
                "n_inner_prediction_rows": n_inner, "seconds": round(result.seconds, 2),
                "aggregator_parameters": trainer.n_aggregator_parameters,
                "fit_log": trainer.fit_log, "stage1_files": trainer.stage1_files,
                "checkpoint_format": CHECKPOINT_FORMAT, "checkpoint_reload": reload,
                "inputs_sha256": inputs["sha256"], "inputs": inputs["files"],
                "torch": torch.__version__, "device": args.device,
                "artifacts": {n: sha256_file(unit / n) for n in UNIT_FILES[:-1]},
            }
            atomic_new(unit / "run_manifest.json",
                       lambda s: s.write((json.dumps(manifest, indent=2) + "\n").encode()))
            print(f"[train] seed {seed} outer {outer}: epoch {result.chosen_epoch}, "
                  f"{len(result.predictions)} test videos -> {unit}", flush=True)
            del trainer
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
