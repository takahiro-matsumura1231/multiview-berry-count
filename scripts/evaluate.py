#!/usr/bin/env python3
"""Score the outer-test predictions written by scripts/train.py.

Every video is predicted once per seed, by the model of the outer fold that held it out.
Reported, for each seed and for the mean over seeds (seed ensemble):

  * accuracy: MAE, RMSE, MAPE, bias, share within 1/3/5 berries;
  * counting fidelity on bunches filmed at >= 3 distinct counts (thinning series);
  * a 95% percentile interval of MAE from a bootstrap that resamples whole bunches.

Example:
    PYTHONPATH=src python scripts/evaluate.py --runs results/model/runs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grapecount.metrics.fidelity import evaluate  # noqa: E402
from grapecount.metrics.interval import cluster_bootstrap  # noqa: E402


def mae(records: list[dict]) -> float:
    return mean(abs(r["pred"] - r["gt"]) for r in records)


def score(records: list[dict], n_boot: int) -> dict:
    metrics, _ = evaluate(records)
    metrics["MAE_95ci_bunch_bootstrap"] = cluster_bootstrap(records, mae, n_resamples=n_boot)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=REPO / "results/model/runs")
    ap.add_argument("--out", type=Path, default=None, help="default: <runs>/../evaluation.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    by_seed: dict[int, dict[str, dict]] = defaultdict(dict)
    files = sorted(args.runs.glob("seed*/outer*/predictions.tsv"))
    if not files:
        raise SystemExit(f"no predictions.tsv under {args.runs}")
    for path in files:
        with path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                seed, uid = int(r["seed"]), r["video_uid"]
                if uid in by_seed[seed]:
                    raise SystemExit(f"video {uid} predicted twice for seed {seed}")
                by_seed[seed][uid] = {"video_uid": uid, "bunch": r["bunch_id"],
                                      "gt": float(r["gt"]), "pred": float(r["pred"])}

    report: dict = {"n_prediction_files": len(files), "per_seed": {}}
    for seed in sorted(by_seed):
        report["per_seed"][seed] = score(list(by_seed[seed].values()), args.n_boot)

    common = set.intersection(*(set(v) for v in by_seed.values()))
    if any(len(v) != len(common) for v in by_seed.values()):
        print("[evaluate] warning: seeds cover different videos; the ensemble uses the "
              f"{len(common)} videos common to all seeds")
    ensemble = []
    for uid in sorted(common, key=int):
        rows = [by_seed[s][uid] for s in by_seed]
        ensemble.append({**rows[0], "pred": mean(r["pred"] for r in rows)})
    report["seed_ensemble"] = score(ensemble, args.n_boot)
    report["seed_mean"] = {k: mean(report["per_seed"][s][k] for s in by_seed)
                           for k in ("MAE", "RMSE")}

    out = args.out or args.runs.parent / "evaluation.json"
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    e = report["seed_ensemble"]
    ci = e["MAE_95ci_bunch_bootstrap"]
    print(f"[evaluate] {len(by_seed)} seeds, {e['n']} videos")
    for seed, m in report["per_seed"].items():
        print(f"  seed {seed}: MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}")
    print(f"  seed ensemble: MAE {e['MAE']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  "
          f"RMSE {e['RMSE']:.3f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
