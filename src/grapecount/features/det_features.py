"""Per-view features derived from the detector's output.

23 numbers per frame, in six named groups (stored in the cache, so a feature set can be
selected by group name):

    count       n_raw, log1p(n)                     how many berries are visible
    pos         mean_x, mean_y, std_x, std_y        where the berry cloud sits, how spread
    density     mean_nn, median_nn, x_range,        how tightly packed -- occlusion proxy
                y_range, packing
    conf        mean_conf, std_conf                 detector certainty
    size        mean_w, mean_h, mean_area,          apparent berry size
                std_area, median_area, fill_ratio
    silhouette  hull_area, hull_aspect,             the bunch outline from the center hull
                convexity, coverage

Stored RAW. Standardisation is deliberately NOT done here: the mean and standard deviation
must be computed on the training fold only, inside the trainer, or the statistics leak
across the split.

Detector confidence (`conf`) is not used by the model: it correlates with image quality
rather than with berry count.

Each frame row stores 49 numbers: the 23 above over all boxes (columns 0..22), the same
features without confidence over the berry boxes only (23..43), and five global-box values
(44..48: present, cx, cy, w, h). The global box is a single detection that encloses the
whole bunch; it is separated by a label-blind geometric rule (`global_box_index`) so that it
does not inflate the berry statistics. The model uses columns 23..48.

Usage:
    python -m grapecount.features.det_features [--dataset-root data/dataset_v2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError

REPO = Path(__file__).resolve().parents[3]

NAMES = [
    "n_raw", "log_n",
    "mean_x", "mean_y", "std_x", "std_y",
    "mean_nn", "median_nn", "x_range", "y_range", "packing",
    "mean_conf", "std_conf",
    "mean_w", "mean_h", "mean_area", "std_area", "median_area", "fill_ratio",
    "hull_area", "hull_aspect", "convexity", "coverage",
]
GROUPS = {
    "count": [0, 1],
    "pos": [2, 3, 4, 5],
    "density": [6, 7, 8, 9, 10],
    "conf": [11, 12],
    "size": [13, 14, 15, 16, 17, 18],
    "silhouette": [19, 20, 21, 22],
}
BERRY_COLS = [i for i in range(len(NAMES)) if i not in GROUPS["conf"]]
GLOBAL_NAMES = ["present", "cx", "cy", "w", "h"]
ALL_NAMES = (NAMES + [f"berry_{NAMES[i]}" for i in BERRY_COLS]
             + [f"global_{x}" for x in GLOBAL_NAMES])
ALL_GROUPS = {**GROUPS, "berry": list(range(23, 44)), "global": list(range(44, 49))}
GLOBAL_RULE = ("n>=2; max_area>=0.05; max_area>=10*second_area; "
               "contains>=90% of other centers; first index on ties")


def view_features(centers: np.ndarray, conf: np.ndarray, wh: np.ndarray) -> list[float]:
    """One frame -> 23 numbers. All-zero when nothing was detected, which is meaningful:
    a frame with no visible berries genuinely carries no berry evidence."""
    c = np.asarray(centers, np.float32).reshape(-1, 2)
    n = len(c)
    if n == 0:
        return [0.0] * len(NAMES)

    mean_x, mean_y = float(c[:, 0].mean()), float(c[:, 1].mean())
    std_x, std_y = float(c[:, 0].std()), float(c[:, 1].std())
    x_range = float(c[:, 0].ptp()) if n > 1 else 0.0
    y_range = float(c[:, 1].ptp()) if n > 1 else 0.0

    if n >= 2:
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nn = d.min(1)
        mean_nn, median_nn = float(nn.mean()), float(np.median(nn))
    else:
        mean_nn = median_nn = 0.0

    bbox_area = x_range * y_range
    packing = n / bbox_area if bbox_area > 1e-4 else 0.0

    cf = np.asarray(conf, np.float32).reshape(-1)
    mean_conf = float(cf.mean()) if cf.size else 0.0
    std_conf = float(cf.std()) if cf.size else 0.0

    box = np.asarray(wh, np.float32).reshape(-1, 2)
    if box.size:
        areas = box[:, 0] * box[:, 1]
        mean_w, mean_h = float(box[:, 0].mean()), float(box[:, 1].mean())
        mean_area, std_area = float(areas.mean()), float(areas.std())
        median_area = float(np.median(areas))
        total_area = float(areas.sum())
    else:
        mean_w = mean_h = mean_area = std_area = median_area = total_area = 0.0
    fill_ratio = total_area / bbox_area if bbox_area > 1e-4 else 0.0

    hull_area = hull_aspect = convexity = 0.0
    if n >= 3:
        try:
            hull = ConvexHull(c)
            hull_area = float(hull.volume)          # 2-D: `volume` is the enclosed area
            pts = c[hull.vertices]
            w_ = float(pts[:, 0].ptp()) or 1e-6
            h_ = float(pts[:, 1].ptp()) or 1e-6
            hull_aspect = w_ / h_
            convexity = hull_area / bbox_area if bbox_area > 1e-4 else 0.0
        except (QhullError, ValueError):
            pass                                    # collinear berries: no hull, leave zeros
    coverage = total_area / hull_area if hull_area > 1e-6 else 0.0

    return [float(n), float(np.log1p(n)),
            mean_x, mean_y, std_x, std_y,
            mean_nn, median_nn, x_range, y_range, packing,
            mean_conf, std_conf,
            mean_w, mean_h, mean_area, std_area, median_area, fill_ratio,
            hull_area, hull_aspect, convexity, coverage]


def global_box_index(centers: np.ndarray, wh: np.ndarray) -> int | None:
    """Unique dominant enclosing box by area, or None. Ties resolve to the first index."""
    c = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    sizes = np.asarray(wh, dtype=np.float32).reshape(-1, 2)
    if len(c) != len(sizes):
        raise ValueError("centers and wh have different box counts")
    if len(c) < 2:
        return None
    area = sizes[:, 0] * sizes[:, 1]
    if not np.isfinite(c).all() or not np.isfinite(sizes).all() or np.any(sizes < 0):
        raise ValueError("invalid detection geometry")
    rank = np.argsort(-area, kind="stable")
    largest, second = int(rank[0]), float(area[rank[1]])
    if float(area[largest]) < 0.05 or float(area[largest]) < 10.0 * second:
        return None
    others = np.arange(len(c)) != largest
    inside = np.all(np.abs(c[others] - c[largest]) <= sizes[largest] / 2, axis=1)
    if float(inside.mean()) < 0.9:
        return None
    return largest


def frame_features(centers: np.ndarray, conf: np.ndarray,
                   wh: np.ndarray) -> tuple[list[float], int | None]:
    """One frame -> the 49-number row described in the module docstring."""
    c = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    cf = np.asarray(conf, dtype=np.float32).reshape(-1)
    sizes = np.asarray(wh, dtype=np.float32).reshape(-1, 2)
    if not (len(c) == len(cf) == len(sizes)):
        raise ValueError("detection arrays have different box counts")
    global_idx = global_box_index(c, sizes)
    mixed = view_features(c, cf, sizes)
    keep = np.ones(len(c), dtype=bool)
    global_features = [0.0] * 5
    if global_idx is not None:
        keep[global_idx] = False
        global_features = [1.0, *c[global_idx].tolist(), *sizes[global_idx].tolist()]
    berry_full = view_features(c[keep], cf[keep], sizes[keep])
    return mixed + [berry_full[i] for i in BERRY_COLS] + global_features, global_idx


def build(dataset_root: Path, out: Path) -> dict:
    feat_dir = dataset_root / "features"
    shards = sorted(feat_dir.glob("detections_shard*.pt"))
    if not shards:
        raise SystemExit(f"no detection shards in {feat_dir}; "
                         "run grapecount.features.detection first")
    cache: dict[str, dict] = {}
    meta = None
    n_frames = n_global = 0
    for shard in shards:
        blob = torch.load(shard, map_location="cpu", weights_only=False)
        meta = {k: v for k, v in blob.items() if k != "cache"}
        overlap = set(cache) & set(blob["cache"])
        if overlap:
            raise SystemExit(f"shards overlap on {len(overlap)} videos, e.g. {sorted(overlap)[:3]}")
        for uid, entry in blob["cache"].items():
            rows = []
            for c, cf, sizes in zip(entry["centers"], entry["conf"], entry["wh"], strict=True):
                row, global_idx = frame_features(c, cf, sizes)
                rows.append(row)
                n_global += global_idx is not None
            if len(rows) != len(entry["t"]):
                raise ValueError(f"video {uid}: frame count mismatch")
            cache[uid] = {"t": entry["t"], "gt": entry["gt"], "bunch": entry["bunch"],
                          "feats": torch.tensor(rows, dtype=torch.float32).reshape(-1, 49)}
            n_frames += len(rows)
    result = {"cache": cache, "names": ALL_NAMES, "groups": ALL_GROUPS,
              "detector": meta, "standardised": False, "global_rule": GLOBAL_RULE,
              "global_frames": n_global, "frames": n_frames,
              "note": "RAW features; z-score with TRAIN-FOLD statistics inside the trainer"}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=REPO / "data" / "dataset_v2")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.dataset_root / "features" / "det_features.pt"
    result = build(args.dataset_root, out)
    print(f"[detfeat] {len(result['cache'])} videos, {result['frames']} frames, "
          f"{result['global_frames']} global-box frames -> {out}")


if __name__ == "__main__":
    main()
