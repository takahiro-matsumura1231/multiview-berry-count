"""Run the berry detector over every dataset frame.

The YOLOv8 detector is an external pretrained module. Its training data and class-name
mapping are not documented in the available local artifacts. A separate bunch-total count regressor is
deliberately **not** used as a feature, because that output directly predicts the target
being regressed here.

The detector is fixed and fold-independent, so this runs once over all frames
rather than per fold. Nothing here touches the split.

Output: one `.pt` per shard, keyed by video_uid:

    {uid: {"t": [ms...], "gt": float, "bunch": str,
           "n_det": LongTensor[N],
           "centers": [FloatTensor[n_i, 2]],   # normalised to [0,1] in the ORIGINAL frame
           "conf":    [FloatTensor[n_i]],
           "wh":      [FloatTensor[n_i, 2]]}}  # box size, normalised by frame W,H

Usage:
    python -m grapecount.features.detection --shard 0 --n-shards 2   # one per GPU
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path(os.environ.get("GRAPECOUNT_DETECTOR", REPO / "models/yolov8.torchscript"))
IMGSZ = 640
PAD_VALUE = 114


def letterbox(im: Image.Image, size: int = IMGSZ):
    """Aspect-preserving resize onto a square grey canvas, the way YOLO expects.

    Returns the tensor plus what is needed to undo it, so detections come back in
    original-frame coordinates rather than letterbox pixels.
    """
    w, h = im.size
    scale = size / max(w, h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = Image.new("RGB", (size, size), (PAD_VALUE,) * 3)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(im.resize((nw, nh)), (pad_x, pad_y))
    tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
    return tensor, (scale, pad_x, pad_y, w, h)


@torch.no_grad()
def decode(raw: torch.Tensor, metas: list, conf_thr: float, iou_thr: float):
    """Raw YOLOv8 output (B, 4+C, A) -> per-image detections in normalised frame coords.

    NMS is class-agnostic, matching the cached detections used in the original pipeline.
    The model emits two class scores, but its class-name mapping is not verified here.
    Class IDs are discarded by this decoder; verify the class mapping before interpreting
    every returned box as an individual berry.
    """
    out = []
    for b, (scale, pad_x, pad_y, w, h) in enumerate(metas):
        p = raw[b].T                                  # (anchors, 4 + n_classes)
        scores = p[:, 4:]
        conf, cls = scores.max(1)
        keep = conf >= conf_thr
        if keep.sum() == 0:
            out.append((torch.zeros(0, 2), torch.zeros(0), torch.zeros(0, 2)))
            continue
        box, conf, cls = p[keep, :4], conf[keep], cls[keep]
        cx, cy, bw, bh = box.T
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        nms = torchvision.ops.nms(xyxy, conf, iou_thr)
        cx, cy, bw, bh, conf = cx[nms], cy[nms], bw[nms], bh[nms], conf[nms]

        # letterbox pixels -> original pixels -> [0,1]
        centers = torch.stack([((cx - pad_x) / scale) / w, ((cy - pad_y) / scale) / h], 1)
        wh = torch.stack([(bw / scale) / w, (bh / scale) / h], 1)
        out.append((centers.cpu(), conf.cpu(), wh.cpu()))
    return out


def load_frames(dataset_root: Path) -> tuple[dict, dict]:
    with (dataset_root / "manifests" / "videos.tsv").open(newline="", encoding="utf-8") as fh:
        videos = {v["video_uid"]: v for v in csv.DictReader(fh, delimiter="\t")}
    frames: dict[str, list[dict]] = defaultdict(list)
    with (dataset_root / "manifests" / "frames.tsv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            frames[r["video_uid"]].append(r)
    for rows in frames.values():
        rows.sort(key=lambda r: int(r["frame_idx"]))
    return videos, frames


def detect(dataset_root: Path, model_path: Path = DEFAULT_MODEL, out_dir: Path | None = None,
           conf: float = 0.25, iou: float = 0.45, batch_size: int = 64, shard: int = 0,
           n_shards: int = 1, device: str | None = None) -> Path:
    """Run the detector on every frame of one shard of a dataset; returns the shard path."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = out_dir or dataset_root / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"detections_shard{shard}of{n_shards}.pt"
    if not Path(model_path).is_file():
        raise SystemExit(f"detector weights not found: {model_path} "
                         "(set GRAPECOUNT_DETECTOR or pass --model)")

    videos, frames = load_frames(dataset_root)
    uids = sorted(frames, key=int)[shard::n_shards]
    total_frames = sum(len(frames[u]) for u in uids)
    print(f"[det] shard {shard}/{n_shards}: {len(uids)} videos, "
          f"{total_frames} frames, conf>={conf} iou={iou} on {device}", flush=True)

    model = torch.jit.load(str(model_path), map_location=device).eval()

    cache: dict[str, dict] = {}
    done = 0
    t0 = time.time()
    for uid in uids:
        rows = frames[uid]
        centers_l, conf_l, wh_l, n_det = [], [], [], []
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            tensors, metas = [], []
            for r in chunk:
                with Image.open(dataset_root / r["frame_path"]) as im:
                    t, meta = letterbox(im.convert("RGB"))
                tensors.append(t)
                metas.append(meta)
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                raw = model(batch)
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
            for centers, box_conf, wh in decode(raw.float(), metas, conf, iou):
                centers_l.append(centers)
                conf_l.append(box_conf)
                wh_l.append(wh)
                n_det.append(len(centers))
            done += len(chunk)

        v = videos[uid]
        cache[uid] = {"t": [int(r["timestamp_ms"]) for r in rows],
                      "gt": float(v["gt_count"] or "nan"), "bunch": v["bunch_id"],
                      "n_det": torch.tensor(n_det, dtype=torch.long),
                      "centers": centers_l, "conf": conf_l, "wh": wh_l}
        if len(cache) % 200 == 0:
            rate = done / (time.time() - t0)
            print(f"[det] {len(cache)}/{len(uids)} videos, {done}/{total_frames} frames, "
                  f"{rate:.0f} fps, eta {(total_frames - done) / max(rate, 1e-9) / 60:.1f} min",
                  flush=True)

    torch.save({"cache": cache, "conf_thr": conf, "iou_thr": iou,
                "model": Path(model_path).name, "imgsz": IMGSZ}, out_path)
    if not cache:
        print(f"[det] wrote {out_path} (no videos in this shard)")
        return out_path
    counts = torch.cat([c["n_det"] for c in cache.values()]).float()
    print(f"[det] wrote {out_path} ({len(cache)} videos, {done} frames, "
          f"{(time.time() - t0) / 60:.1f} min)")
    print(f"[det] berries/frame: mean {counts.mean():.1f} median {counts.median():.0f} "
          f"max {counts.max():.0f} zero-detection frames {(counts == 0).sum()}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=REPO / "data" / "dataset_v2")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    detect(args.dataset_root, args.model, args.out_dir, args.conf, args.iou, args.batch,
           args.shard, args.n_shards)


if __name__ == "__main__":
    main()
