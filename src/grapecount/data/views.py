"""Datasets that turn a bunch set into K-view training and evaluation samples.

The sampling asymmetry is deliberate and is the part most easily got wrong:

    training     a fresh random phase every epoch, with the partial-orbit mixture. The same
                 video yields different view-sets across epochs, which is the augmentation.
    evaluation   M fixed view-sets at phases i/M, identical on every call. Nothing is
                 sampled, so a re-run reproduces and the M predictions can be median-
                 aggregated to one vote per video without introducing variance.

A sample is a *stage-video*, never a frame and never a view-set: that is the evaluation
unit, and building batches from anything finer would let one video contribute many times.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..views.sampling import evaluation_view_sets, sample_training_views, tau_of

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(size: int | tuple[int, int], train: bool) -> transforms.Compose:
    """Photometric jitter only for training. No geometric augmentation: the orbit geometry
    carries the multi-view signal, and flipping or cropping would corrupt the relationship
    between views that the aggregator is meant to exploit."""
    shape_hw = (size, size) if isinstance(size, int) else tuple(size)
    steps = [transforms.Resize(shape_hw)]
    if train:
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class VideoIndex:
    """Frames and detection features per stage-video, aligned by frame index."""

    def __init__(self, dataset_root: Path, det_features: Path | None = None) -> None:
        self.root = dataset_root
        with (dataset_root / "manifests" / "videos.tsv").open(newline="", encoding="utf-8") as fh:
            self.videos = {v["video_uid"]: v for v in csv.DictReader(fh, delimiter="\t")}
        self.frames: dict[str, list[dict]] = defaultdict(list)
        with (dataset_root / "manifests" / "frames.tsv").open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                self.frames[r["video_uid"]].append(r)
        for rows in self.frames.values():
            rows.sort(key=lambda r: int(r["frame_idx"]))

        self.det: dict[str, torch.Tensor] = {}
        self.det_names: list[str] = []
        self.det_groups: dict[str, list[int]] = {}
        if det_features is not None:
            blob = torch.load(det_features, map_location="cpu", weights_only=False)
            self.det_names, self.det_groups = blob["names"], blob["groups"]
            for uid, entry in blob["cache"].items():
                got, want = list(entry["t"]), [int(r["timestamp_ms"]) for r in self.frames[uid]]
                if got != want:
                    raise ValueError(f"video {uid}: detection features are not aligned with "
                                     f"the frame manifest ({len(got)} vs {len(want)} rows)")
                self.det[uid] = entry["feats"]

        self.by_bunch: dict[str, list[str]] = defaultdict(list)
        for uid, v in self.videos.items():
            self.by_bunch[v["bunch_id"]].append(uid)

    def uids_for(self, bunches: set[str]) -> list[str]:
        return sorted((u for b in sorted(bunches) for u in self.by_bunch.get(b, [])), key=int)

    def taus(self, uid: str) -> list[float]:
        return tau_of([int(r["timestamp_ms"]) for r in self.frames[uid]])

    def det_columns(self, feature_set: str) -> list[int]:
        """Column indices of a named detection-feature set, resolved from the stored groups.

        `berry_global` is the model's input: the 21 berry-box features and the five
        global-box values (columns 23..48 of the cache written by det_features).
        """
        sets = {
            "berry_global": ["berry", "global"],
            "berry": ["berry"],
            "all_boxes": ["count", "pos", "density", "size", "silhouette"],
        }
        if feature_set not in sets:
            raise ValueError(f"unknown feature set {feature_set!r}; known: {sorted(sets)}")
        cols: list[int] = []
        for g in sets[feature_set]:
            if g not in self.det_groups:
                raise ValueError(f"feature set {feature_set!r} needs group {g!r}, which is not "
                                 f"in the cache (have: {sorted(self.det_groups)})")
            cols.extend(self.det_groups[g])
        return sorted(cols)


class ViewSetDataset(Dataset):
    """One item per stage-video: K views of images, and optionally their det features."""

    def __init__(self, index: VideoIndex, bunches: set[str], k: int,
                 image_size: int | tuple[int, int],
                 train: bool, seed: int = 0, det_columns: list[int] | None = None) -> None:
        self.index = index
        self.uids = index.uids_for(bunches)
        self.k = k
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.det_columns = det_columns
        self.transform = build_transform(image_size, train)
        if not self.uids:
            raise ValueError("empty bunch set")

    def set_epoch(self, epoch: int) -> None:
        """Re-seeds training view selection so each epoch sees different arcs, reproducibly."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.uids)

    def _choose(self, uid: str) -> list[int]:
        taus = self.index.taus(uid)
        if not self.train:
            return evaluation_view_sets(taus, self.k, m=1)[0]
        rng = random.Random(f"{self.seed}:{self.epoch}:{uid}")
        return sample_training_views(taus, self.k, rng)

    def _load(self, uid: str, rows: list[int]) -> tuple[torch.Tensor, torch.Tensor | None]:
        frames = self.index.frames[uid]
        imgs = []
        for i in rows:
            with Image.open(self.index.root / frames[i]["frame_path"]) as im:
                imgs.append(self.transform(im.convert("RGB")))
        views = torch.stack(imgs)
        det = None
        if self.det_columns is not None:
            det = self.index.det[uid][rows][:, self.det_columns]
        return views, det

    def __getitem__(self, i: int):
        uid = self.uids[i]
        views, det = self._load(uid, self._choose(uid))
        gt = torch.tensor(float(self.index.videos[uid]["gt_count"]))
        if det is None:
            return views, gt, uid
        return views, det, gt, uid


def collate(batch):
    """Default collation, with the uid list kept as strings rather than tensorised."""
    if len(batch[0]) == 3:
        views, gt, uid = zip(*batch)
        return torch.stack(views), torch.stack(gt), list(uid)
    views, det, gt, uid = zip(*batch)
    return torch.stack(views), torch.stack(det), torch.stack(gt), list(uid)
