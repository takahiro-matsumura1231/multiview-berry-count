"""The real two-stage trainer, behind the same `Trainer` contract as the reference models.

    fit(train, val)   stage 1: fine-tune the backbone on `train`
                      stage 2: freeze it, cache features, train the aggregator, and return
                               its per-epoch trace on `val`
    fit(train, None)  the refit: both stages on the full outer-train, for a fixed number of
                      aggregator epochs, returning nothing to select on
    predict(bunches)  M deterministic view-sets per video, median-aggregated to one vote

The runner cannot tell this apart from `ConstantMean`, which is the point: whatever the
reference-trainer tests verify about the run path holds here unchanged. Nothing in this file reads the split, the
outer fold, or anything beyond the bunch sets it is handed.

Two things are easy to get wrong and are handled explicitly:

* **Feature standardisation is fitted on training bunches only.** Fitting it on everything
  would leak the evaluation fold's distribution into training.
* **The feature cache is rebuilt per fit.** It is a function of the backbone, and the
  backbone differs for every inner fold under full nesting, so a cache reused across folds
  would silently carry another fold's fitting.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import protocol
from ..data.views import ViewSetDataset, VideoIndex, build_transform, collate
from ..models import BackboneCounter, FeatureScaler, ViewTokenCounter, parameter_groups
from ..views.sampling import evaluation_view_sets
from .runner import EpochScore, Prediction


class _FrameSet(torch.utils.data.Dataset):
    """Flat list of every frame of the given videos, in manifest order, for cache building."""

    def __init__(self, index: VideoIndex, uids: list[str],
                 image_size: int | tuple[int, int]) -> None:
        self.index = index
        self.transform = build_transform(image_size, train=False)
        self.rows = [(uid, r) for uid in uids for r in index.frames[uid]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> torch.Tensor:
        from PIL import Image
        _uid, row = self.rows[i]
        with Image.open(self.index.root / row["frame_path"]) as im:
            return self.transform(im.convert("RGB"))


def warmup_cosine(optimizer, total_steps: int, warmup_frac: float = 0.1):
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)


class TwoStageTrainer:
    """A CNN backbone @448 fine-tuned, then kept fixed, with a view-token transformer on cached features.

    `backbone` names the backbone; with it unset the first configured backbone is used.
    `name` identifies the cell (backbone crossed with feature set) and is what lands in the
    `method` column of predictions.tsv. `positional_encoding` adds view-slot embeddings to the
    stage-2 transformer.
    """

    def __init__(self, index: VideoIndex, cfg: dict, arm: str, device: str = "cuda",
                 workers: int = 8, log=print, store=None, backbone: str | None = None,
                 name: str | None = None, positional_encoding: bool = False) -> None:
        research = cfg["model"]["research"]
        # Shared between feature sets so that both get the IDENTICAL backbone; see store.py.
        self.store = store
        self.index = index
        self.arm = arm
        self.device = device
        self.workers = workers
        self.log = log

        self.k = int(research["views_per_set"])
        self.positional_encoding = bool(positional_encoding)
        if "image_shape_hw" in research:
            shape = research["image_shape_hw"]
            if (not isinstance(shape, (list, tuple)) or len(shape) != 2
                    or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
                           for n in shape)):
                raise ValueError("model.research.image_shape_hw must be [positive height, positive width]")
            self.image_shape_hw = tuple(shape)
        else:
            size = int(research["resolution"])
            if size <= 0:
                raise ValueError("model.research.resolution must be positive")
            self.image_shape_hw = (size, size)
        self.image_size = self.image_shape_hw
        self.backbone_name = backbone or protocol.backbones(cfg)[0]
        self.name = name or arm
        self.backbone_epochs = int(research["backbone_epochs"])
        self.backbone_batch = int(research["backbone_batch"])
        self.lr_backbone = float(research["lr_backbone"])
        self.lr_head = float(research["lr_head"])
        self.agg_max_epochs = int(research["aggregator_max_epochs"])
        self.agg_batch = int(research["aggregator_batch"])
        self.agg_lr = float(research["aggregator_lr"])
        self.agg_wd = float(research["aggregator_weight_decay"])
        self.m_view_sets = int(cfg["evaluation"]["view_sets"])
        self.det_columns = index.det_columns(arm)

        self._backbone: BackboneCounter | None = None
        self._aggregator: ViewTokenCounter | None = None
        self._scaler: FeatureScaler | None = None

    # ------------------------------------------------------------------ stage 1
    def _finetune_backbone(self, bunches: set[str], seed: int) -> BackboneCounter:
        torch.manual_seed(seed)
        model = BackboneCounter(self.backbone_name).to(self.device)
        ds = ViewSetDataset(self.index, bunches, self.k, self.image_size, train=True, seed=seed)
        loader = DataLoader(ds, batch_size=self.backbone_batch, shuffle=True,
                            num_workers=self.workers, collate_fn=collate, drop_last=True,
                            pin_memory=True, persistent_workers=self.workers > 0)
        opt = torch.optim.AdamW(parameter_groups(model, self.lr_backbone, self.lr_head),
                                weight_decay=1e-4)
        sched = warmup_cosine(opt, self.backbone_epochs * max(1, len(loader)))
        loss_fn = nn.L1Loss()
        scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")

        model.train()
        for epoch in range(self.backbone_epochs):
            ds.set_epoch(epoch)
            total, n, t0 = 0.0, 0, time.time()
            for views, gt, _ in loader:
                views, gt = views.to(self.device, non_blocking=True), gt.to(self.device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.device == "cuda"):
                    loss = loss_fn(model(views), gt)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.item() * len(gt)
                n += len(gt)
            self.log(f"      [stage1] epoch {epoch + 1}/{self.backbone_epochs} "
                     f"train L1 {total / max(n, 1):.3f} ({time.time() - t0:.0f}s)")
        model.eval()
        return model

    # ------------------------------------------------------------------ cache
    @torch.no_grad()
    def _cache_features(self, model: BackboneCounter, uids: list[str]) -> dict[str, torch.Tensor]:
        """Per-frame backbone features for every frame of the given videos.

        Rebuilt for every fit, because it is a function of THIS fold's backbone -- under full
        nesting the backbone differs for every inner fold, so a cache carried across folds
        would silently contain another fold's fitting.

        JPEG decode dominates, not the GPU: a first version loaded images serially in the
        main process and spent 95% of its time there. Decoding runs in worker processes and
        the features are stored fp16, which halves a cache that reaches several GB per fold.
        """
        loader = DataLoader(
            _FrameSet(self.index, uids, self.image_size),
            batch_size=128, shuffle=False, num_workers=self.workers,
            pin_memory=True, persistent_workers=False,
        )
        chunks: list[torch.Tensor] = []
        for images in loader:
            with torch.amp.autocast("cuda", enabled=self.device == "cuda"):
                chunks.append(model.encode(images.to(self.device, non_blocking=True)).half().cpu())
        flat = torch.cat(chunks)

        cache: dict[str, torch.Tensor] = {}
        at = 0
        for uid in uids:
            n = len(self.index.frames[uid])
            cache[uid] = flat[at:at + n]
            at += n
        return cache

    def _tokens(self, cache: dict[str, torch.Tensor], uid: str,
                rows: list[int]) -> torch.Tensor:
        img = cache[uid][rows].float()
        det = self.index.det[uid][rows][:, self.det_columns]
        return torch.cat([img, det], dim=1)

    # ------------------------------------------------------------------ stage 2
    def _train_aggregator(self, cache: dict, train_uids: list[str], val_uids: list[str],
                          seed: int, max_epochs: int) -> tuple[ViewTokenCounter, list[EpochScore]]:
        torch.manual_seed(seed + 1)
        import random as _random

        # standardisation fitted on TRAINING bunches only
        train_rows = torch.cat([self._tokens(cache, u, list(range(len(cache[u]))))
                                for u in train_uids])
        self._scaler = FeatureScaler().fit(train_rows)
        dim = train_rows.shape[1]
        del train_rows

        model = ViewTokenCounter(dim, positional_encoding=self.positional_encoding,
                                 view_slots=self.k).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.agg_lr, weight_decay=self.agg_wd)
        sched = warmup_cosine(opt, max_epochs)
        loss_fn = nn.L1Loss()

        def batch_for(uids: list[str], epoch: int, train: bool):
            tokens, gts = [], []
            for uid in uids:
                taus = self.index.taus(uid)
                if train:
                    from ..views.sampling import sample_training_views
                    rows = sample_training_views(taus, self.k, _random.Random(f"{seed}:{epoch}:{uid}"))
                else:
                    rows = evaluation_view_sets(taus, self.k, m=1)[0]
                tokens.append(self._scaler.transform(self._tokens(cache, uid, rows)))
                gts.append(float(self.index.videos[uid]["gt_count"]))
            return torch.stack(tokens), torch.tensor(gts)

        trace: list[EpochScore] = []
        for epoch in range(1, max_epochs + 1):
            model.train()
            order = list(train_uids)
            _random.Random(f"{seed}:order:{epoch}").shuffle(order)
            for i in range(0, len(order), self.agg_batch):
                x, y = batch_for(order[i:i + self.agg_batch], epoch, train=True)
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            if val_uids:
                model.eval()
                errs = []
                with torch.no_grad():
                    for i in range(0, len(val_uids), self.agg_batch):
                        x, y = batch_for(val_uids[i:i + self.agg_batch], epoch, train=False)
                        pred = model(x.to(self.device)).cpu()
                        errs.append((pred - y).abs())
                e = torch.cat(errs)
                trace.append(EpochScore(epoch=epoch, mae=float(e.mean()),
                                        rmse=float((e ** 2).mean().sqrt())))
        return model, trace

    # ------------------------------------------------------------- the contract
    def fit(self, train_bunches: set[str], val_bunches: set[str] | None,
            seed: int, max_epochs: int | None = None) -> list[EpochScore]:
        t0 = time.time()
        train_uids = self.index.uids_for(train_bunches)
        val_uids = self.index.uids_for(val_bunches) if val_bunches else []
        self.log(f"    [fit] {len(train_uids)} train videos, {len(val_uids)} val, "
                 f"cell={self.name} seed={seed}"
                 + (f", refit to epoch {max_epochs}" if val_bunches is None else ""))

        cached = (self.store.get(self.backbone_name, seed, train_bunches,
                                 image_shape_hw=self.image_shape_hw)
                  if self.store is not None else None)
        if cached is not None:
            backbone, cache = cached
            missing = [u for u in train_uids + val_uids if u not in cache]
            if missing:
                cache = {**cache, **self._cache_features(backbone, missing)}
            self.log(f"    [fit] reusing the {self.backbone_name} backbone from the other "
                     f"feature arm")
        else:
            backbone = self._finetune_backbone(train_bunches, seed)
            cache = self._cache_features(backbone, train_uids + val_uids)
            if self.store is not None:
                self.store.put(self.backbone_name, seed, train_bunches, backbone, cache,
                               image_shape_hw=self.image_shape_hw)
        epochs = max_epochs or self.agg_max_epochs
        aggregator, trace = self._train_aggregator(cache, train_uids, val_uids, seed, epochs)

        self._backbone, self._aggregator = backbone, aggregator
        self.log(f"    [fit] done in {(time.time() - t0) / 60:.1f} min"
                 + (f", best inner-val MAE {min(t.mae for t in trace):.3f}" if trace else ""))
        return trace

    @torch.no_grad()
    def predict(self, bunches: set[str], seed: int) -> list[Prediction]:
        """M deterministic view-sets per video, median-aggregated to one vote."""
        if self._backbone is None or self._aggregator is None or self._scaler is None:
            raise RuntimeError("predict() before fit()")
        uids = self.index.uids_for(bunches)
        cache = self._cache_features(self._backbone, uids)
        self._aggregator.eval()

        out: list[Prediction] = []
        for uid in uids:
            taus = self.index.taus(uid)
            sets = evaluation_view_sets(taus, self.k, self.m_view_sets)
            x = torch.stack([self._scaler.transform(self._tokens(cache, uid, rows))
                             for rows in sets]).to(self.device)
            preds = self._aggregator(x).cpu()
            v = self.index.videos[uid]
            out.append(Prediction(video_uid=uid, bunch_id=v["bunch_id"],
                                  gt=float(v["gt_count"]), pred=float(preds.median()),
                                  outer_fold=-1, seed=seed, method=self.name))
        return out
