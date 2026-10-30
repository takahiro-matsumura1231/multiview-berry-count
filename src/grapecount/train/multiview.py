"""Trainer for the multi-view counter: View-token Transformer with view-position encoding.

Stage 1 is `BaselineTrainer`'s: the ResNet-18 backbone is fine-tuned on single frames and
then frozen, and its 512-d image features are cached. Stage 2 trains the aggregator on
K=9 ordered views, each token being [image feature (512), detection features (26)],
standardised with statistics of the current fit's training bunches only.

Weights are never held only in memory: every Stage 1 backbone is written atomically and
re-read in a separate process before training continues, and the scored model is saved
as one self-contained checkpoint (`checkpoint_blob`) that `load_checkpoint` restores.
"""

from __future__ import annotations

import hashlib
import math
import os
import random as _random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from ..data.views import VideoIndex
from ..models import BackboneCounter, FeatureScaler, ViewTokenCounter, n_parameters
from ..views.sampling import evaluation_view_sets
from .baselines import BaselineTrainer, PhasePrediction, digest_state_dict
from .pipeline import warmup_cosine
from .runner import EpochScore

CHECKPOINT_FORMAT = "grapecount_multiview_v1"
STAGE1_FORMAT = "grapecount_stage1_v1"
MODEL_NAME = "view_token_transformer_pos"
FEATURE_SET = "berry_global"
VIEWS = 9


def build_model(input_dim: int, cfg: dict) -> nn.Module:
    """The only constructor. Call right after `torch.manual_seed(seed + 1)`.

    Width, depth, heads and dropouts come from `model.research.aggregator_common`; the
    feed-forward width is 2 x model_dim and the head's hidden width is 256.
    """
    common = cfg["model"]["research"]["aggregator_common"]
    return ViewTokenCounter(input_dim, model_dim=int(common["model_dim"]),
                            layers=int(common["layers"]), heads=int(common["heads"]),
                            dropout=float(common["dropout"]),
                            head_dropout=float(common["head_dropout"]),
                            positional_encoding=True, view_slots=VIEWS)


# ------------------------------------------------------------------ atomic file helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_new(path: Path, writer) -> None:
    """Write to a temp file, fsync, then hard-link into place; never replaces a file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def verify_backbone_subprocess(path: Path, expected_digest: str) -> None:
    """Re-read a saved Stage 1 backbone in a separate process and compare its digest."""
    code = ("import sys, torch; from grapecount.models import BackboneCounter; "
            "from grapecount.train.baselines import digest_state_dict; "
            "b = torch.load(sys.argv[1], map_location='cpu', weights_only=True); "
            "m = BackboneCounter(b['backbone_name'], pretrained=False); "
            "m.load_state_dict(b['backbone_state'], strict=True); "
            "print(digest_state_dict(m))")
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    out = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True,
                         text=True, env=env, check=False)
    got = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    if out.returncode != 0 or got != expected_digest:
        raise RuntimeError(f"Stage 1 backbone reload failed for {path}: "
                           f"rc={out.returncode} digest={got!r} stderr={out.stderr[-500:]}")


def _gt(video: dict) -> float:
    value = video.get("gt_count", "")
    return float(value) if value not in ("", None) else math.nan


# ------------------------------------------------------------------------- trainer
class MultiViewTrainer(BaselineTrainer):
    """Stage 1 inherited from `BaselineTrainer`; stage 2 is the View-token Transformer."""

    def __init__(self, index: VideoIndex, cfg: dict, device: str = "cuda", workers: int = 8,
                 log=print, store=None, view_cache=None, stage1_dir: Path | None = None) -> None:
        super().__init__(index, cfg, aggregator=MODEL_NAME, agg_k=VIEWS, device=device,
                         workers=workers, log=log, store=store, name=MODEL_NAME,
                         view_cache=view_cache)
        self.cfg = cfg
        if self.k != VIEWS:
            raise ValueError(f"Stage 1 K={self.k} must equal the model's K={VIEWS}")
        if self.arm != FEATURE_SET or len(self.det_columns) != 26:
            raise ValueError(f"the model expects the 26-column {FEATURE_SET!r} feature set")
        self.stage1_dir = Path(stage1_dir) if stage1_dir else None
        self.inner_records: list[tuple] = []          # (fit_no, epoch, uid, gt, pred)
        self.stage1_files: list[dict] = []
        self._fit_no = -1
        self._stage = ""

    # ------------------------------------------------------------ stage 1 persistence
    def fit(self, train_bunches, val_bunches, seed, max_epochs=None):
        self._fit_no += 1
        self._stage = "refit" if val_bunches is None else f"inner{self._fit_no}"
        trace = super().fit(train_bunches, val_bunches, seed, max_epochs)
        self.fit_log[-1]["fit_no"] = self._fit_no
        self.fit_log[-1]["stage_name"] = self._stage
        return trace

    def _finetune_backbone(self, bunches, seed):
        bunch_sha = hashlib.sha256("\n".join(sorted(bunches)).encode()).hexdigest()
        path = self.stage1_dir / self._stage / "backbone.pt" if self.stage1_dir else None
        if path is not None and path.exists():
            # A resumed run re-reads the saved Stage 1 instead of fine-tuning again, after
            # checking it is the fit this run needs.
            blob = torch.load(path, map_location="cpu", weights_only=True)
            if (blob.get("format") != STAGE1_FORMAT or blob["seed"] != seed
                    or blob["stage"] != self._stage or blob["k"] != self.k
                    or blob["train_bunches_sha256"] != bunch_sha
                    or blob["backbone_name"] != self.backbone_name):
                raise RuntimeError(f"saved Stage 1 does not match this fit: {path}")
            backbone = BackboneCounter(self.backbone_name, pretrained=False).to(self.device)
            backbone.load_state_dict(blob["backbone_state"], strict=True)
            backbone.eval()
            if digest_state_dict(backbone) != blob["backbone_sha256"]:
                raise RuntimeError(f"saved Stage 1 digest differs: {path}")
            self.stage1_files.append({"stage": self._stage, "path": str(path),
                                      "file_sha256": sha256_file(path),
                                      "backbone_sha256": blob["backbone_sha256"],
                                      "reloaded": True})
            self.log(f"      [stage1] re-read saved backbone {path}")
            return backbone
        backbone = super()._finetune_backbone(bunches, seed)
        if path is not None:
            digest = digest_state_dict(backbone)
            blob = {"format": STAGE1_FORMAT, "backbone_name": self.backbone_name,
                    "backbone_state": {k: v.detach().cpu()
                                       for k, v in backbone.state_dict().items()},
                    "seed": seed, "stage": self._stage, "k": self.k,
                    "train_bunches_sha256": bunch_sha, "n_train_bunches": len(bunches),
                    "backbone_sha256": digest}
            atomic_new(path, lambda s: torch.save(blob, s))
            verify_backbone_subprocess(path, digest)
            self.stage1_files.append({"stage": self._stage, "path": str(path),
                                      "file_sha256": sha256_file(path),
                                      "backbone_sha256": digest})
            self.log(f"      [stage1] saved and re-read {path}")
        return backbone

    # ------------------------------------------------------------------ inputs
    def _tokens_for(self, uid: str, rows: torch.Tensor, data: dict | None, cache) -> torch.Tensor:
        if data is not None and uid in data:
            return data[uid][rows]
        return self._scaler.transform(self._tokens(cache, uid, rows.tolist()))

    # ------------------------------------------------------------------ stage 2
    def _train_aggregator(self, cache, train_uids, val_uids, seed, max_epochs):
        torch.manual_seed(seed + 1)
        all_frames = lambda u: list(range(len(cache[u])))
        train_rows = torch.cat([self._tokens(cache, u, all_frames(u)) for u in train_uids])
        self._scaler = FeatureScaler().fit(train_rows)
        dim = train_rows.shape[1]
        del train_rows
        data = {u: self._scaler.transform(self._tokens(cache, u, all_frames(u)))
                for u in train_uids + val_uids}
        model = build_model(dim, self.cfg).to(self.device)
        self.n_aggregator_parameters = n_parameters(model)
        opt = torch.optim.AdamW(model.parameters(), lr=self.agg_lr, weight_decay=self.agg_wd)
        sched = warmup_cosine(opt, max_epochs)
        loss_fn = nn.L1Loss()
        gt = {u: float(self.index.videos[u]["gt_count"]) for u in train_uids + val_uids}

        def batch_for(uids, epoch, train):
            toks = []
            for uid in uids:
                taus = self.taus(uid)
                rows = (self.view_cache.training_rows(taus, self.agg_k, seed, epoch, uid)
                        if train else self.view_cache.evaluation_rows(taus, self.agg_k, uid))
                toks.append(self._tokens_for(uid, rows, data, cache))
            return torch.stack(toks), torch.tensor([gt[u] for u in uids])

        trace: list[EpochScore] = []
        for epoch in range(1, max_epochs + 1):
            model.train()
            order = list(train_uids)
            _random.Random(f"{seed}:order:{epoch}").shuffle(order)
            for i in range(0, len(order), self.agg_batch):
                x, y = batch_for(order[i:i + self.agg_batch], epoch, train=True)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x.to(self.device)), y.to(self.device))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite training loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            if val_uids:
                model.eval()
                errs = []
                with torch.no_grad():
                    for i in range(0, len(val_uids), self.agg_batch):
                        chunk = val_uids[i:i + self.agg_batch]
                        x, y = batch_for(chunk, epoch, train=False)
                        pred = model(x.to(self.device)).cpu()
                        errs.append((pred - y).abs())
                        self.inner_records.extend(
                            (self._fit_no, epoch, u, gt[u], float(p))
                            for u, p in zip(chunk, pred))
                e = torch.cat(errs)
                trace.append(EpochScore(epoch=epoch, mae=float(e.mean()),
                                        rmse=float((e ** 2).mean().sqrt())))
        return model, trace

    @torch.no_grad()
    def predict(self, bunches, seed):
        """One prediction per video: the median over the M fixed evaluation view-sets."""
        if self._backbone is None or self._aggregator is None or self._scaler is None:
            raise RuntimeError("predict() before fit()")
        uids = self.index.uids_for(bunches)
        cache = self._cache if self._cache is not None else {}
        missing = [u for u in uids if u not in cache]
        if missing:
            cache.update(self._cache_features(self._backbone, missing))
        model = self._aggregator
        model.eval()
        out: list[PhasePrediction] = []
        for i in range(0, len(uids), self.agg_batch):
            chunk = uids[i:i + self.agg_batch]
            toks = [self._tokens_for(uid, torch.tensor(rows), None, cache)
                    for uid in chunk
                    for rows in evaluation_view_sets(self.taus(uid), self.agg_k,
                                                     self.m_view_sets)]
            preds = model(torch.stack(toks).to(self.device)).cpu()
            for n, uid in enumerate(chunk):
                phase = preds[n * self.m_view_sets:(n + 1) * self.m_view_sets]
                v = self.index.videos[uid]
                out.append(PhasePrediction(
                    video_uid=uid, bunch_id=v["bunch_id"], gt=_gt(v),
                    pred=float(phase.median()), outer_fold=-1, seed=seed,
                    method=self.name, phase_preds=tuple(float(x) for x in phase)))
        return out

    # ------------------------------------------------------------------ artifacts
    def write_inner_predictions(self, path: Path) -> int:
        def writer(stream):
            lines = ["fit_no\tepoch\tvideo_uid\tgt\tpred"]
            lines += [f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]:.6f}" for r in self.inner_records]
            stream.write(("\n".join(lines) + "\n").encode("utf-8"))
        atomic_new(path, writer)
        return len(self.inner_records)

    def checkpoint_blob(self, cfg: dict, provenance: dict) -> dict:
        if any(x is None for x in (self._backbone, self._aggregator, self._scaler)):
            raise RuntimeError("outer-refit model is incomplete")
        cpu = lambda m: {k: v.detach().cpu() for k, v in m.state_dict().items()}
        return {
            "format": CHECKPOINT_FORMAT,
            "model": MODEL_NAME,
            "backbone_state": cpu(self._backbone),
            "model_state": cpu(self._aggregator),
            "scaler_state": self._scaler.state_dict(),
            "input_dim": len(self.det_columns) + self._backbone.feature_dim,
            "config": cfg,
            "inference": {
                "backbone": self.backbone_name, "feature_set": self.arm,
                "det_columns": list(self.det_columns),
                "det_feature_names": [self.index.det_names[i] for i in self.det_columns],
                "views_per_set": self.agg_k, "view_sets": self.m_view_sets,
                "image_shape_hw": [self.image_size, self.image_size],
                "positional_encoding": "on",
                "view_sampler": f"evaluation_view_sets(taus, {self.agg_k}, "
                                f"{self.m_view_sets}); median over view-sets",
            },
            "provenance": provenance,
        }


def load_checkpoint(path: Path, index: VideoIndex, device: str = "cpu") -> MultiViewTrainer:
    """Restore a saved model so that `predict()` runs on any dataset built by this package."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format {blob.get('format')}")
    trainer = MultiViewTrainer(index, blob["config"], device=device, workers=0)
    spec = blob["inference"]
    if (trainer.det_columns != spec["det_columns"]
            or [index.det_names[i] for i in trainer.det_columns] != spec["det_feature_names"]):
        raise ValueError("detection features differ from the saved inference contract")
    backbone = BackboneCounter(spec["backbone"], pretrained=False).to(device)
    backbone.load_state_dict(blob["backbone_state"], strict=True)
    backbone.eval()
    model = build_model(blob["input_dim"], blob["config"]).to(device)
    model.load_state_dict(blob["model_state"], strict=True)
    model.eval()
    trainer._backbone, trainer._aggregator = backbone, model
    trainer._scaler = FeatureScaler().load_state_dict(blob["scaler_state"])
    return trainer
