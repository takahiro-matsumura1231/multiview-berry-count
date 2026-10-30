"""The baseline trainer: one stage-1 backbone, any stage-2 aggregator, any K.

Behind the same `Trainer` contract as `TwoStageTrainer` and the reference models, so the
runner, the seal and the epoch selector are the shared, tested ones -- not re-implemented.
What it adds:

* **the aggregator is a constructor argument**, resolved through `models/aggregators.py`, so
  five architectures reach the identical training loop, the identical optimiser, the identical
  budget and the identical inner-fold selection rule;
* **K reaches the aggregator only.** The backbone is fine-tuned once at K=9 per (seed,
  training bunch set) and shared. A K=1 arm is the same backbone reading one token instead of
  nine, which is what makes it a view-count ablation rather than a second training
  condition;
* **every per-phase prediction is stored**, not only their median. Storing only the median
  over the M=5 fixed phase-sets is the right primary number and the wrong thing to keep: a K=1 arm scored that way consumes five frames. The M per-phase values are written to
  `predictions.tsv` so the single-view, equal-budget and median rules are all derivable from
  the stored file rather than from the code that wrote it.

Two properties are load-bearing and are asserted rather than assumed:

* **The backbone store is keyed on (backbone, seed, training bunch set) and nothing else.**
  If the aggregator or K entered that key, each arm would fine-tune its own backbone and the
  aggregator contrast would carry stage-1 initialisation noise, which can be of the same
  order as the effect being measured. Every unit therefore
  records a sha256 of the backbone it actually trained on and of the feature cache it read, so
  sharing is evidence in the run manifest rather than a claim in a docstring.

* **The feature standardiser is fitted on the current fit's training bunches only.** Fitting
  it on everything leaks the evaluation fold's distribution into training.

The stage-2 inner loop is arithmetically identical to `TwoStageTrainer`'s and reordered for
cost: the standardised token matrix of each video is materialised once per fit instead of once
per video per epoch. Indexing after an element-wise affine transform is the same tensor as the
transform after indexing, so the result is the same as with the per-video transform.
"""

from __future__ import annotations

import hashlib
import random as _random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..data.views import VideoIndex
from ..models import AggregatorConfig, FeatureScaler, make_aggregator, n_parameters
from ..views.sampling import evaluation_view_sets, sample_training_views
from .pipeline import TwoStageTrainer, warmup_cosine
from .runner import EpochScore


class ViewSetCache:
    """Memoised view-set selection, shared by the arms of one (seed, outer fold) block.

    `sample_training_views(taus, k, Random(f"{seed}:{epoch}:{uid}"))` is a pure function of
    `(k, seed, epoch, uid)`: the RNG is constructed fresh from that string on every call and
    `taus` is fixed per video. Several arms train on the same videos for the same epochs with
    the same K, so without memoisation the same 9-of-51 selection is recomputed once per arm,
    and it is the single most expensive thing in the stage-2 loop -- pure-Python, O(k x frames),
    for every video, every epoch and every fit.

    This changes no result. The cached value is the value the function returns, stored as the
    index tensor the token matrix is gathered with; `tests/test_view_cache.py` asserts the
    memoised and unmemoised selections are identical over every video of a fold and a range of
    epochs, because "it is obviously a pure function" is how a cache that is not one gets in.

    Cleared between blocks by the runner, next to the backbone store it parallels: a stale
    entry cannot be wrong here, but an unbounded cache across many blocks would not be free.
    """

    def __init__(self) -> None:
        self._train: dict[tuple, torch.Tensor] = {}
        self._eval: dict[tuple, torch.Tensor] = {}
        self.hits = 0
        self.misses = 0

    def training_rows(self, taus: list[float], k: int, seed: int, epoch: int,
                      uid: str) -> torch.Tensor:
        key = (k, seed, epoch, uid)
        got = self._train.get(key)
        if got is None:
            self.misses += 1
            rows = sample_training_views(taus, k, _random.Random(f"{seed}:{epoch}:{uid}"))
            got = torch.tensor(rows, dtype=torch.long)
            self._train[key] = got
        else:
            self.hits += 1
        return got

    def evaluation_rows(self, taus: list[float], k: int, uid: str) -> torch.Tensor:
        key = (k, uid)
        got = self._eval.get(key)
        if got is None:
            self.misses += 1
            got = torch.tensor(evaluation_view_sets(taus, k, m=1)[0], dtype=torch.long)
            self._eval[key] = got
        else:
            self.hits += 1
        return got

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "held": len(self._train) + len(self._eval)}

    def clear(self) -> None:
        self._train.clear()
        self._eval.clear()


@dataclass(frozen=True)
class PhasePrediction:
    """One stage-video, with the M per-phase predictions kept beside their median.

    Field-compatible with `runner.Prediction`, because `run_outer_fold` stamps fold identity
    with `dataclasses.replace` and must keep working on this type unchanged.
    """

    video_uid: str
    bunch_id: str
    gt: float
    pred: float
    outer_fold: int
    seed: int
    method: str
    phase_preds: tuple[float, ...]


def digest_state_dict(model: nn.Module) -> str:
    """A sha256 over every parameter of a model, in sorted key order.

    This is what turns "the aggregators shared a backbone" from a docstring into evidence: two
    units of one (seed, outer fold) block that report different digests did not share, and the
    block is invalid however plausible its numbers look.
    """
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def digest_cache(cache: dict[str, torch.Tensor], uids: list[str], n_videos: int = 32) -> str:
    """A sha256 over the cached features of the FIRST `n_videos` of `uids`, in uid order.

    The video set is passed in rather than taken from the cache's keys, and a smoke test
    showed why. A cache is extended in place as a block proceeds -- the first arm adds
    the outer-test videos when it scores them -- so a digest over "whatever the cache holds"
    changes between the first arm and the second even when every shared feature is identical,
    and the sharing evidence would report a violation that did not happen. Digesting the
    videos the fit actually trained on is both stable across arms and the thing the claim is
    about: these arms consumed byte-identical per-frame features.

    A prefix rather than the whole cache because the whole cache is gigabytes and a prefix
    already separates two backbones: the features are a deterministic function of the weights,
    so identical prefixes over 32 videos do not happen by accident.
    """
    h = hashlib.sha256()
    for uid in sorted(uids, key=int)[:n_videos]:
        h.update(uid.encode())
        h.update(cache[uid].contiguous().numpy().tobytes())
    return h.hexdigest()


class BaselineTrainer(TwoStageTrainer):
    """Stage 1 from `TwoStageTrainer`, stage 2 from the aggregator registry.

    `_finetune_backbone` and `_cache_features` are inherited unchanged, so stage 1 is identical
    for every aggregator.
    """

    def __init__(self, index: VideoIndex, cfg: dict, aggregator: str, agg_k: int,
                 device: str = "cuda", workers: int = 8, log=print, store=None,
                 name: str | None = None, view_cache: "ViewSetCache | None" = None) -> None:
        research = cfg["model"]["research"]
        self.index = index
        self.store = store
        self.view_cache = view_cache if view_cache is not None else ViewSetCache()
        self.device = device
        self.workers = workers
        self.log = log

        self.backbone_name = str(research["backbone"])
        self.arm = str(research["feature_arm"])
        self.aggregator_name = aggregator
        self.agg_cfg = AggregatorConfig.from_protocol(cfg)

        # `k` is stage 1's view count and is a constant of the configuration; `agg_k` is the arm.
        self.k = int(research["backbone_views_per_set"])
        self.agg_k = int(agg_k)
        self.image_size = int(research["resolution"])
        self.backbone_epochs = int(research["backbone_epochs"])
        self.backbone_batch = int(research["backbone_batch"])
        self.lr_backbone = float(research["lr_backbone"])
        self.lr_head = float(research["lr_head"])
        self.agg_max_epochs = int(research["aggregator_max_epochs"])
        self.agg_batch = int(research["aggregator_batch"])
        self.agg_lr = float(research["aggregator_lr"])
        self.agg_wd = float(research["aggregator_weight_decay"])
        self.m_view_sets = int(cfg["evaluation"]["view_sets"])
        self.det_columns = index.det_columns(self.arm)

        self.name = name or f"{self.backbone_name}/{self.arm}/{aggregator}/k{self.agg_k}"

        self._backbone = None
        self._aggregator = None
        self._scaler: FeatureScaler | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        self._taus: dict[str, list[float]] = {}

        # Provenance accumulated across the five fits of one unit, written to the manifest.
        self.fit_log: list[dict] = []
        self.n_aggregator_parameters: int | None = None

    # ------------------------------------------------------------------ helpers
    def taus(self, uid: str) -> list[float]:
        """Cached per-video tau. Memoised: it is otherwise recomputed every epoch."""
        t = self._taus.get(uid)
        if t is None:
            t = self.index.taus(uid)
            self._taus[uid] = t
        return t

    # ------------------------------------------------------------------ stage 2
    def _train_aggregator(self, cache: dict, train_uids: list[str], val_uids: list[str],
                          seed: int, max_epochs: int) -> tuple[nn.Module, list[EpochScore]]:
        torch.manual_seed(seed + 1)

        # Standardisation fitted on TRAINING bunches only. Never on val, never on test.
        train_rows = torch.cat([self._tokens(cache, u, list(range(len(cache[u]))))
                                for u in train_uids])
        self._scaler = FeatureScaler().fit(train_rows)
        dim = train_rows.shape[1]
        del train_rows

        # Standardise every video once instead of once per epoch. The transform is
        # element-wise, so `scaled[uid][rows] == scaler.transform(tokens(uid, rows))` exactly.
        scaled = {u: self._scaler.transform(self._tokens(cache, u, list(range(len(cache[u])))))
                  for u in train_uids + val_uids}

        model = make_aggregator(self.aggregator_name, dim, self.agg_cfg).to(self.device)
        self.n_aggregator_parameters = n_parameters(model)
        opt = torch.optim.AdamW(model.parameters(), lr=self.agg_lr, weight_decay=self.agg_wd)
        sched = warmup_cosine(opt, max_epochs)
        loss_fn = nn.L1Loss()

        def batch_for(uids: list[str], epoch: int, train: bool):
            tokens, gts = [], []
            for uid in uids:
                taus = self.taus(uid)
                rows = (self.view_cache.training_rows(taus, self.agg_k, seed, epoch, uid)
                        if train else
                        self.view_cache.evaluation_rows(taus, self.agg_k, uid))
                tokens.append(scaled[uid][rows])
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
                loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    # Never silently skipped: a non-finite loss makes this a failed unit to be
                    # re-run, not a batch to step over.
                    raise RuntimeError(
                        f"non-finite training loss in {self.name} at epoch {epoch}; "
                        f"the unit must be re-run")
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
        stage = "refit" if val_bunches is None else "inner"
        self.log(f"    [fit] {len(train_uids)} train videos, {len(val_uids)} val, "
                 f"cell={self.name} seed={seed}"
                 + (f", refit to epoch {max_epochs}" if val_bunches is None else ""))

        cached = (self.store.get(self.backbone_name, seed, train_bunches)
                  if self.store is not None else None)
        if cached is not None:
            backbone, cache = cached
            missing = [u for u in train_uids + val_uids if u not in cache]
            if missing:
                # Mutated in place so the extension is visible to every later arm of this
                # block; a copy here would re-encode the same frames for every arm.
                cache.update(self._cache_features(backbone, missing))
            self.log(f"    [fit] reusing the {self.backbone_name} backbone from an earlier "
                     f"aggregator arm of this block")
        else:
            backbone = self._finetune_backbone(train_bunches, seed)
            cache = self._cache_features(backbone, train_uids + val_uids)
            if self.store is not None:
                self.store.put(self.backbone_name, seed, train_bunches, backbone, cache)

        epochs = max_epochs or self.agg_max_epochs
        aggregator, trace = self._train_aggregator(cache, train_uids, val_uids, seed, epochs)

        self._backbone, self._aggregator, self._cache = backbone, aggregator, cache
        self.fit_log.append({
            "stage": stage,
            "n_train_bunches": len(train_bunches),
            "n_train_videos": len(train_uids),
            "n_val_videos": len(val_uids),
            "backbone_from_store": cached is not None,
            "backbone_sha256": digest_state_dict(backbone),
            "feature_cache_sha256": digest_cache(cache, train_uids),
            "n_cached_videos": len(cache),
            "aggregator_parameters": self.n_aggregator_parameters,
            "epochs": epochs,
            "seconds": round(time.time() - t0, 2),
        })
        self.log(f"    [fit] done in {(time.time() - t0) / 60:.1f} min"
                 + (f", best inner-val MAE {min(t.mae for t in trace):.3f}" if trace else ""))
        return trace

    @torch.no_grad()
    def predict(self, bunches: set[str], seed: int) -> list[PhasePrediction]:
        """M deterministic view-sets per video. All M are kept; `pred` is their median.

        Reached only after the runner has unsealed the outer-test set, so the features cached
        here are produced by a backbone that trained on outer-train alone.
        """
        if self._backbone is None or self._aggregator is None or self._scaler is None:
            raise RuntimeError("predict() before fit()")
        uids = self.index.uids_for(bunches)
        cache = self._cache if self._cache is not None else {}
        missing = [u for u in uids if u not in cache]
        if missing:
            cache.update(self._cache_features(self._backbone, missing))
        self._aggregator.eval()

        out: list[PhasePrediction] = []
        for i in range(0, len(uids), self.agg_batch):
            chunk = uids[i:i + self.agg_batch]
            batch, per_video_sets = [], []
            for uid in chunk:
                sets = evaluation_view_sets(self.taus(uid), self.agg_k, self.m_view_sets)
                per_video_sets.append(len(sets))
                for rows in sets:
                    batch.append(self._scaler.transform(self._tokens(cache, uid, rows)))
            preds = self._aggregator(torch.stack(batch).to(self.device)).cpu()
            at = 0
            for uid, m in zip(chunk, per_video_sets):
                phase = preds[at:at + m]
                at += m
                v = self.index.videos[uid]
                out.append(PhasePrediction(
                    video_uid=uid, bunch_id=v["bunch_id"], gt=float(v["gt_count"]),
                    pred=float(phase.median()), outer_fold=-1, seed=seed, method=self.name,
                    phase_preds=tuple(float(x) for x in phase)))
        return out
