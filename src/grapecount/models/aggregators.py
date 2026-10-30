"""Stage-2 aggregators, behind one registry, for comparison against the view-token model.

Every model here maps cached per-view tokens `[B, K, input_dim]` to one count `[B]`. They
differ in exactly one thing -- how the K views are collapsed into a set representation --
because that is the quantity the comparison measures. Width, depth, feed-forward size, dropout and
the regression head are shared constants (`AggregatorConfig`), and the input tokens, the
stage-1 backbone and the feature cache are shared objects, not merely shared settings.

    deep_sets_mean          Zaheer et al., Deep Sets, NeurIPS 2017
    deep_sets_max           the same, element-wise max: the classical MIL pooling control
    attention_mil           Ilse et al., Attention-based Deep MIL, ICML 2018
    set_transformer         Lee et al., Set Transformer, ICML 2019 -- SAB x2 then PMA
    view_token_transformer  this project: a CLS token inside the sequence

None of these is a reproduction of its paper's published numbers. Each is a re-implementation
evaluated on THIS dataset under THIS protocol, which is the only way a number from it may be
described.

**No positional encoding in these baselines.** They treat the K views as an unordered set, so
each aggregator must be permutation-invariant; `tests/test_aggregators.py` permutes
the token axis and asserts the output does not move, on all five. That is tested rather than
reviewed because a positional encoding added by accident can lower training loss and still
produce a plausible results table.

One deliberate deviation: **SAB is built pre-norm**, like the view-token encoder it is
compared against. Lee et al. specify post-norm. The two transformer arms should differ in
readout alone -- PMA seed against CLS-in-sequence -- and a difference in normalisation
placement would confound exactly that contrast.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import AttentionMIL
from .viewtoken import ViewTokenCounter


@dataclass(frozen=True)
class AggregatorConfig:
    """What every arm holds fixed. Read from `model.research.aggregator_common`.

    These are constants of the comparison, not hyper-parameters: nothing in the comparison
    selects them, and changing one changes every arm at once.
    """

    model_dim: int = 512
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    head_hidden: int = 256
    head_dropout: float = 0.2
    feedforward_dim: int = 1024

    @classmethod
    def from_protocol(cls, cfg: dict) -> "AggregatorConfig":
        common = cfg["model"]["research"]["aggregator_common"]
        return cls(
            model_dim=int(common["model_dim"]), layers=int(common["layers"]),
            heads=int(common["heads"]), dropout=float(common["dropout"]),
            head_hidden=int(common["head_hidden"]),
            head_dropout=float(common["head_dropout"]),
            feedforward_dim=int(common["feedforward_dim"]),
        )


def _regression_head(cfg: AggregatorConfig) -> nn.Module:
    """LayerNorm then MLP to one value. Byte-identical in shape to `ViewTokenCounter`'s.

    Shared so that a difference between two rows of the results table is a difference in
    pooling and not in how the pooled vector is read out.
    """
    return nn.Sequential(
        nn.LayerNorm(cfg.model_dim),
        nn.Linear(cfg.model_dim, cfg.head_hidden), nn.ReLU(inplace=True),
        nn.Dropout(cfg.head_dropout), nn.Linear(cfg.head_hidden, 1),
    )


# ------------------------------------------------------------------------ Deep Sets
class DeepSetsCounter(nn.Module):
    """`rho(pool_k(phi(x_k)))` -- Zaheer et al. 2017.

    `phi` is applied to each view independently and `pool` is a symmetric function, which is
    what makes the model permutation-invariant by construction rather than by training. The
    mean and the max share every parameter shape; only the pool differs, so the two rows of
    the results table isolate the pooling operation alone.
    """

    def __init__(self, input_dim: int, cfg: AggregatorConfig, pool: str = "mean") -> None:
        super().__init__()
        if pool not in ("mean", "max"):
            raise ValueError(f"unknown Deep Sets pool {pool!r}; known: mean, max")
        self.pool = pool
        d = cfg.model_dim
        layers: list[nn.Module] = []
        prev = input_dim
        for _ in range(cfg.layers):
            layers += [nn.Linear(prev, d), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout)]
            prev = d
        self.phi = nn.Sequential(*layers)
        self.rho = _regression_head(cfg)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.phi(tokens)                                    # [B, K, d]
        z = h.mean(dim=1) if self.pool == "mean" else h.amax(dim=1)
        return self.rho(z).squeeze(-1)


# ---------------------------------------------------------------------- Attention-MIL
class AttentionMILCounter(nn.Module):
    """Gated attention pooling over view tokens -- Ilse et al. 2018.

    The pooling module is the one that already trains stage 1 (`models/backbone.py`), reused
    rather than re-typed: two implementations of the same equation tend to drift apart.

    Attention weights each view independently. It can say "trust this view" but not "these
    two views show the same berries", which is the distinction the view-token transformer was
    introduced to test -- so this arm is the control that makes that claim measurable.
    """

    def __init__(self, input_dim: int, cfg: AggregatorConfig) -> None:
        super().__init__()
        self.project = nn.Linear(input_dim, cfg.model_dim)
        self.pool = AttentionMIL(cfg.model_dim, hidden=cfg.head_hidden, dropout=cfg.dropout)
        self.head = _regression_head(cfg)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.project(tokens))).squeeze(-1)


# --------------------------------------------------------------------- Set Transformer
class PMA(nn.Module):
    """Pooling by Multihead Attention with a single seed -- Lee et al. 2019.

    One learned seed vector queries the K encoded tokens, so the set representation is
    produced by cross-attention from outside the sequence. That is the whole difference from
    the view-token transformer, where the readout token sits INSIDE the sequence and is
    attended to by the views in every layer.

    Pre-norm, matching the encoder above it; see this module's docstring.
    """

    def __init__(self, cfg: AggregatorConfig) -> None:
        super().__init__()
        d = cfg.model_dim
        self.seed = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.seed, std=0.02)
        # rFF(Z) in the paper: the row-wise feed-forward applied to the set before pooling.
        self.rff = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True))
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg.feedforward_dim), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.feedforward_dim, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # [B, K, d] -> [B, d]
        z = self.rff(x)
        q = self.seed.expand(x.shape[0], -1, -1)
        kv = self.norm_kv(z)
        h = q + self.attn(self.norm_q(q), kv, kv, need_weights=False)[0]
        h = h + self.ff(self.norm_ff(h))
        return h.squeeze(1)


class SetTransformerCounter(nn.Module):
    """SAB x `layers` then PMA -- Lee et al. 2019.

    SAB is self-attention over the set with no positional information, i.e. a standard
    transformer encoder block over the K tokens; it is permutation-EQUIVARIANT. PMA collapses
    the equivariant output to one vector and is permutation-INVARIANT, which is what makes the
    whole model invariant.
    """

    def __init__(self, input_dim: int, cfg: AggregatorConfig) -> None:
        super().__init__()
        self.project = nn.Linear(input_dim, cfg.model_dim)
        block = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim, nhead=cfg.heads, dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.sab = nn.TransformerEncoder(block, num_layers=cfg.layers)
        self.pma = PMA(cfg)
        self.head = _regression_head(cfg)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.pma(self.sab(self.project(tokens)))).squeeze(-1)


# ------------------------------------------------------------- the proposed aggregator
class ViewTokenAggregator(nn.Module):
    """`ViewTokenCounter` under the registry's constructor signature.

    A thin wrapper rather than an edit to `viewtoken.py`, so the proposed arm is the same
    class used elsewhere and not a re-typing of it. Positional encoding is left off here, so
    this arm is the view-token model without view-order information.
    """

    def __init__(self, input_dim: int, cfg: AggregatorConfig) -> None:
        super().__init__()
        self.inner = ViewTokenCounter(
            input_dim, model_dim=cfg.model_dim, layers=cfg.layers, heads=cfg.heads,
            dropout=cfg.dropout, head_dropout=cfg.head_dropout,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.inner(tokens)


# ------------------------------------------------------------------------- the registry
AGGREGATORS = {
    "deep_sets_mean": lambda dim, cfg: DeepSetsCounter(dim, cfg, pool="mean"),
    "deep_sets_max": lambda dim, cfg: DeepSetsCounter(dim, cfg, pool="max"),
    "attention_mil": AttentionMILCounter,
    "set_transformer": SetTransformerCounter,
    "view_token_transformer": ViewTokenAggregator,
}

# Where each arm comes from, carried beside the code so a results table can cite it without a
# human retyping the citation. "this project" is stated as plainly as the papers are.
AGGREGATOR_SOURCE = {
    "deep_sets_mean": "Zaheer et al., Deep Sets, NeurIPS 2017 (re-implementation)",
    "deep_sets_max": "Zaheer et al., Deep Sets, NeurIPS 2017 (re-implementation, max pool)",
    "attention_mil": "Ilse et al., Attention-based Deep MIL, ICML 2018 (re-implementation)",
    "set_transformer": "Lee et al., Set Transformer, ICML 2019 (re-implementation, SAB+PMA)",
    "view_token_transformer": "this project -- the proposed method (view token, no positional encoding)",
}


def make_aggregator(name: str, input_dim: int, cfg: AggregatorConfig) -> nn.Module:
    """The only way an aggregator is constructed.

    A factory rather than a branch at the call site, because the runner, the smoke test and
    the permutation test must all build the same object from the same string -- an arm that
    exists in one of those places and not the others is an arm nobody has checked.
    """
    if name not in AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; registered: {sorted(AGGREGATORS)}")
    return AGGREGATORS[name](input_dim, cfg)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
