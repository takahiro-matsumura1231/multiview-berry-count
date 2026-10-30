"""The image backbone and the stage-1 model that fine-tunes it.

Two stages, and conflating them is the mistake this docstring exists to prevent:

    stage 1   ResNet18 @448 is fine-tuned END TO END on grape frames, with an
              Attention-MIL head over K views. This is where the backbone learns.
    stage 2   that backbone is FROZEN, per-view features are cached once, and only the
              view-token aggregator trains on top.

"Frozen" in stage 2 therefore means frozen after fine-tuning on this data, not an
off-the-shelf ImageNet backbone. A backbone that is never fine-tuned is a much weaker
feature extractor for counting; fine-tuning and input resolution were the dominant levers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

FEATURE_DIM = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}


def make_backbone(name: str = "resnet18", pretrained: bool = True) -> tuple[nn.Module, int]:
    """A CNN trunk returning a pooled feature vector per image.

    ImageNet initialisation is the starting point, not the end point: stage 1 fine-tunes
    from here.
    """
    if name not in FEATURE_DIM:
        raise ValueError(f"unknown backbone {name!r}; known: {sorted(FEATURE_DIM)}")
    weights = "IMAGENET1K_V1" if pretrained else None
    net = getattr(torchvision.models, name)(weights=weights)
    net.fc = nn.Identity()          # keep the global-average-pooled feature
    return net, FEATURE_DIM[name]


class AttentionMIL(nn.Module):
    """Gated attention pooling over K views.

    A bunch is photographed from K angles and the count is a property of the object, not of
    any one view, so the aggregator must be permutation-invariant. Attention-MIL is used here only
    to give stage 1 a training signal; the stage-2 aggregator is the view-token transformer.
    """

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.attend = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.score = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [B, K, D]
        w = self.score(self.attend(x) * self.gate(x))        # [B, K, 1]
        w = torch.softmax(w, dim=1)
        return self.drop((w * x).sum(dim=1))                 # [B, D]


class BackboneCounter(nn.Module):
    """Stage-1 model: per-view CNN features -> attention pooling -> count.

    Its predictions are never reported. It exists to teach the backbone what a berry looks
    like at 448; what survives into stage 2 is `self.backbone` alone.
    """

    def __init__(self, backbone_name: str = "resnet18", dropout: float = 0.2,
                 pretrained: bool = True) -> None:
        super().__init__()
        self.backbone, dim = make_backbone(backbone_name, pretrained=pretrained)
        self.pool = AttentionMIL(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, 1)
        )
        self.feature_dim = dim

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """views: [B, K, 3, H, W] -> [B] predicted counts."""
        b, k = views.shape[:2]
        feats = self.backbone(views.flatten(0, 1)).view(b, k, -1)
        return self.head(self.pool(feats)).squeeze(-1)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Per-image features, for building the stage-2 cache. [N,3,H,W] -> [N,D]."""
        return self.backbone(images)


def parameter_groups(model: BackboneCounter, lr_backbone: float, lr_head: float) -> list[dict]:
    """Discriminative learning rates.

    The trunk carries ImageNet structure worth preserving and is moved slowly; the freshly
    initialised pooling and head are moved an order of magnitude faster. Typical values
    are 1e-4 for the trunk and 1e-3 for the head.
    """
    trunk, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (trunk if name.startswith("backbone.") else rest).append(p)
    return [{"params": trunk, "lr": lr_backbone}, {"params": rest, "lr": lr_head}]
