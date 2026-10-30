"""Helpers for reading the run configuration."""

from __future__ import annotations


def backbones(cfg: dict) -> list[str]:
    """The backbone names listed in the config.

    Accepts either `model.research.backbones` (a list, validated for emptiness and
    duplicates) or the single-backbone form `model.research.backbone`, which yields a
    one-element list.
    """
    research = cfg["model"]["research"]
    if "backbones" in research:
        names = list(research["backbones"])
        if not names:
            raise ValueError("model.research.backbones is empty")
        if len(set(names)) != len(names):
            raise ValueError(f"model.research.backbones has duplicates: {names}")
        return names
    return [research["backbone"]]
