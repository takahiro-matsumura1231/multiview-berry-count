"""Image geometry must reach preprocessing and distinguish backbone cache entries."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image  # noqa: E402
import yaml  # noqa: E402

from grapecount.data.views import build_transform  # noqa: E402
from grapecount.train.pipeline import TwoStageTrainer  # noqa: E402
from grapecount.train.store import BackboneStore  # noqa: E402


def test_legacy_square_and_explicit_rectangular_resize():
    image = Image.new("RGB", (640, 480))
    assert build_transform(448, train=False)(image).shape == (3, 448, 448)
    assert build_transform((598, 336), train=False)(image).shape == (3, 598, 336)


def test_geometry_is_part_of_backbone_store_key():
    bunches = {"A"}
    store = BackboneStore()
    store.put("resnet18", 0, bunches, "square", {}, image_shape_hw=(448, 448))
    assert store.get("resnet18", 0, bunches, image_shape_hw=(598, 336)) is None
    assert store.get("resnet18", 0, bunches, image_shape_hw=(448, 448))[0] == "square"
    store.put("resnet18", 0, bunches, "rectangle", {}, image_shape_hw=(598, 336))
    assert store.get("resnet18", 0, bunches, image_shape_hw=(598, 336))[0] == "rectangle"


