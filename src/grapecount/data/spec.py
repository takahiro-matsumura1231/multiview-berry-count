"""Load and interpret configs/dataset.yaml.

Every rule that decides which frames exist lives here, so that "the dataset is a pure
function of (raw root, spec, ffmpeg version)" is a property of the code and not just a
claim in a document.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = REPO / "configs" / "dataset.yaml"


@dataclass(frozen=True)
class DatasetSpec:
    version: str
    raw_root: Path
    filename_re: re.Pattern[str]
    extensions: tuple[str, ...]
    exclude_dir_substrings: tuple[str, ...]
    exclude_name_substrings: tuple[str, ...]
    exclude_path_substrings: tuple[str, ...]
    exclude_hidden: bool
    dedup_require_byte_identical: bool
    merge_labels_across_sessions: bool
    head_sec: float
    tail_sec: float
    fps: float
    min_frames: int
    image_width: int
    jpeg_quality: int
    assert_portrait: bool
    frame_path_template: str
    forbidden_path_tokens: tuple[str, ...]
    output_root: Path
    unlabeled: bool
    raw: dict
    spec_sha256: str            # of the file text: exact provenance
    spec_semantic_sha256: str   # of the parsed values: what actually invalidates a dataset

    # ---------------------------------------------------------------- frame grid
    #
    # The whole grid is integer arithmetic on milliseconds. That is not tidiness: some
    # videos have durations a few
    # tenths of a millisecond below an exact frame boundary -- 23.499600 s against a
    # boundary at 23.500 s -- so a float formula gains or loses a frame depending on how
    # the duration was rounded on its way in. Rounding the duration to whole milliseconds
    # first, at the same resolution the timestamps themselves use, removes the knife edge.
    #
    # `duration_ms` is what the manifest stores, so the verifier can re-derive the grid
    # exactly instead of approximately.

    @property
    def step_ms(self) -> int:
        step = round(1000.0 / self.fps)
        if abs(step - 1000.0 / self.fps) > 1e-9:
            raise ValueError(
                f"fps={self.fps} does not give whole-millisecond frame spacing; "
                "the spec requires exact_ms timestamps"
            )
        return step

    @property
    def head_ms(self) -> int:
        head = round(self.head_sec * 1000.0)
        if abs(head - self.head_sec * 1000.0) > 1e-9:
            raise ValueError(f"head_sec={self.head_sec} is not a whole number of ms")
        return head

    @property
    def tail_ms(self) -> int:
        tail = round(self.tail_sec * 1000.0)
        if abs(tail - self.tail_sec * 1000.0) > 1e-9:
            raise ValueError(f"tail_sec={self.tail_sec} is not a whole number of ms")
        return tail

    @staticmethod
    def duration_ms(duration_s: float) -> int:
        return round(duration_s * 1000.0)

    def n_frames(self, duration_ms: int) -> int:
        """Number of frames the grid yields. Pure integer function of the spec."""
        return max(0, (duration_ms - self.head_ms - self.tail_ms) // self.step_ms)

    def frame_times_ms(self, duration_ms: int) -> list[int]:
        """t_k = head_ms + k * step_ms, exact integers.

        The last frame is always at least one step before the end of the video, so no
        seek can land past the final decodable frame.
        """
        return [self.head_ms + k * self.step_ms for k in range(self.n_frames(duration_ms))]

    def admits(self, duration_ms: int) -> bool:
        return self.n_frames(duration_ms) >= self.min_frames

    # ------------------------------------------------------------------- paths
    def frame_relpath(self, bunch_id: str, video_uid: int, frame_idx: int, ts_ms: int) -> str:
        return self.frame_path_template.format(
            bunch_id=bunch_id, video_uid=video_uid, frame_idx=frame_idx, timestamp_ms=ts_ms
        )

    def path_violations(self, relpath: str) -> list[str]:
        """Tokens that must never appear in a derived path (the GT is in the raw filename)."""
        low = relpath.lower()
        return [t for t in self.forbidden_path_tokens if t in low]


UNLABELED_PATTERN = r"^(?P<bunch>.+)\.(?P<ext>[A-Za-z0-9]+)$"


def load_spec(path: Path = DEFAULT_SPEC, raw_root: Path | None = None,
              output_root: Path | None = None, unlabeled: bool = False) -> DatasetSpec:
    """Read a dataset spec, optionally pointing it at another raw tree and output root.

    Overrides are applied before hashing, so `verify` must be given the same
    --raw-root / --out / --unlabeled as `build`. `unlabeled` admits any video file name and leaves the
    count empty (NaN); such a dataset is for prediction, not for training.
    """
    text = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    if raw_root is not None:
        cfg["source"]["raw_root"] = str(Path(raw_root).resolve())
    if output_root is not None:
        cfg["output"]["root"] = str(Path(output_root).resolve())
    if unlabeled:
        cfg["source"]["filename_pattern"] = UNLABELED_PATTERN
        cfg["source"]["unlabeled"] = True
    if raw_root is not None or output_root is not None or unlabeled:
        text = yaml.safe_dump(cfg, sort_keys=True)   # the text hash reflects the overrides
    src, trim, fr, img, nam, out = (
        cfg["source"], cfg["trim"], cfg["frames"], cfg["image"], cfg["naming"], cfg["output"]
    )
    flags = 0
    for name in src.get("filename_pattern_flags", []):
        flags |= getattr(re, name)
    if fr.get("grid") != "analytic":
        raise ValueError("only the analytic frame grid is implemented")
    if out.get("line_endings") != "LF":
        raise ValueError("only LF output is supported")
    return DatasetSpec(
        version=cfg["version"],
        raw_root=REPO / src["raw_root"],
        filename_re=re.compile(src["filename_pattern"], flags),
        extensions=tuple(src["extensions"]),
        exclude_dir_substrings=tuple(src["exclude"]["dir_substrings"]),
        exclude_name_substrings=tuple(src["exclude"]["name_substrings"]),
        exclude_path_substrings=tuple(src["exclude"]["path_substrings"]),
        exclude_hidden=bool(src["exclude"]["hidden_files"]),
        dedup_require_byte_identical=bool(src["deduplicate"]["require_byte_identical"]),
        merge_labels_across_sessions=bool(cfg["grouping"]["merge_labels_across_sessions"]),
        head_sec=float(trim["head_sec"]),
        tail_sec=float(trim["tail_sec"]),
        fps=float(fr["fps"]),
        min_frames=int(cfg["admission"]["min_frames"]),
        image_width=int(img["width"]),
        jpeg_quality=int(img["jpeg_quality"]),
        assert_portrait=bool(img["rotation"]["assert_portrait"]),
        frame_path_template=nam["frame_path"],
        forbidden_path_tokens=tuple(nam["forbid"]),
        output_root=REPO / out["root"],
        unlabeled=bool(src.get("unlabeled", False)),
        raw=cfg,
        spec_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        spec_semantic_sha256=semantic_hash(cfg),
    )


def semantic_hash(cfg: dict) -> str:
    """Hash the parsed values, not the file text.

    A dataset is invalidated by a changed *value*, not by a reworded comment. Hashing the
    raw text would force regenerating every frame to fix a typo, which makes the
    check something people route around. Sorted-key JSON is a stable canonical form.
    """
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def qscale_for(jpeg_quality: int) -> int:
    """ffmpeg -q:v from a 0-100 quality, (92 -> 3)."""
    return int(31 - jpeg_quality * 0.3)
