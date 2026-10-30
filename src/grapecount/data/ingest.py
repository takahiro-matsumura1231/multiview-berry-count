"""Scan the raw capture tree and decide which videos the dataset admits.

Two things here are deliberately strict:

* Deduplication verifies byte-identity. Grouping by basename and keeping one file would
  silently drop distinct content, so a non-identical collision raises.
* The bunch label is the grouping (leak) unit and is taken from the filename, so the same
  label in two sessions is merged. That is a conservative choice, not a claim that they
  are the same physical bunch.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .spec import DatasetSpec


@dataclass
class VideoRecord:
    relpath: str          # relative to raw_root; the manifest stores this, so it stays portable
    session_id: str
    bunch_id: str
    stage_id: int
    gt_count: int | float   # NaN for an unlabeled dataset
    video_uid: int = -1
    duration_ms: int = -1
    n_frames: int = 0


@dataclass
class IngestResult:
    admitted: list[VideoRecord]
    excluded: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    counts: dict[str, int] = field(default_factory=dict)


def session_of(relpath: str) -> str:
    """Session id: the top-level capture directory name, normalised by stripping fixed
    naming affixes so equivalent sessions compare equal."""
    if os.sep not in relpath:
        return ""                     # videos placed directly in raw_root: no session
    top = relpath.split(os.sep, 1)[0]
    return top.removeprefix("berry_").split("_")[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def scan(spec: DatasetSpec) -> IngestResult:
    res = IngestResult(admitted=[])
    counts: dict[str, int] = defaultdict(int)
    candidates: list[VideoRecord] = []

    for dirpath, _dirs, files in os.walk(spec.raw_root):
        rel_dir = os.path.relpath(dirpath, spec.raw_root)
        low_dir = dirpath.lower()
        in_frame_dir = any(s in low_dir for s in spec.exclude_dir_substrings)
        in_excluded_path = any(s in low_dir for s in spec.exclude_path_substrings)
        for name in files:
            rel = os.path.normpath(os.path.join(rel_dir, name))
            if spec.exclude_hidden and (name.startswith(".") or name in {"Thumbs.db"}):
                counts["hidden_or_system"] += 1
                continue
            if os.path.splitext(name)[1].lower() not in spec.extensions:
                counts["not_a_video"] += 1
                continue
            counts["video_files"] += 1
            if in_frame_dir:
                counts["in_frame_directory"] += 1
                res.excluded["in_frame_directory"].append(rel)
                continue
            if in_excluded_path:
                counts["in_excluded_path"] += 1
                res.excluded["in_excluded_path"].append(rel)
                continue
            low_name = name.lower()
            if any(s in low_name for s in spec.exclude_name_substrings):
                counts["marked_duplicate_or_trash"] += 1
                res.excluded["marked_duplicate_or_trash"].append(rel)
                continue
            m = spec.filename_re.match(name)
            if not m:
                counts["no_gt_suffix"] += 1
                res.excluded["no_gt_suffix"].append(rel)
                continue
            counts["name_conforming"] += 1
            if spec.unlabeled:
                # Each video is its own unit; the ID is assigned after sorting below,
                # because a free-form file name may contain path tokens the spec forbids.
                record = VideoRecord(relpath=rel, session_id=session_of(rel), bunch_id="",
                                     stage_id=0, gt_count=float("nan"))
            else:
                record = VideoRecord(relpath=rel, session_id=session_of(rel),
                                     bunch_id=m.group("bunch").upper(),
                                     stage_id=int(m.group("stage")),
                                     gt_count=int(m.group("count")))
            candidates.append(record)

    kept = _deduplicate(spec, candidates, res, counts)
    # video_uid is the index into the sorted relative paths, so it is readable and
    # reproducible; the manifest also stores a path hash so renumbering is detectable.
    kept.sort(key=lambda r: r.relpath)
    for i, rec in enumerate(kept):
        rec.video_uid = i
        if spec.unlabeled:
            rec.bunch_id = f"V{i:05d}"
    res.admitted = kept
    counts["deduplicated"] = len(kept)
    res.counts = dict(counts)
    return res


def _deduplicate(
    spec: DatasetSpec,
    candidates: list[VideoRecord],
    res: IngestResult,
    counts: dict[str, int],
) -> list[VideoRecord]:
    """Collapse basename collisions, requiring byte-identity.

    The key is the basename *including* its extension. Keying on the stem would be wrong:
    the same stem can exist as both .MOV and .mp4 with genuinely different content
    (different durations), and collapsing those would delete part of a bunch.
    """
    if spec.unlabeled:
        return list(candidates)       # free-form names: equal basenames are different videos
    by_name: dict[str, list[VideoRecord]] = defaultdict(list)
    for rec in candidates:
        by_name[os.path.basename(rec.relpath)].append(rec)

    kept: list[VideoRecord] = []
    for name, group in sorted(by_name.items()):
        if len(group) == 1:
            kept.append(group[0])
            continue
        group.sort(key=lambda r: (len(r.relpath), r.relpath))  # prefer the shortest path
        winner = group[0]
        if spec.dedup_require_byte_identical:
            ref = sha256(spec.raw_root / winner.relpath)
            for other in group[1:]:
                if sha256(spec.raw_root / other.relpath) != ref:
                    raise AssertionError(
                        f"basename collision on {name!r} between non-identical files:\n"
                        f"  {winner.relpath}\n  {other.relpath}\n"
                        "The spec requires byte-identity before a collision may be collapsed."
                    )
        kept.append(winner)
        for other in group[1:]:
            counts["byte_identical_duplicate"] += 1
            res.excluded["byte_identical_duplicate"].append(other.relpath)
    return kept


def apply_admission(spec: DatasetSpec, records: list[VideoRecord], res: IngestResult) -> list[VideoRecord]:
    """Keep only videos the grid can give min_frames views of.

    The single mechanical, label-blind inclusion rule. It reads duration, trim and fps --
    never an image, a label or a model output.
    """
    admitted: list[VideoRecord] = []
    for rec in records:
        rec.n_frames = spec.n_frames(rec.duration_ms)
        if rec.n_frames >= spec.min_frames:
            admitted.append(rec)
        else:
            res.excluded["below_min_frames"].append(
                f"{rec.relpath}\tduration_ms={rec.duration_ms}\tframes={rec.n_frames}"
            )
    res.counts["below_min_frames"] = len(records) - len(admitted)
    res.counts["admitted"] = len(admitted)
    return admitted
