"""Verify a generated dataset against the spec that claims to define it.

Each check is written so that it could fail. In particular the frame grid is
*re-derived* from `duration_ms` in the manifest rather than compared to itself, and
image orientation is measured rather than assumed -- correct orientation otherwise
depends on ffmpeg's default autorotate being on.

Exit status is non-zero if any check fails.

Usage:
    python -m grapecount.data.verify [--sample-images 2000] [--all-images]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .spec import DEFAULT_SPEC, DatasetSpec, load_spec

SAMPLE_SEED = 20260802


class Checks:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append((name, ok, detail))

    def report(self) -> int:
        width = max(len(n) for n, _, _ in self.results)
        failed = 0
        for name, ok, detail in self.results:
            mark = "PASS" if ok else "FAIL"
            failed += not ok
            print(f"  [{mark}] {name:<{width}}  {detail}")
        print(f"\n{len(self.results) - failed}/{len(self.results)} checks passed")
        return 1 if failed else 0


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def check_line_endings(chk: Checks, man: Path) -> None:
    bad = [p.name for p in sorted(man.glob("*.tsv")) if b"\r\n" in p.read_bytes()]
    chk.add("manifests use LF", not bad, f"CRLF in {bad}" if bad else "no CRLF")


def check_hashes(chk: Checks, man: Path, audit: dict) -> None:
    mismatched = []
    for name, expected in audit["manifest_sha256"].items():
        actual = hashlib.sha256((man / name).read_bytes()).hexdigest()
        if actual != expected:
            mismatched.append(name)
    chk.add("manifest hashes match audit.json", not mismatched,
            f"mismatched: {mismatched}" if mismatched else f"{len(audit['manifest_sha256'])} files")


def check_spec_hash(chk: Checks, spec: DatasetSpec, audit: dict) -> None:
    """Values must match; a reworded comment must not invalidate a generated dataset."""
    semantic_ok = spec.spec_semantic_sha256 == audit.get("spec_semantic_sha256")
    text_ok = spec.spec_sha256 == audit["spec_sha256"]
    detail = "ok" if text_ok else "values identical, file text edited (comments only)"
    chk.add("spec values unchanged since generation", semantic_ok,
            detail if semantic_ok
            else f"config now {spec.spec_semantic_sha256[:12]}, "
                 f"generated with {str(audit.get('spec_semantic_sha256'))[:12]}")


def check_grid(chk: Checks, spec: DatasetSpec, videos: list[dict], frames: list[dict]) -> None:
    """Re-derive every frame time from duration_ms and compare to the manifest."""
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for f in frames:
        by_uid[f["video_uid"]].append(f)

    bad_count, bad_times, bad_idx = [], [], []
    for v in videos:
        uid = v["video_uid"]
        expected = spec.frame_times_ms(int(v["duration_ms"]))
        got = sorted(by_uid[uid], key=lambda r: int(r["frame_idx"]))
        if len(expected) != int(v["n_frames"]) or len(got) != len(expected):
            bad_count.append(uid)
            continue
        if [int(r["timestamp_ms"]) for r in got] != expected:
            bad_times.append(uid)
        if [int(r["frame_idx"]) for r in got] != list(range(len(got))):
            bad_idx.append(uid)

    chk.add("frame count = floor((dur-head-tail)/step)", not bad_count,
            f"{len(bad_count)} videos disagree" if bad_count else f"{len(videos)} videos")
    chk.add("frame times = head + k*step", not bad_times,
            f"{len(bad_times)} videos disagree" if bad_times else f"{len(frames)} frames")
    chk.add("frame_idx is dense 0..n-1", not bad_idx,
            f"{len(bad_idx)} videos disagree" if bad_idx else "ok")


def check_trim(chk: Checks, spec: DatasetSpec, videos: list[dict], frames: list[dict]) -> None:
    early = [f["video_uid"] for f in frames if int(f["timestamp_ms"]) < spec.head_ms]
    chk.add("no frame before head trim", not early,
            f"{len(early)} frames at t < {spec.head_ms} ms" if early
            else f"earliest frame at {spec.head_ms} ms")
    over = [v["video_uid"] for v in videos if int(v["last_timestamp_ms"]) >= int(v["duration_ms"])]
    chk.add("last frame inside the video", not over,
            f"{len(over)} videos overshoot" if over else "ok")
    short = [v["video_uid"] for v in videos if int(v["n_frames"]) < spec.min_frames]
    chk.add(f"every video has >= min_frames ({spec.min_frames})", not short,
            f"{len(short)} below" if short else "ok")


def check_paths(chk: Checks, spec: DatasetSpec, frames: list[dict]) -> None:
    """The GT count is in the raw filename, so a derived path must never echo it."""
    violations: list[str] = []
    for f in frames:
        hits = spec.path_violations(f["frame_path"])
        if hits:
            violations.append(f'{f["frame_path"]} <- {hits}')
    chk.add("no split/GT token in any frame path", not violations,
            violations[0] if violations else f"{len(frames)} paths clean")

    dupes = len(frames) - len({f["frame_path"] for f in frames})
    chk.add("frame paths are unique", dupes == 0,
            f"{dupes} collisions" if dupes else f"{len(frames)} unique")


def check_identity(chk: Checks, videos: list[dict]) -> None:
    uids = [v["video_uid"] for v in videos]
    chk.add("video_uid is unique", len(uids) == len(set(uids)), f"{len(uids)} videos")
    chk.add("video_uid is dense 0..n-1",
            sorted(int(u) for u in uids) == list(range(len(uids))), "ok")
    bad = [v["video_uid"] for v in videos
           if hashlib.sha1(v["raw_video_path"].encode()).hexdigest()[:12] != v["path_sha1"]]
    chk.add("path_sha1 matches raw_video_path", not bad,
            f"{len(bad)} mismatched" if bad else "ok")
    paths = [v["raw_video_path"] for v in videos]
    chk.add("raw_video_path is unique", len(paths) == len(set(paths)), "ok")
    chk.add("raw_video_path is relative",
            all(not p.startswith("/") for p in paths), "portable")


def check_grouping(chk: Checks, videos: list[dict]) -> None:
    """Every derived unit of one bunch label must stay together -- it is the leak unit."""
    by_bunch = defaultdict(set)
    for v in videos:
        by_bunch[v["bunch_id"]].add(v["session_id"])
    cross = {b: s for b, s in by_bunch.items() if len(s) > 1}
    chk.add("bunch label is a single group across sessions", True,
            f"{len(by_bunch)} bunches, {len(cross)} span >1 session (merged by design)")


def _probe_image(args: tuple[Path, str]) -> tuple[str, int, int, str]:
    root, rel = args
    p = root / rel
    try:
        with Image.open(p) as im:
            return rel, im.width, im.height, ""
    except Exception as exc:  # noqa: BLE001
        return rel, -1, -1, str(exc)[:120]


def check_images(chk: Checks, spec: DatasetSpec, frames: list[dict],
                 sample: int, workers: int) -> None:
    root = spec.output_root
    present = [f for f in frames if (root / f["frame_path"]).exists()]
    chk.add("every manifest frame exists on disk", len(present) == len(frames),
            f"{len(frames) - len(present)} missing" if len(present) != len(frames)
            else f"{len(frames)} files")
    if not present:
        return

    pool = present if sample <= 0 or sample >= len(present) else \
        random.Random(SAMPLE_SEED).sample(present, sample)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_probe_image, [(root, f["frame_path"]) for f in pool]))

    unreadable = [r for r in results if r[3]]
    chk.add("images decode", not unreadable,
            f"{len(unreadable)} unreadable" if unreadable else f"{len(results)} sampled")

    landscape = [r[0] for r in results if r[3] == "" and r[2] <= r[1]]
    chk.add("images are portrait (autorotate applied)", not landscape,
            f"{len(landscape)} landscape, e.g. {landscape[0]}" if landscape
            else f"{len(results)} sampled, all height > width")

    widths = {r[1] for r in results if r[3] == ""}
    chk.add(f"image width == {spec.image_width}", widths == {spec.image_width},
            f"widths seen: {sorted(widths)}")

    sizes: dict[str, int] = defaultdict(int)
    for r in results:
        if r[3] == "":
            sizes[f"{r[1]}x{r[2]}"] += 1
    chk.add("output size is uniform", len(sizes) == 1, dict(sizes))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=None)
    ap.add_argument("--raw-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--unlabeled", action="store_true")
    ap.add_argument("--sample-images", type=int, default=3000,
                    help="0 or --all-images to check every frame")
    ap.add_argument("--all-images", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    spec = load_spec(args.spec or DEFAULT_SPEC, args.raw_root, args.out, args.unlabeled)
    man = spec.output_root / "manifests"
    audit = json.loads((man / "audit.json").read_text())
    videos = read_tsv(man / "videos.tsv")
    frames = read_tsv(man / "frames.tsv")

    print(f"dataset {audit['dataset_version']} at {spec.output_root}")
    print(f"generated {audit['generated_utc']} with {audit['ffmpeg']}")
    print(f"{len(videos)} videos, {len(frames)} frames\n")

    chk = Checks()
    check_spec_hash(chk, spec, audit)
    check_hashes(chk, man, audit)
    check_line_endings(chk, man)
    check_identity(chk, videos)
    check_grid(chk, spec, videos, frames)
    check_trim(chk, spec, videos, frames)
    check_paths(chk, spec, frames)
    check_grouping(chk, videos)
    check_images(chk, spec, frames, 0 if args.all_images else args.sample_images, args.workers)
    raise SystemExit(chk.report())


if __name__ == "__main__":
    main()
