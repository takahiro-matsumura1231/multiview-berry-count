"""Generate the frame dataset from configs/dataset.yaml and nothing else.

Two stages, deliberately separable so the cheap one can be checked before the expensive
one runs:

    --manifests-only   scan, probe durations, admit, write the manifests   (~1 min)
    (default)          the above, then extract every frame                 (~30 min)

Nothing is written outside `output.root`.

Usage:
    python -m grapecount.data.build --manifests-only
    python -m grapecount.data.build --workers 24
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from .ingest import IngestResult, VideoRecord, apply_admission, scan
from .spec import DEFAULT_SPEC, DatasetSpec, load_spec, qscale_for

VIDEO_FIELDS = [
    "video_uid", "bunch_id", "session_id", "stage_id", "gt_count",
    "raw_video_path", "path_sha1", "duration_ms", "n_frames",
    "first_timestamp_ms", "last_timestamp_ms",
]
FRAME_FIELDS = ["video_uid", "bunch_id", "frame_idx", "timestamp_ms", "tau", "frame_path"]


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> str:
    """Write LF-terminated TSV and return its sha256. csv defaults to CRLF, which
    silently mis-parses in awk/cut pipelines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_duration(args: tuple[Path, str]) -> tuple[str, int, str]:
    raw_root, rel = args
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(raw_root / rel)],
            capture_output=True, text=True, timeout=120, check=True,
        ).stdout.strip()
        return rel, round(float(out) * 1000.0), ""
    except Exception as exc:  # noqa: BLE001
        return rel, -1, str(exc)[:200]


def tool_version(name: str) -> str:
    try:
        return subprocess.run([name, "-version"], capture_output=True, text=True,
                              check=True).stdout.splitlines()[0]
    except Exception:  # noqa: BLE001
        return "unknown"


def extract_one(job: tuple[str, str, int, int, bool]) -> tuple[str, str, bool]:
    """One frame. `-ss` before `-i` so ffmpeg input-seeks instead of decoding from 0.

    `-threads 1` is required for reproducibility, not for speed. Some source videos
    contain a corrupt frame, and with multi-threaded decoding ffmpeg's
    error concealment gives a *different image on every run* -- three extractions of the
    same timestamp produced two distinct files. Single-threaded decoding is deterministic
    there, and with many parallel workers it is also faster, since each ffmpeg no longer
    fights the others for cores.

    Rotation is intentionally left to ffmpeg's display-matrix handling (no -noautorotate):
    most videos are stored landscape with a -90 matrix. The verifier asserts the
    result is portrait rather than trusting it.

    Returns (path, error, changed).
    """
    src, dst, ts_ms, qscale, force = job
    out = Path(dst)
    previous = out.read_bytes() if (force and out.exists()) else None
    if out.exists() and not force:
        return dst, "", False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1",
        "-ss", f"{ts_ms / 1000.0:.3f}", "-i", src,
        "-frames:v", "1", "-vf", "scale=640:-1", "-q:v", str(qscale), "-y", str(tmp),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True)
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced no output")
        changed = previous is not None and tmp.read_bytes() != previous
        tmp.replace(out)
        return dst, "", changed
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return dst, str(exc)[:200], False


def build_manifests(spec: DatasetSpec, workers: int) -> tuple[list[VideoRecord], IngestResult, dict]:
    t0 = time.time()
    print(f"[scan] {spec.raw_root}", file=sys.stderr)
    res = scan(spec)
    print(f"[scan] name-conforming={res.counts.get('name_conforming')} "
          f"after dedup={res.counts.get('deduplicated')} ({time.time() - t0:.1f}s)", file=sys.stderr)

    print(f"[probe] durations for {len(res.admitted)} videos", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        by_rel = {r.relpath: r for r in res.admitted}
        for rel, dur_ms, err in ex.map(probe_duration, [(spec.raw_root, r.relpath) for r in res.admitted]):
            if err:
                raise RuntimeError(f"ffprobe failed on {rel}: {err}")
            by_rel[rel].duration_ms = dur_ms

    admitted = apply_admission(spec, res.admitted, res)
    for i, rec in enumerate(admitted):
        rec.video_uid = i   # renumber densely over the admitted set

    video_rows, frame_rows = [], []
    for rec in admitted:
        times = spec.frame_times_ms(rec.duration_ms)
        lo, hi = times[0], times[-1]
        span = (hi - lo) or 1
        video_rows.append({
            "video_uid": rec.video_uid, "bunch_id": rec.bunch_id, "session_id": rec.session_id,
            "stage_id": rec.stage_id, "gt_count": rec.gt_count, "raw_video_path": rec.relpath,
            "path_sha1": hashlib.sha1(rec.relpath.encode()).hexdigest()[:12],
            "duration_ms": rec.duration_ms, "n_frames": len(times),
            "first_timestamp_ms": lo, "last_timestamp_ms": hi,
        })
        for idx, ts in enumerate(times):
            frame_rows.append({
                "video_uid": rec.video_uid, "bunch_id": rec.bunch_id, "frame_idx": idx,
                "timestamp_ms": ts, "tau": f"{(ts - lo) / span:.4f}",
                "frame_path": spec.frame_relpath(rec.bunch_id, rec.video_uid, idx, ts),
            })

    root = spec.output_root
    man = root / "manifests"
    hashes = {
        "videos.tsv": write_tsv(man / "videos.tsv", video_rows, VIDEO_FIELDS),
        "frames.tsv": write_tsv(man / "frames.tsv", frame_rows, FRAME_FIELDS),
    }
    (man / "excluded.json").write_text(
        json.dumps({k: sorted(v) for k, v in sorted(res.excluded.items())},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    hashes["excluded.json"] = hashlib.sha256((man / "excluded.json").read_bytes()).hexdigest()

    bunches = defaultdict(set)
    for r in video_rows:
        bunches[r["bunch_id"]].add(r["gt_count"])
    audit = {
        "dataset_version": spec.version,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec_sha256": spec.spec_sha256,
        "spec_semantic_sha256": spec.spec_semantic_sha256,
        "manifest_sha256": hashes,
        "ffmpeg": tool_version("ffmpeg"),
        "ffprobe": tool_version("ffprobe"),
        "raw_root": str(spec.raw_root),
        "counts": {
            **res.counts,
            "videos": len(video_rows),
            "bunches": len(bunches),
            "frames": len(frame_rows),
            "bunches_ge3_distinct_gt": sum(1 for v in bunches.values() if len(v) >= 3),
            "bunches_ge5_distinct_gt": sum(1 for v in bunches.values() if len(v) >= 5),
        },
        "spec_echo": {
            "head_sec": spec.head_sec, "tail_sec": spec.tail_sec, "fps": spec.fps,
            "min_frames": spec.min_frames, "image_width": spec.image_width,
            "jpeg_quality": spec.jpeg_quality,
        },
    }
    (man / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    print(f"[manifests] {man}", file=sys.stderr)
    for k, v in audit["counts"].items():
        print(f"    {k:<28} {v}", file=sys.stderr)
    return admitted, res, audit


def extract_frames(spec: DatasetSpec, workers: int, force: bool = False) -> None:
    man = spec.output_root / "manifests"
    rows = list(csv.DictReader((man / "frames.tsv").open(newline=""), delimiter="\t"))
    vids = {r["video_uid"]: r["raw_video_path"]
            for r in csv.DictReader((man / "videos.tsv").open(newline=""), delimiter="\t")}
    q = qscale_for(spec.jpeg_quality)
    jobs = [(str(spec.raw_root / vids[r["video_uid"]]),
             str(spec.output_root / r["frame_path"]), int(r["timestamp_ms"]), q, force)
            for r in rows]
    todo = jobs if force else [j for j in jobs if not Path(j[1]).exists()]
    print(f"[extract] {len(todo)} of {len(jobs)} frames to write"
          f"{' (forced re-extract)' if force else ''}, {workers} workers", file=sys.stderr)

    errors, changed, done, t0 = [], 0, 0, time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for dst, err, was_changed in ex.map(extract_one, todo, chunksize=16):
            done += 1
            changed += was_changed
            if err:
                errors.append((dst, err))
            if done % 5000 == 0:
                rate = done / (time.time() - t0)
                print(f"[extract] {done}/{len(todo)}  {rate:.0f}/s  "
                      f"eta {(len(todo) - done) / max(rate, 1e-9) / 60:.1f} min", file=sys.stderr)
    print(f"[extract] done in {(time.time() - t0) / 60:.1f} min, errors={len(errors)}"
          + (f", bytes changed on {changed} frames" if force else ""), file=sys.stderr)
    for dst, err in errors[:20]:
        print(f"    FAIL {dst}: {err}", file=sys.stderr)
    if errors:
        (man / "extract_errors.json").write_text(
            json.dumps([{"path": d, "error": e} for d, e in errors], indent=1), encoding="utf-8")
        raise SystemExit(f"{len(errors)} frames failed to extract")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=None)
    ap.add_argument("--raw-root", type=Path, default=None,
                    help="directory holding the videos (overrides source.raw_root)")
    ap.add_argument("--out", type=Path, default=None,
                    help="dataset output directory (overrides output.root)")
    ap.add_argument("--unlabeled", action="store_true",
                    help="admit any video name; counts are left empty (prediction only)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--manifests-only", action="store_true")
    ap.add_argument("--force-extract", action="store_true",
                    help="re-extract every frame and report how many changed")
    args = ap.parse_args()
    spec = load_spec(args.spec or DEFAULT_SPEC, args.raw_root, args.out, args.unlabeled)
    if not spec.raw_root.is_dir():
        raise SystemExit(f"raw video directory not found: {spec.raw_root}")
    print(f"[spec] version={spec.version} sha256={spec.spec_sha256[:12]} "
          f"head={spec.head_sec} tail={spec.tail_sec} fps={spec.fps} "
          f"min_frames={spec.min_frames}", file=sys.stderr)
    build_manifests(spec, args.workers)
    if not args.manifests_only:
        extract_frames(spec, args.workers, force=args.force_extract)


if __name__ == "__main__":
    main()
