"""Tests for the dataset frame grid.

The grid decides which frames exist, so it is the piece most worth pinning down. The
knife-edge cases below have container durations that sit a few tenths of a
millisecond below an exact frame boundary, and a float implementation gains or loses a
frame on them depending on how the duration was rounded upstream.

    python -m pytest tests/test_spec.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grapecount.data.spec import load_spec, qscale_for  # noqa: E402

SPEC = load_spec()


# --------------------------------------------------------------------- basics
def test_spec_matches_the_expected_values():
    assert SPEC.version == "v2.1"
    assert (SPEC.head_sec, SPEC.tail_sec) == (2.0, 0.0)
    assert SPEC.fps == 4.0
    assert SPEC.min_frames == 16
    assert SPEC.image_width == 640


def test_grid_constants_are_whole_milliseconds():
    assert SPEC.step_ms == 250
    assert SPEC.head_ms == 2000
    assert SPEC.tail_ms == 0


def test_qscale_mapping():
    """JPEG quality maps to ffmpeg -q:v as int(31 - Q*0.3); at Q=92 that is 3, not 4."""
    assert qscale_for(92) == 3


# ------------------------------------------------------------------ grid math
@pytest.mark.parametrize(
    "duration_ms,expected",
    [
        (2000, 0),      # exactly the trim, nothing left
        (2250, 1),
        (6000, 16),     # exactly at the admission threshold
        (12000, 40),
        (10134, 32),
    ],
)
def test_n_frames(duration_ms, expected):
    assert SPEC.n_frames(duration_ms) == expected


def test_frame_times_are_exact_multiples_of_the_step():
    times = SPEC.frame_times_ms(12000)
    assert times[0] == SPEC.head_ms
    assert times == [2000 + 250 * k for k in range(40)]
    assert all(t % SPEC.step_ms == 0 for t in times)


def test_last_frame_never_reaches_the_end_of_the_video():
    """A seek must never land past the final decodable frame."""
    for duration_ms in range(2000, 65_000, 37):
        times = SPEC.frame_times_ms(duration_ms)
        if times:
            assert times[-1] <= duration_ms - SPEC.step_ms


def test_no_frame_precedes_the_head_trim():
    for duration_ms in (2001, 6000, 12833, 64800):
        assert all(t >= SPEC.head_ms for t in SPEC.frame_times_ms(duration_ms))


# ------------------------------------------------------- the knife-edge cases
@pytest.mark.parametrize(
    "name,duration_s,duration_ms,expected_frames",
    [
        ("video_a", 23.499600, 23500, 86),
        ("video_b", 34.249900, 34250, 129),
        ("video_c", 18.249900, 18250, 65),
        ("video_d", 11.999700, 12000, 40),
    ],
)
def test_durations_just_below_a_frame_boundary(name, duration_s, duration_ms, expected_frames):
    """Rounding to whole ms first is what makes these deterministic.

    Truncating the float product instead would lose a frame on every one of them.
    """
    assert SPEC.duration_ms(duration_s) == duration_ms
    assert SPEC.n_frames(duration_ms) == expected_frames
    naive = int((duration_s - SPEC.head_sec - SPEC.tail_sec) * SPEC.fps)
    assert naive == expected_frames - 1, "this case is only interesting if the float form differs"


# ------------------------------------------------------------------ admission
@pytest.mark.parametrize(
    "duration_ms,admitted",
    [(900, False), (1633, False), (1767, False), (5132, False), (5667, False), (6000, True)],
)
def test_admission_threshold(duration_ms, admitted):
    """5132 ms (DB5) gives 12 frames and is rejected; 6000 ms gives exactly 16 and is kept."""
    assert SPEC.admits(duration_ms) is admitted


def test_admission_is_label_blind():
    """The rule may read duration, trim and fps -- nothing else."""
    import inspect

    source = inspect.getsource(SPEC.admits) + inspect.getsource(SPEC.n_frames)
    for forbidden in ("gt_count", "label", "bunch", "pred", "model", "image"):
        assert forbidden not in source


# ----------------------------------------------------------------- path rules
def test_frame_path_carries_no_label_or_split():
    rel = SPEC.frame_relpath("A0", 7, 3, 2750)
    assert rel == "frames/A0/v00007/f000003__t00002750ms.jpg"
    assert SPEC.path_violations(rel) == []


@pytest.mark.parametrize("bad", ["frames/A0_gt_49/v1/f0.jpg", "frames/test/A0/f0.jpg",
                                 "frames/fold0/A0/f0.jpg", "frames/A0/count_49.jpg"])
def test_forbidden_tokens_are_caught(bad):
    assert SPEC.path_violations(bad)
