"""Pure-logic tests for B8 DIS frame interpolation (scripts/_t2v_npy_to_mp4.py).

Fast, CPU-only, no model load — synthetic 32x32 frames. Validates the frame
arithmetic and invariants that the production .mp4 path relies on.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "_t2v_npy_to_mp4", Path(__file__).resolve().parent.parent / "scripts" / "_t2v_npy_to_mp4.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
dis_interpolate = _mod.dis_interpolate


def _ramp_clip(t=5, n=32):
    """A clip with real translational motion: a bright block sliding right."""
    frames = np.full((t, n, n, 3), 30, np.uint8)
    for i in range(t):
        x = 2 + i * 3
        frames[i, 8:20, x:x + 6] = 220
    return frames


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_output_length(factor):
    clip = _ramp_clip(t=5)
    out = dis_interpolate(clip, factor)
    assert len(out) == (5 - 1) * factor + 1


def test_dtype_and_shape_preserved():
    clip = _ramp_clip(t=4)
    out = dis_interpolate(clip, 3)
    assert out.dtype == np.uint8
    assert out.shape[1:] == clip.shape[1:]


def test_endpoints_unchanged():
    clip = _ramp_clip(t=4)
    out = dis_interpolate(clip, 3)
    assert np.array_equal(out[0], clip[0])
    assert np.array_equal(out[-1], clip[-1])


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_all_originals_byte_preserved(factor):
    """Lossless-lens bar for B8 (LOSSLESS_PLAN §3): interpolation is INSERT-ONLY.
    Every source frame must survive byte-identical at output stride `factor`;
    the (factor-1) inserted frames live strictly between them. This guards the
    'originals untouched' guarantee that lets B8 count as lossless w.r.t. the
    generated frames (it invents in-betweens, never rewrites a real frame)."""
    clip = _ramp_clip(t=6)
    out = dis_interpolate(clip, factor)
    # out[::factor] are exactly the originals, in order, unmodified.
    assert np.array_equal(out[::factor], clip)


def test_factor_one_is_identity():
    clip = _ramp_clip(t=4)
    out = dis_interpolate(clip, 1)
    assert np.array_equal(out, clip)


def test_static_clip_stays_static():
    # zero motion -> flow ~0 -> every interpolated frame equals the source
    clip = np.full((3, 32, 32, 3), 120, np.uint8)
    clip[:, 10:20, 10:20] = 200
    out = dis_interpolate(clip, 3)
    for f in out:
        assert np.array_equal(f, clip[0])


def test_interpolated_frame_is_between():
    # the inserted frame's moving block should sit between the two endpoints'
    # positions, not outside them (motion-compensated, not a hard cut)
    clip = _ramp_clip(t=2)            # block at x=2 and x=5
    out = dis_interpolate(clip, 2)    # one frame inserted at t=0.5
    mid = out[1]
    col_energy = (mid[8:20].mean(axis=(0, 2)) > 100)  # bright columns
    cols = np.where(col_energy)[0]
    assert len(cols) > 0
    center = cols.mean()
    assert 2 <= center <= 11           # between left(~2) and right(~8) extents
