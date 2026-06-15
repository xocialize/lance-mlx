"""Pure-CPU tests for spatial MultiDiffusion windowing primitives (no model load).

Covers the index/pad math that is silently corruptible (SPATIAL_MD_PLAN.md risks):
window coverage (no 1/0 in inv_counts), lpe-bound (<4096), 2-D pad-extent placement,
taper separability, and the per-window in-grid guard.
"""

import mlx.core as mx
import numpy as np
import pytest

from lance_mlx.pipeline.t2i_multidiff import (
    MAX_LATENT_SIDE,
    _grid_window_starts,
    _window_lpe,
    _taper2d,
    _inv_counts,
    _pad_window,
    _freenoise_spatial,
    _upsample_latent,
)
from lance_mlx.pipeline._md_common import window_starts as _axis_starts


# ----------------------------------------------------------------- axis starts

def test_axis_starts_cover_and_pullback():
    # 96 grid, window 64, overlap 16, stride 48: s=0, then min(48, 96-64=32)=32
    assert _axis_starts(96, 64, 16) == [0, 32]
    # last window always ends exactly at total
    starts = _axis_starts(96, 64, 16)
    assert starts[-1] + 64 == 96


def test_axis_starts_grid_just_over_window():
    # 68 grid (1088²), window 64: starts [0, 4], both windows end-aligned
    assert _axis_starts(68, 64, 16) == [0, 4]
    assert _axis_starts(68, 64, 16)[-1] + 64 == 68


def test_axis_starts_single_window_when_equal():
    assert _axis_starts(64, 64, 16) == [0]


def test_axis_starts_rejects_bad_args():
    with pytest.raises(ValueError):
        _axis_starts(40, 64, 16)        # window > grid
    with pytest.raises(ValueError):
        _axis_starts(96, 64, 64)        # overlap >= window (stride < 1)


# ------------------------------------------------------------- grid window starts

def test_grid_window_starts_product():
    starts = _grid_window_starts(96, 96, 64, 64, 16)  # h_win=w_win=64
    # 2 row starts × 2 col starts = 4 windows
    assert starts == [(0, 0), (0, 32), (32, 0), (32, 32)]


def test_grid_window_in_grid_guard():
    # every window must fit the 64×64 lpe table
    for (r0, c0) in _grid_window_starts(96, 96, 64, 64, 16):
        assert r0 + 64 <= 96 and c0 + 64 <= 96


def test_grid_window_rejects_oversize_window():
    with pytest.raises(ValueError):
        _grid_window_starts(130, 130, 65, 65, 16)   # window > 64 per axis


# ------------------------------------------------------------------- window lpe

def test_window_lpe_bound_and_formula():
    lpe = np.array(_window_lpe(64, 64))
    assert lpe.max() < MAX_LATENT_SIDE * MAX_LATENT_SIDE   # < 4096, in table
    assert lpe.min() == 0
    # exact relative formula r*64+c, row-major
    expect = np.array([r * 64 + c for r in range(64) for c in range(64)])
    assert np.array_equal(lpe, expect)


def test_window_lpe_smaller_window():
    lpe = np.array(_window_lpe(48, 48))
    assert lpe.shape == (48 * 48,)
    assert lpe.max() == 47 * 64 + 47


# ----------------------------------------------------------------------- taper

def test_taper_uniform_is_ones():
    t = _taper2d(64, 64, "uniform")
    assert t.shape == (1, 1, 64, 64, 1)
    assert float(mx.min(t)) == 1.0 and float(mx.max(t)) == 1.0


def test_taper_cosine_separable_outer_product():
    h_win, w_win = 48, 64
    t = np.array(_taper2d(h_win, w_win, "cosine")).reshape(h_win, w_win)
    hh = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(h_win) + 1) / (h_win + 1))
    ww = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(w_win) + 1) / (w_win + 1))
    assert np.allclose(t, np.outer(hh, ww), atol=1e-6)
    assert (t > 0).all()   # floored away from zero — no covered cell gets weight 0


def test_taper_rejects_unknown():
    with pytest.raises(ValueError):
        _taper2d(64, 64, "hann")


# ------------------------------------------------------------------ inv_counts

def test_inv_counts_full_cover_no_inf():
    starts = _grid_window_starts(96, 96, 64, 64, 16)
    taper = _taper2d(64, 64, "uniform")
    inv = np.array(_inv_counts(starts, 96, 96, 64, 64, taper))
    assert inv.shape == (1, 1, 96, 96, 1)
    assert np.isfinite(inv).all()       # full coverage → no 1/0
    assert (inv > 0).all()


def test_inv_counts_overlap_region_weighted_down():
    # uniform taper: a cell covered by k windows must get inv_count = 1/k
    starts = _grid_window_starts(96, 96, 64, 64, 16)
    taper = _taper2d(64, 64, "uniform")
    inv = np.array(_inv_counts(starts, 96, 96, 64, 64, taper)).reshape(96, 96)
    # corner cell (0,0) covered by exactly 1 window → 1.0
    assert inv[0, 0] == pytest.approx(1.0)
    # centre cell (48,48) is in the overlap of all 4 windows → 1/4
    assert inv[48, 48] == pytest.approx(0.25)


def test_inv_counts_raises_on_gap():
    # two tiny non-touching windows leave a gap → must raise
    with pytest.raises(ValueError):
        _inv_counts([(0, 0)], 96, 96, 64, 64, _taper2d(64, 64, "uniform"))


# -------------------------------------------------------------------- pad spec

def test_pad_window_places_exactly():
    # the highest-leverage check: a ones-window padded into a zeros buffer must
    # land EXACTLY at [:, :, r0:r0+h_win, c0:c0+w_win, :] and nowhere else.
    h_lat = w_lat = 96
    h_win = w_win = 64
    r0, c0 = 32, 0
    v_win = mx.ones((1, 1, h_win, w_win, 3))
    pad = _pad_window(v_win, r0, c0, h_lat, w_lat)
    placed = np.array(mx.pad(v_win, pad))
    assert placed.shape == (1, 1, h_lat, w_lat, 3)
    # the window region is all ones
    assert placed[:, :, r0:r0 + h_win, c0:c0 + w_win, :].min() == 1.0
    # everything outside is zero — build a boolean mask of the window region
    mask = np.zeros((h_lat, w_lat), dtype=bool)
    mask[r0:r0 + h_win, c0:c0 + w_win] = True
    outside = placed[0, 0, ~mask, :]
    assert outside.size > 0 and outside.max() == 0.0


def test_pad_window_all_four_corners():
    h_lat = w_lat = 96
    h_win = w_win = 64
    for (r0, c0) in _grid_window_starts(96, 96, 64, 64, 16):
        v_win = mx.ones((1, 1, h_win, w_win, 1))
        placed = np.array(mx.pad(v_win, _pad_window(v_win, r0, c0, h_lat, w_lat)))
        total_ones = placed.sum()
        assert total_ones == h_win * w_win    # no spillover, no truncation
        assert placed[0, 0, r0, c0, 0] == 1.0
        assert placed[0, 0, r0 + h_win - 1, c0 + w_win - 1, 0] == 1.0


# ---------------------------------------------------------------- freenoise

def test_freenoise_shape_and_variance():
    fn = _freenoise_spatial(64, 64, 128, 128, 48, mx.float32, 42)
    assert fn.shape == (1, 1, 128, 128, 48)
    arr = np.array(fn)
    assert np.isfinite(arr).all()
    assert abs(arr.std() - 1.0) < 0.05      # unit-ish-variance normal
    assert abs(arr.mean()) < 0.05


def test_freenoise_tiles_share_base_vocabulary():
    # every FULL h_win×w_win tile must be a row/col permutation of the base block,
    # so it holds the SAME multiset of values (shared noise vocabulary).
    h_win = w_win = 64
    fn = np.array(_freenoise_spatial(h_win, w_win, 128, 128, 8, mx.float32, 7))[0, 0]
    base = fn[0:h_win, 0:w_win, :]                      # the identity (origin) tile
    base_sorted = np.sort(base.reshape(-1, 8), axis=0)
    for (r0, c0) in [(0, 64), (64, 0), (64, 64)]:
        tile = fn[r0:r0 + h_win, c0:c0 + w_win, :]
        tile_sorted = np.sort(tile.reshape(-1, 8), axis=0)
        assert np.allclose(base_sorted, tile_sorted)   # same values, permuted


def test_freenoise_deterministic_for_seed():
    a = np.array(_freenoise_spatial(64, 64, 96, 96, 4, mx.float32, 13))
    b = np.array(_freenoise_spatial(64, 64, 96, 96, 4, mx.float32, 13))
    assert np.array_equal(a, b)


# ------------------------------------------------------------- anchor upsample

def test_upsample_latent_shape_and_dtype():
    anchor = mx.random.normal((1, 1, 64, 64, 48))
    up = _upsample_latent(anchor, 96, 96)
    assert up.shape == (1, 1, 96, 96, 48)
    assert up.dtype == anchor.dtype


def test_upsample_latent_constant_preserved():
    # bilinear upsample of a constant field must stay that constant everywhere
    anchor = mx.full((1, 1, 64, 64, 48), 0.7)
    up = np.array(_upsample_latent(anchor, 128, 128))
    assert np.allclose(up, 0.7, atol=1e-5)


def test_upsample_latent_identity_when_same_size():
    anchor = mx.random.normal((1, 1, 64, 64, 4))
    up = _upsample_latent(anchor, 64, 64)
    assert np.allclose(np.array(up), np.array(anchor), atol=1e-5)
