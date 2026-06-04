"""Unit tests for Phase D temporal-chunking coordinate threading (model-free).

Guards the highest-risk piece of Phase D — the CLEAN/NOISY coord threading — in
both regimes:

- SEQUENTIAL (co_temporal=False, legacy): clean [anchor..anchor+k_lat), noisy
  [anchor+k_lat..anchor+k_lat+w_lat) — disjoint coords.
- CO-TEMPORAL (co_temporal=True, default — Lever 1): noisy starts at the clean
  base, so the first k_lat noisy frames COINCIDE with the clean reference
  (video_edit's trained regime) and [anchor+k_lat..anchor+w_lat) extend.

Both checked in the mRoPE position_ids and the latent_pos_embed indices, plus
the block mask (NOISY attends CLEAN, not vice-versa). No model/weights/tokenizer.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from lance_mlx.pipeline.t2v_long import (
    long_lpe_indices, long_position_ids, long_block_mask,
)
from lance_mlx.pipeline.t2v import MAX_LATENT_SIDE

K_LAT, W_LAT, H_LAT, W_GRID, ANCHOR = 2, 4, 2, 2, 2000


def _layout():
    """A plausible [text | clean | gap | noisy | tail] token layout."""
    n_clean = K_LAT * H_LAT * W_GRID
    n_noisy = W_LAT * H_LAT * W_GRID
    p_text = 5
    clean = list(range(p_text, p_text + n_clean))
    gap = 3
    noisy = list(range(clean[-1] + 1 + gap, clean[-1] + 1 + gap + n_noisy))
    T = noisy[-1] + 2
    return T, clean, noisy


def test_lpe_frames_are_sequential():
    clean, noisy = long_lpe_indices(K_LAT, W_LAT, H_LAT, W_GRID)
    assert clean.shape[0] == K_LAT * H_LAT * W_GRID
    assert noisy.shape[0] == W_LAT * H_LAT * W_GRID
    # NOISY frame indexing continues AFTER the clean frames (k_lat offset).
    assert int(noisy[0]) == K_LAT * (MAX_LATENT_SIDE ** 2)
    assert int(clean[0]) == 0
    # frame stride = 64² between consecutive frames (cell 0 of each frame).
    fpf = H_LAT * W_GRID
    assert int(clean[fpf]) - int(clean[0]) == MAX_LATENT_SIDE ** 2


def test_position_ids_temporal_is_sequential_local_window():
    T, clean, noisy = _layout()
    pos = np.array(long_position_ids(
        T, clean, noisy, k_lat=K_LAT, w_lat=W_LAT, h_lat=H_LAT, w_grid=W_GRID, anchor=ANCHOR))
    clean_t = sorted(set(pos[0, 0, clean].tolist()))
    noisy_t = sorted(set(pos[0, 0, noisy].tolist()))
    # CLEAN occupies [anchor, anchor+k_lat); NOISY continues [anchor+k_lat, ...).
    assert clean_t == list(range(ANCHOR, ANCHOR + K_LAT))
    assert noisy_t == list(range(ANCHOR + K_LAT, ANCHOR + K_LAT + W_LAT))
    # THE sequential property: noisy starts exactly where clean ends + 1.
    assert min(noisy_t) == max(clean_t) + 1


def test_position_ids_spatial_grid_shared_and_bounded():
    T, clean, noisy = _layout()
    pos = np.array(long_position_ids(
        T, clean, noisy, k_lat=K_LAT, w_lat=W_LAT, h_lat=H_LAT, w_grid=W_GRID, anchor=ANCHOR))
    base = clean[0]
    # height/width coords span base..base+side-1 within each block (both blocks share the grid).
    for block in (clean, noisy):
        assert sorted(set(pos[1, 0, block].tolist())) == list(range(base, base + H_LAT))
        assert sorted(set(pos[2, 0, block].tolist())) == list(range(base, base + W_GRID))


def test_block_mask_noisy_sees_clean_but_not_reverse():
    T, clean, noisy = _layout()
    m = np.array(long_block_mask(T, clean, noisy, dtype=mx.float32))
    ALLOW = 0.0
    # NOISY query attends a CLEAN key (causal: clean precedes noisy) → allowed.
    assert m[noisy[0], clean[0]] == ALLOW
    # CLEAN query must NOT attend a NOISY key (future) → blocked.
    assert m[clean[0], noisy[0]] < 0
    # within-block bidirectional: noisy attends a later noisy key.
    assert m[noisy[0], noisy[-1]] == ALLOW
    # within clean bidirectional too.
    assert m[clean[0], clean[-1]] == ALLOW


# ── co-temporal regime (co_temporal=True, the video_edit-aligned default) ────

def test_cotemporal_lpe_noisy_overlaps_clean():
    clean, noisy = long_lpe_indices(K_LAT, W_LAT, H_LAT, W_GRID, co_temporal=True)
    assert clean.shape[0] == K_LAT * H_LAT * W_GRID
    assert noisy.shape[0] == W_LAT * H_LAT * W_GRID
    # NOISY frame indexing starts at 0 (NOT k_lat-offset) → coincides with CLEAN.
    assert int(noisy[0]) == 0
    assert int(clean[0]) == 0
    # The first k_lat noisy frames share their lpe index with the clean frames.
    n_overlap = K_LAT * H_LAT * W_GRID
    assert noisy[:n_overlap].tolist() == clean.tolist()
    # The extension frames continue past the overlap (frame k_lat onward).
    fpf = H_LAT * W_GRID
    assert int(noisy[n_overlap]) == K_LAT * (MAX_LATENT_SIDE ** 2)
    assert int(noisy[fpf]) - int(noisy[0]) == MAX_LATENT_SIDE ** 2


def test_cotemporal_position_ids_noisy_starts_at_clean_base():
    T, clean, noisy = _layout()
    pos = np.array(long_position_ids(
        T, clean, noisy, k_lat=K_LAT, w_lat=W_LAT, h_lat=H_LAT, w_grid=W_GRID,
        anchor=ANCHOR, co_temporal=True))
    clean_t = sorted(set(pos[0, 0, clean].tolist()))
    noisy_t = sorted(set(pos[0, 0, noisy].tolist()))
    # CLEAN occupies [anchor, anchor+k_lat); NOISY now STARTS at anchor (overlap).
    assert clean_t == list(range(ANCHOR, ANCHOR + K_LAT))
    assert noisy_t == list(range(ANCHOR, ANCHOR + W_LAT))
    # THE co-temporal property: the noisy overlap shares clean's frame coords.
    assert min(noisy_t) == min(clean_t)
    # …and the extension reaches one frame per new latent past the overlap.
    assert max(noisy_t) == ANCHOR + W_LAT - 1


def test_cotemporal_spatial_grid_unchanged():
    """The spatial (h,w) axes are regime-independent — only the t-axis shifts."""
    T, clean, noisy = _layout()
    base = clean[0]
    for ct in (False, True):
        pos = np.array(long_position_ids(
            T, clean, noisy, k_lat=K_LAT, w_lat=W_LAT, h_lat=H_LAT, w_grid=W_GRID,
            anchor=ANCHOR, co_temporal=ct))
        for block in (clean, noisy):
            assert sorted(set(pos[1, 0, block].tolist())) == list(range(base, base + H_LAT))
            assert sorted(set(pos[2, 0, block].tolist())) == list(range(base, base + W_GRID))
