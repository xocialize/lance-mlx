"""Pure-logic tests for the MultiDiffusion window tiling + coord shifts.

No model load — just the deterministic bookkeeping (window placement, blend
weights, lpe/position-id offsets). Catches off-by-one / coverage bugs before a
multi-minute GPU run.
"""
import numpy as np
import mlx.core as mx
import pytest

from lance_mlx.pipeline.t2v_multidiff import (
    _window_starts, _window_lpe_and_pos, _taper_weights, _freenoise_latents,
    _img2img_init, _img2img_dense, _upsample_latent_temporal,
)
from lance_mlx.pipeline.t2v import MAX_LATENT_SIDE
from lance_mlx.model.flow_head import timestep_schedule


def _coverage(total_t, starts, W):
    cov = np.zeros(total_t, dtype=int)
    for s in starts:
        cov[s:s + W] += 1
    return cov


def test_starts_cover_every_frame():
    starts = _window_starts(total_t=13, window_t=5, overlap_t=2)
    cov = _coverage(13, starts, 5)
    assert (cov >= 1).all(), f"uncovered frames: {np.where(cov < 1)[0]}"


def test_starts_uniform_window_and_clamped_last():
    # stride = 5-2 = 3 -> 0,3,6, then last pulled back to 13-5=8
    starts = _window_starts(total_t=13, window_t=5, overlap_t=2)
    assert starts == [0, 3, 6, 8], starts
    # every window is full-size and in-bounds
    for s in starts:
        assert 0 <= s and s + 5 <= 13


def test_starts_single_window_when_fits():
    assert _window_starts(total_t=4, window_t=4, overlap_t=2) == [0]


def test_starts_no_duplicate_last():
    # exact tiling: stride divides evenly, last clamp must not duplicate
    starts = _window_starts(total_t=9, window_t=5, overlap_t=1)  # stride 4 -> 0,4, clamp 4
    assert starts == [0, 4], starts


def test_starts_rejects_window_bigger_than_total():
    with pytest.raises(ValueError):
        _window_starts(total_t=3, window_t=5, overlap_t=2)


def test_starts_rejects_nonpositive_stride():
    with pytest.raises(ValueError):
        _window_starts(total_t=10, window_t=4, overlap_t=4)


def test_relative_coords_are_template_identity():
    W, h, w = 4, 2, 2
    lpe = mx.array([f * MAX_LATENT_SIDE**2 + r * MAX_LATENT_SIDE + c
                    for f in range(W) for r in range(h) for c in range(w)], dtype=mx.int32)
    pos = mx.array(np.stack([np.arange(W * h * w)] * 3)[:, None, :].astype(np.int32))
    lpe2, pos2 = _window_lpe_and_pos(lpe, pos, start=2, window_coords="relative")
    assert mx.array_equal(lpe2, lpe)
    assert mx.array_equal(pos2, pos)


def test_absolute_coords_shift_temporal_only():
    W, h, w = 3, 2, 2
    # template: t-axis = frame f, h-axis = r, w-axis = c (base 0)
    pos = np.zeros((3, 1, W * h * w), dtype=np.int32)
    for idx in range(W * h * w):
        f = idx // (h * w)
        r = (idx % (h * w)) // w
        c = (idx % (h * w)) % w
        pos[0, 0, idx] = f
        pos[1, 0, idx] = r
        pos[2, 0, idx] = c
    lpe = mx.array([f * MAX_LATENT_SIDE**2 + r * MAX_LATENT_SIDE + c
                    for f in range(W) for r in range(h) for c in range(w)], dtype=mx.int32)
    start = 5
    lpe2, pos2 = _window_lpe_and_pos(lpe, mx.array(pos), start=start, window_coords="absolute")
    pos2 = np.array(pos2)
    # temporal axis shifted by start, spatial axes unchanged
    assert (pos2[0, 0, :] == pos[0, 0, :] + start).all()
    assert (pos2[1, 0, :] == pos[1, 0, :]).all()
    assert (pos2[2, 0, :] == pos[2, 0, :]).all()
    # lpe frame term advanced by start*64^2, spatial term preserved
    assert mx.array_equal(lpe2, lpe + start * MAX_LATENT_SIDE**2)


def test_blend_weights_sum_to_one_after_normalization():
    total_t, W = 13, 5
    starts = _window_starts(total_t, W, overlap_t=2)
    counts = np.zeros(total_t)
    for s in starts:
        counts[s:s + W] += 1
    # normalized contribution per frame integrates to exactly 1 (uniform blend)
    assert (counts >= 1).all()
    contrib = np.zeros(total_t)
    for s in starts:
        contrib[s:s + W] += 1.0 / counts[s:s + W]
    assert np.allclose(contrib, 1.0), contrib


# --- B6: FreeNoise correlated init + cosine blend taper -----------------------

def test_taper_uniform_is_all_ones():
    w = np.array(_taper_weights(5, "uniform")).reshape(-1)
    assert w.shape == (5,)
    assert np.allclose(w, 1.0)


def test_taper_cosine_peaks_in_middle_and_is_positive():
    W = 7
    w = np.array(_taper_weights(W, "cosine")).reshape(-1)
    assert (w > 0).all(), f"a zero-weight frame would blow up the normalizer: {w}"
    mid = W // 2
    assert w[mid] > w[0] and w[mid] > w[-1]      # peaks in the middle
    assert w[0] == pytest.approx(w[-1])          # symmetric


def test_taper_w1_is_unit():
    assert np.allclose(np.array(_taper_weights(1, "cosine")).reshape(-1), 1.0)


def test_taper_invalid_kind_raises():
    with pytest.raises(ValueError):
        _taper_weights(5, "triangle")


def test_freenoise_shape_and_dtype():
    z = _freenoise_latents(W=5, total_t=13, h_lat=4, w_lat=4, C=3, dtype=mx.float32, seed=1)
    assert z.shape == (1, 13, 4, 4, 3)
    assert z.dtype == mx.float32


def test_freenoise_first_window_distinct():
    W = 5
    z = np.array(_freenoise_latents(W=W, total_t=20, h_lat=4, w_lat=4, C=3,
                                    dtype=mx.float32, seed=7))[0]
    base = z[:W].reshape(W, -1)
    # the W base frames are independent draws -> all pairwise distinct
    for i in range(W):
        for j in range(i + 1, W):
            assert not np.allclose(base[i], base[j])


def test_freenoise_beyond_window_drawn_from_base_vocabulary():
    # every frame past the first window must EQUAL one of the W base frames
    # (that is the whole point: a shared noise vocabulary across the clip)
    W = 5
    z = np.array(_freenoise_latents(W=W, total_t=23, h_lat=4, w_lat=4, C=3,
                                    dtype=mx.float32, seed=3))[0]
    base = [z[f] for f in range(W)]
    for f in range(W, len(z)):
        assert any(np.array_equal(z[f], b) for b in base), f"frame {f} not from base set"


def test_freenoise_deterministic_in_seed():
    a = np.array(_freenoise_latents(5, 17, 4, 4, 3, mx.float32, seed=42))
    b = np.array(_freenoise_latents(5, 17, 4, 4, 3, mx.float32, seed=42))
    assert np.array_equal(a, b)


# --- B7: single-shot anchor img2img seeding -----------------------------------

def _toy_pair(seed=0, shape=(1, 3, 2, 2, 4)):
    rng = np.random.default_rng(seed)
    a = mx.array(rng.standard_normal(shape).astype(np.float32))
    n = mx.array(rng.standard_normal(shape).astype(np.float32))
    return a, n


def test_img2img_tstart_one_runs_full_schedule():
    # t_start=1.0 -> first step (sched[0] <= 1) -> step0=0, run the whole schedule
    a, n = _toy_pair()
    sched = timestep_schedule(num_steps=30, shift=3.5)
    seeded, step0 = _img2img_init(a, n, sched, 30, t_start=1.0)
    assert step0 == 0
    t0 = float(sched[0])
    exp = (1 - t0) * np.array(a) + t0 * np.array(n)
    assert np.allclose(np.array(seeded), exp, atol=1e-5)


def test_img2img_lower_tstart_starts_later():
    # less trust in MD / more in the anchor -> start deeper into the schedule
    a, n = _toy_pair()
    sched = timestep_schedule(num_steps=30, shift=3.5)
    _, s_hi = _img2img_init(a, n, sched, 30, t_start=0.8)
    _, s_mid = _img2img_init(a, n, sched, 30, t_start=0.55)
    _, s_lo = _img2img_init(a, n, sched, 30, t_start=0.3)
    assert s_hi <= s_mid <= s_lo
    # step0 is the FIRST step at/below the request: its t is <=, the prior t is >
    assert float(sched[s_mid]) <= 0.55
    assert s_mid == 0 or float(sched[s_mid - 1]) > 0.55


def test_img2img_noise_equals_anchor_is_identity():
    # (1-t0)*a + t0*a == a for any t0; if the "noise" is the anchor the seed is it
    a, _ = _toy_pair(1)
    sched = timestep_schedule(num_steps=20, shift=3.5)
    seeded, _ = _img2img_init(a, a, sched, 20, t_start=0.5)
    assert np.allclose(np.array(seeded), np.array(a), atol=1e-5)


def test_img2img_blend_matches_chosen_step_time():
    a, n = _toy_pair(2)
    sched = timestep_schedule(num_steps=24, shift=3.5)
    seeded, step0 = _img2img_init(a, n, sched, 24, t_start=0.45)
    t0 = float(sched[step0])
    exp = (1 - t0) * np.array(a) + t0 * np.array(n)
    assert np.allclose(np.array(seeded), exp, atol=1e-5)


def test_img2img_very_low_tstart_clamps_to_valid_step():
    # below every nonzero schedule value -> fall back to the last step, still valid
    a, n = _toy_pair(3)
    sched = timestep_schedule(num_steps=10, shift=3.5)
    _, step0 = _img2img_init(a, n, sched, 10, t_start=1e-9)
    assert 0 <= step0 <= 9


# --- B7 dense-tail: fresh full-resolution schedule over [t_start, 0] -----------

def test_dense_schedule_spans_tstart_to_zero():
    a, n = _toy_pair()
    sched = timestep_schedule(num_steps=30, shift=3.5)
    seeded, loop_sched = _img2img_dense(a, n, sched, t_start=0.55)
    ls = np.array([float(loop_sched[i]) for i in range(31)])
    assert ls[0] == pytest.approx(0.55, abs=1e-5)   # top == t_start exactly
    assert ls[-1] == pytest.approx(0.0, abs=1e-6)    # ends at clean
    assert (np.diff(ls) < 0).all()                   # strictly decreasing


def test_dense_seed_blend_matches_tstart():
    a, n = _toy_pair(2)
    sched = timestep_schedule(num_steps=24, shift=3.5)
    seeded, _ = _img2img_dense(a, n, sched, t_start=0.4)
    exp = (1 - 0.4) * np.array(a) + 0.4 * np.array(n)
    assert np.allclose(np.array(seeded), exp, atol=1e-5)


def test_dense_runs_more_tail_steps_than_tail_mode():
    # the whole point: dense puts ALL num_steps below t_start; tail puts only a few
    a, n = _toy_pair()
    sched = timestep_schedule(num_steps=30, shift=3.5)
    _, loop_sched = _img2img_dense(a, n, sched, t_start=0.4)
    # float32 schedule vs float64 literal -> compare with a small tolerance
    dense_steps_below = sum(float(loop_sched[i]) <= 0.4 + 1e-4 for i in range(30))
    _, step0 = _img2img_init(a, n, sched, 30, t_start=0.4)
    tail_steps_below = 30 - step0
    assert dense_steps_below == 30                # dense puts ALL evals below t_start
    assert dense_steps_below > tail_steps_below   # tail puts only the few orig-sched tail steps


def test_dense_is_tstart_scaled_standard_schedule():
    a, n = _toy_pair()
    sched = timestep_schedule(num_steps=16, shift=3.5)
    _, loop_sched = _img2img_dense(a, n, sched, t_start=0.7)
    exp = 0.7 * np.array([float(sched[i]) for i in range(17)])
    got = np.array([float(loop_sched[i]) for i in range(17)])
    assert np.allclose(got, exp, atol=1e-6)


# --- D: temporal latent upsample (full-length anchor from a shorter clip) ------

def test_temporal_upsample_shape_and_dtype():
    z = mx.zeros((1, 7, 4, 4, 3), dtype=mx.bfloat16)
    up = _upsample_latent_temporal(z, 13)
    assert up.shape == (1, 13, 4, 4, 3)
    assert up.dtype == mx.bfloat16          # dtype preserved for the seed blend


def test_temporal_upsample_identity_when_same_length():
    rng = np.random.default_rng(0)
    z = mx.array(rng.standard_normal((1, 9, 2, 2, 4)).astype(np.float32))
    up = _upsample_latent_temporal(z, 9)
    assert np.array_equal(np.array(up), np.array(z))   # no-op, returns the input


def test_temporal_upsample_constant_field_preserved():
    # a clip that is constant in time must upsample to the same constant -> no
    # interpolation artifacts injected where there is no motion
    base = np.random.default_rng(1).standard_normal((1, 1, 3, 3, 5)).astype(np.float32)
    z = mx.array(np.repeat(base, 6, axis=1))            # (1,6,3,3,5) constant in t
    up = np.array(_upsample_latent_temporal(z, 14))
    assert up.shape == (1, 14, 3, 3, 5)
    for f in range(14):
        assert np.allclose(up[0, f], base[0, 0], atol=1e-5)


def test_temporal_upsample_endpoints_preserved():
    # first/last output frames land exactly on first/last source frames (the
    # clip's start and end are not blurred by the stretch)
    rng = np.random.default_rng(2)
    z = mx.array(rng.standard_normal((1, 5, 2, 2, 3)).astype(np.float32))
    up = np.array(_upsample_latent_temporal(z, 11))
    src = np.array(z)
    assert np.allclose(up[0, 0], src[0, 0], atol=1e-5)
    assert np.allclose(up[0, -1], src[0, -1], atol=1e-5)


def test_temporal_upsample_is_linear_blend_of_neighbors():
    # a ramp in time (frame f = f) must upsample to a linear ramp: every output
    # frame is an affine blend of its two source neighbors, never an overshoot
    st, T = 4, 9
    ramp = np.arange(st, dtype=np.float32).reshape(1, st, 1, 1, 1) * np.ones((1, st, 2, 2, 3), np.float32)
    up = np.array(_upsample_latent_temporal(mx.array(ramp), T))[0, :, 0, 0, 0]
    ts = np.clip((np.arange(T) + 0.5) * st / T - 0.5, 0, st - 1)   # expected positions
    assert np.allclose(up, ts, atol=1e-5)
    assert up.min() >= 0.0 and up.max() <= st - 1                  # no over/undershoot


def test_temporal_upsample_downsample_also_works():
    # the helper is symmetric: it can also REDUCE frame count (build a skeleton)
    rng = np.random.default_rng(4)
    z = mx.array(rng.standard_normal((1, 13, 2, 2, 3)).astype(np.float32))
    down = _upsample_latent_temporal(z, 7)
    assert down.shape == (1, 7, 2, 2, 3)


# --- D Phase-1 fix: slerp temporal upsample (norm-preserving, no washout) ------

def test_temporal_upsample_slerp_shape_and_dtype():
    z = mx.zeros((1, 7, 4, 4, 3), dtype=mx.bfloat16)
    up = _upsample_latent_temporal(z, 13, interp="slerp")
    assert up.shape == (1, 13, 4, 4, 3)
    assert up.dtype == mx.bfloat16          # dtype preserved for the seed blend


def test_temporal_upsample_slerp_identity_when_same_length():
    rng = np.random.default_rng(8)
    z = mx.array(rng.standard_normal((1, 9, 2, 2, 4)).astype(np.float32))
    up = _upsample_latent_temporal(z, 9, interp="slerp")
    assert np.array_equal(np.array(up), np.array(z))   # early return, untouched


def test_temporal_upsample_slerp_endpoints_preserved():
    # first/last output frames land exactly on first/last source frames
    rng = np.random.default_rng(7)
    z = mx.array(rng.standard_normal((1, 5, 2, 2, 3)).astype(np.float32))
    up = np.array(_upsample_latent_temporal(z, 11, interp="slerp"))
    src = np.array(z)
    assert np.allclose(up[0, 0], src[0, 0], atol=1e-4)
    assert np.allclose(up[0, -1], src[0, -1], atol=1e-4)


def test_temporal_upsample_slerp_constant_field_preserved():
    # constant in time -> no interpolation artifacts (omega=0 falls back to lerp)
    base = np.random.default_rng(9).standard_normal((1, 1, 3, 3, 5)).astype(np.float32)
    z = mx.array(np.repeat(base, 6, axis=1))            # (1,6,3,3,5) constant in t
    up = np.array(_upsample_latent_temporal(z, 14, interp="slerp"))
    for f in range(14):
        assert np.allclose(up[0, f], base[0, 0], atol=1e-4)


def test_temporal_upsample_slerp_preserves_norm_no_sag():
    # THE point of slerp: two orthogonal equal-norm frames; the temporal midpoint
    # keeps its norm on the arc, where bilinear sags via vector cancellation.
    f0 = np.array([1.0, 0.0], np.float32)
    f1 = np.array([0.0, 1.0], np.float32)
    z = mx.array(np.stack([f0, f1]).reshape(1, 2, 1, 1, 2))   # (1,2,1,1,2)
    sl = np.array(_upsample_latent_temporal(z, 3, interp="slerp"))[0]    # ts=[0,.5,1]
    bl = np.array(_upsample_latent_temporal(z, 3, interp="bilinear"))[0]
    mid_sl = float(np.linalg.norm(sl[1].ravel()))
    mid_bl = float(np.linalg.norm(bl[1].ravel()))
    assert abs(mid_sl - 1.0) < 1e-3            # slerp: norm preserved on the arc
    assert abs(mid_bl - np.sqrt(0.5)) < 1e-3   # bilinear: sags to ~0.707
    assert mid_sl > mid_bl + 0.2               # slerp strictly less washed-out


def test_temporal_upsample_bilinear_unchanged_by_refactor():
    # the default path must stay byte-identical to the pre-slerp bilinear (parity)
    rng = np.random.default_rng(11)
    z = mx.array(rng.standard_normal((1, 13, 4, 4, 6)).astype(np.float32))
    a = np.array(z.astype(mx.float32))[0]
    ts = np.clip((np.arange(25) + 0.5) * 13 / 25 - 0.5, 0, 12)
    f0 = np.floor(ts).astype(int); f1 = np.minimum(f0 + 1, 12)
    w = (ts - f0)[:, None, None, None]
    expect = a[f0] * (1 - w) + a[f1] * w
    got = np.array(_upsample_latent_temporal(z, 25))[0]      # default = bilinear
    assert np.allclose(got, expect, atol=1e-6)


def test_temporal_upsample_bad_interp_raises():
    z = mx.zeros((1, 3, 2, 2, 3))
    with pytest.raises(ValueError):
        _upsample_latent_temporal(z, 5, interp="cubic")
