"""Bit-identity gate for lossless temporal causal-cache streaming decode.

decode_streaming(dec, z) MUST equal the shipped whole-seq dec(z), for any T_lat and
chunk size. Uses a small RANDOM-init Wan22VAEDecoder — the cache logic is value-
independent, so random weights with several (T_lat, chunk) combos expose any frame-
indexing / cache-boundary bug. No model download, small latents → fast & low-memory.

The acceptance bar (LOSSLESS_PLAN.md §2 Phase 1): target bit-identical; the hard
ceiling is ≪ 1.46/255 ≈ 0.0057 in [0,1] (≈0.0115 in [-1,1]) — today's own tiling
error — so streaming is strictly better than the shipped lossy blend either way.
"""
from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_map

from lance_mlx.model.vae_stream import decode_streaming, suggest_spatial_tiles

# whole-seq reference is the shipped Wan22VAEDecoder.__call__
from mlx_video.models.wan_2.vae22 import Wan22VAEDecoder

# bit-identical target; fp32 op-reordering tolerance well under the lossy-blend floor
ATOL_EXACT = 1e-4
LOSSY_BLEND_FLOOR = 0.0115  # 1.46/255 in [-1,1] units — must be far under this


def _random_decoder(seed: int = 0, dec_dim: int = 256) -> Wan22VAEDecoder:
    dec = Wan22VAEDecoder(z_dim=48, dim=160, dec_dim=dec_dim)
    mx.random.seed(seed)
    # randomize every parameter so activations are non-degenerate (zeros init would
    # make any cache bug invisible). Small scale keeps RMS_norm well-conditioned.
    params = tree_map(lambda a: (mx.random.normal(a.shape) * 0.05).astype(a.dtype),
                      dec.parameters())
    dec.update(params)
    mx.eval(dec.parameters())
    return dec


def _latent(T_lat: int, hlat: int = 2, wlat: int | None = None,
            seed: int = 1) -> mx.array:
    mx.random.seed(seed)
    w = hlat if wlat is None else wlat
    return (mx.random.normal((1, T_lat, hlat, w, 48)) * 0.5).astype(mx.float32)


def _max_abs_diff(a: mx.array, b: mx.array) -> float:
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    return float(mx.max(mx.abs(a - b)))


@pytest.mark.parametrize("T_lat,chunk", [
    (2, 1), (3, 1), (4, 1), (5, 1),   # chunk=1 = minimum extent (the memory target)
    (4, 2), (5, 2), (6, 3),            # multi-frame chunks must also match
    (4, 4),                            # chunk == T_lat → single chunk (≈ whole-seq)
])
def test_streaming_matches_whole_seq(T_lat, chunk):
    dec = _random_decoder()
    z = _latent(T_lat)
    ref = dec(z)                                  # shipped whole-seq
    got = decode_streaming(dec, z, chunk_lat=chunk)
    assert got.shape == ref.shape, (T_lat, chunk, got.shape, ref.shape)
    d = _max_abs_diff(got, ref)
    assert d < ATOL_EXACT, f"T_lat={T_lat} chunk={chunk}: max|Δ|={d:.2e} (≥{ATOL_EXACT})"
    assert d < LOSSY_BLEND_FLOOR


# --------------------------------------------------------------------------
# Phase 2: spatial halo-tile + CROP of the high-res suffix.
#
# The halo/crop/cache logic is determined by conv KERNELS (3×3, fixed) and is
# independent of channel count, so a SMALL-dim decoder reproduces the exact same
# spatial receptive field at a fraction of the memory — letting us test LARGE
# suffix grids with genuinely interior tiles (interior on all 4 sides) cheaply.
# suffix grid G = 4·hlat; final RGB = 16·hlat. hlat=12 → G=64 → 256² output.
# --------------------------------------------------------------------------

# T_lat≥2 here; the T_lat=1 (single-frame / t2i) case — including composed with spatial
# tiling — is covered by test_T_lat_1{,_spatial_tiled}_matches_whole_seq (task #48).
@pytest.mark.parametrize("hlat,T_lat,chunk,n", [
    (4, 2, 1, 2),   # G=16: small grid, halo clamps (effectively whole) — must still match
    (8, 2, 1, 2),   # G=32: each tile interior on one side, global on the other
    (8, 3, 1, 3),   # G=32 + temporal streaming (3 chunks) + 3×3 tiling
    (12, 2, 1, 4),  # G=64, n=4: middle tiles interior on ALL FOUR sides (full halo+crop)
    (12, 4, 1, 4),  # G=64 + temporal streaming (4 chunks) + interior tiles (strongest)
    (12, 4, 2, 4),  # G=48, multi-frame chunks + interior tiles
    (12, 2, 2, 4),  # chunk_lat≥T_lat → single multi-frame chunk: keep_cache=False path
    (8, 4, 2, 2),   # spatial tiling with chunk_lat=2 (2 chunks)
    (16, 2, 1, 4),  # G=64 (256² out): DEFAULT halo=12 hits all-4-sides-interior tiles
    (12, 2, 1, 5),  # n=5: non-power-of-2 tile counts
    (16, 2, 1, 6),  # n=6: uneven _tile_bounds partition (64 = 11×4 + 10×2)
])
def test_spatial_tiled_matches_whole_seq(hlat, T_lat, chunk, n):
    dec = _random_decoder(dec_dim=16)  # small channels → cheap; same RF as full model
    z = _latent(T_lat, hlat=hlat)
    ref = dec(z)                                          # shipped whole-seq
    got = decode_streaming(dec, z, chunk_lat=chunk, spatial_tiles=n)
    assert got.shape == ref.shape, (hlat, T_lat, chunk, n, got.shape, ref.shape)
    d = _max_abs_diff(got, ref)
    assert d < ATOL_EXACT, \
        f"hlat={hlat} T_lat={T_lat} chunk={chunk} tiles={n}: max|Δ|={d:.2e} (≥{ATOL_EXACT})"
    assert d < LOSSY_BLEND_FLOOR


@pytest.mark.parametrize("hlat,wlat,T_lat,chunk,n", [
    (12, 8, 2, 1, 4),   # G_h=48 ≠ G_w=32 — independent per-axis tile bounds + crop
    (8, 12, 4, 1, 3),   # transposed aspect + temporal streaming
    (9, 16, 2, 2, 4),   # odd height, even width, multi-frame chunks
])
def test_spatial_tiled_nonsquare_matches_whole_seq(hlat, wlat, T_lat, chunk, n):
    # Every other test uses square latents; non-square exercises the H≠W tile/crop
    # axes independently (a regression here would be silent in the square suite).
    dec = _random_decoder(dec_dim=16)
    z = _latent(T_lat, hlat=hlat, wlat=wlat)
    ref = dec(z)
    got = decode_streaming(dec, z, chunk_lat=chunk, spatial_tiles=n)
    assert got.shape == ref.shape, (hlat, wlat, T_lat, chunk, n, got.shape, ref.shape)
    d = _max_abs_diff(got, ref)
    assert d < ATOL_EXACT, \
        f"{hlat}x{wlat} T={T_lat} chunk={chunk} n={n}: max|Δ|={d:.2e} (≥{ATOL_EXACT})"


def test_spatial_tiled_clip_output_false_matches_whole():
    # clip_output=False is unverified against any reference in the suite. The shipped
    # dec(z) always clips; assert clip(streaming-tiled, unclipped) == dec(z), i.e. the
    # pre-clip tiled values are exactly the whole-seq pre-clip values.
    dec = _random_decoder(dec_dim=16)
    z = _latent(2, hlat=12)
    ref = dec(z)                                          # clipped
    raw = decode_streaming(dec, z, chunk_lat=1, spatial_tiles=4, clip_output=False)
    d = _max_abs_diff(mx.clip(raw, -1.0, 1.0), ref)
    assert d < ATOL_EXACT, f"clip_output=False tiled: max|Δ|={d:.2e}"


def test_warn_ineffective_fires_but_output_still_exact(caplog):
    # When the halo makes every tile span the whole grid, tiling is a no-op — it must
    # LOG (no silent cap) AND still be bit-identical. G=8 (hlat=2), n=4, halo=12.
    import logging
    dec = _random_decoder(dec_dim=16)
    z = _latent(2, hlat=2)
    with caplog.at_level(logging.WARNING, logger="lance_mlx.model.vae_stream"):
        got = decode_streaming(dec, z, chunk_lat=1, spatial_tiles=4)
    assert any("no memory" in r.message for r in caplog.records), \
        "ineffective-tiling warning did not fire"
    assert _max_abs_diff(got, dec(z)) < ATOL_EXACT


def test_T_lat_1_matches_whole_seq():
    # task #48: a lone latent frame (T_lat=1, the t2i / single-image case) is now
    # bit-identical. whole-seq dec(z) DOUBLES the single frame at the first upsample3d
    # (its `first_chunk and T>1` test is False → else branch); decode_streaming mirrors
    # that (single=True) instead of the 'Rep' defer, so frame count AND values match.
    # This unblocks the t2i / image_edit / t2i_multidiff default-flip.
    dec = _random_decoder(dec_dim=16)
    z = _latent(1, hlat=8)
    ref = dec(z)
    got = decode_streaming(dec, z, chunk_lat=1)
    assert got.shape == ref.shape, f"T_lat=1 shape {got.shape} != whole-seq {ref.shape}"
    d = _max_abs_diff(got, ref)
    assert d < ATOL_EXACT, f"T_lat=1 max|Δ|={d:.2e} (≥{ATOL_EXACT})"
    assert d < LOSSY_BLEND_FLOOR


@pytest.mark.parametrize("h_lat,w_lat,expect", [
    (16, 16, 4),    # 256²: the sweet-spot n=4
    (32, 32, 8),    # 512²
    (48, 48, 12),   # 768²
    (64, 64, 16),   # 1024²
    (8, 8, 2),      # 128²
    (4, 4, 1),      # 64²: tiling a no-op
    (16, 48, 12),   # non-square: scale on the LARGER axis (bound the dominant dim)
])
def test_suggest_spatial_tiles_policy(h_lat, w_lat, expect):
    # the flip's tile-count policy: ~16 G-px per tile, scaled on the larger latent axis.
    assert suggest_spatial_tiles(h_lat, w_lat) == expect


def test_suggested_tiles_decode_is_bit_identical():
    # end-to-end: the policy's n must still decode bit-identically (T_lat=1 image path).
    dec = _random_decoder(dec_dim=16)
    z = _latent(1, hlat=16)                       # 256²-equivalent latent grid
    n = suggest_spatial_tiles(16, 16)
    got = decode_streaming(dec, z, chunk_lat=1, spatial_tiles=n)
    d = _max_abs_diff(got, dec(z))
    assert d < ATOL_EXACT, f"suggested n={n}: max|Δ|={d:.2e}"


@pytest.mark.parametrize("h_lat,w_lat,max_n,expect", [
    (16, 16, 4, 4),     # 256² video: policy n=4, cap 4 — no change (the sweet spot)
    (32, 32, 4, 4),     # 512² video: policy n=8 → capped to 4
    (48, 48, 4, 4),     # 768² video: policy n=12 → capped to 4 (avoids the U-curve blow-up)
    (4, 4, 4, 1),       # tiny: policy n=1, cap doesn't raise it (still a no-op)
    (16, 48, 4, 4),     # non-square video: larger-axis n=12 → capped to 4
    (16, 16, None, 4),  # max_n=None (image path) unchanged
    (48, 48, None, 12), # image path uncapped — confirms cap is opt-in, not default
])
def test_suggest_spatial_tiles_max_n_cap(h_lat, w_lat, max_n, expect):
    # The VIDEO sites pass max_n=4 (multi-chunk → #49 cache-drop N/A → U-curve live, so
    # keep n LOW). Images pass None (single-chunk → higher n free). Pin both regimes.
    assert suggest_spatial_tiles(h_lat, w_lat, max_n=max_n) == expect


@pytest.mark.parametrize("T_lat,chunk", [(4, 1), (6, 1)])
def test_video_capped_tiles_decode_is_bit_identical(T_lat, chunk):
    # The exact video-flip call shape: multi-chunk (chunk_lat=1, T_lat≥2 → keep_cache=True
    # path) with the max_n=4-capped tile count. Must stay bit-identical to whole-seq.
    dec = _random_decoder(dec_dim=16)
    z = _latent(T_lat, hlat=16)                   # 256²-equivalent video latent grid
    n = suggest_spatial_tiles(16, 16, max_n=4)
    assert n == 4
    got = decode_streaming(dec, z, chunk_lat=chunk, spatial_tiles=n)
    d = _max_abs_diff(got, dec(z))
    assert d < ATOL_EXACT, f"video T_lat={T_lat} n={n}: max|Δ|={d:.2e} (≥{ATOL_EXACT})"
    assert d < LOSSY_BLEND_FLOOR


@pytest.mark.parametrize("hlat,n", [(8, 2), (12, 4), (16, 4)])
def test_T_lat_1_spatial_tiled_matches_whole_seq(hlat, n):
    # The t2i flip uses BOTH the T_lat=1 fix AND spatial tiling. Verify they compose:
    # single-frame doubling happens in the (whole) prefix, then the suffix is tiled —
    # output must still equal dec(z) exactly. hlat=16→G=64 (256² out).
    dec = _random_decoder(dec_dim=16)
    z = _latent(1, hlat=hlat)
    ref = dec(z)
    got = decode_streaming(dec, z, chunk_lat=1, spatial_tiles=n)
    assert got.shape == ref.shape, (hlat, n, got.shape, ref.shape)
    d = _max_abs_diff(got, ref)
    assert d < ATOL_EXACT, f"T_lat=1 hlat={hlat} n={n}: max|Δ|={d:.2e} (≥{ATOL_EXACT})"


@pytest.mark.parametrize("halo", [10, 11, 16])
def test_spatial_tiled_halo_at_or_above_rf_is_exact(halo):
    # The suffix spatial RF is 10 (G-px). Any halo ≥ 10 must be bit-identical;
    # this pins the derived RF and lets the footprint pass tighten halo with proof.
    dec = _random_decoder(dec_dim=16)
    z = _latent(2, hlat=12)                               # G=64, interior tiles at n=4
    ref = dec(z)
    got = decode_streaming(dec, z, chunk_lat=1, spatial_tiles=4, halo=halo)
    d = _max_abs_diff(got, ref)
    assert d < ATOL_EXACT, f"halo={halo}: max|Δ|={d:.2e} (≥{ATOL_EXACT})"


def test_too_small_halo_is_detectably_wrong():
    # Guard against false confidence: a halo BELOW the suffix RF (10) must produce a
    # visible error at interior tile boundaries — proving the gate can detect breakage
    # (i.e. the tiling isn't a silent no-op). G=64, n=2 → one interior boundary at 32.
    dec = _random_decoder(dec_dim=16)
    z = _latent(2, hlat=12)
    ref = dec(z)
    bad = decode_streaming(dec, z, chunk_lat=1, spatial_tiles=2, halo=2)
    d = _max_abs_diff(bad, ref)
    assert d > ATOL_EXACT, \
        f"halo=2 (< RF 10) should be detectably wrong, but max|Δ|={d:.2e}"


def test_output_frame_count_matches_causal_formula():
    # whole-seq mapping is 1 + (T_lat-1)*4 (temporal_scale=4); streaming must agree
    dec = _random_decoder()
    for T_lat in (2, 3, 4, 6):
        z = _latent(T_lat)
        got = decode_streaming(dec, z, chunk_lat=1)
        assert got.shape[1] == dec(z).shape[1]


if __name__ == "__main__":
    dec = _random_decoder()
    print(f"{'T_lat':>6}{'chunk':>6}{'frames':>8}{'max|Δ|':>12}   verdict")
    worst = 0.0
    for T_lat, chunk in [(2, 1), (3, 1), (4, 1), (5, 1), (8, 1),
                         (4, 2), (5, 2), (6, 3), (4, 4)]:
        z = _latent(T_lat)
        ref = dec(z)
        got = decode_streaming(dec, z, chunk_lat=chunk)
        d = _max_abs_diff(got, ref)
        worst = max(worst, d)
        ok = "BIT-IDENTICAL" if d < ATOL_EXACT else (
            "≪blend-OK" if d < LOSSY_BLEND_FLOOR else "FAIL")
        print(f"{T_lat:>6}{chunk:>6}{ref.shape[1]:>8}{d:>12.2e}   {ok}")
    print(f"\nworst max|Δ| across all = {worst:.2e}  "
          f"(exact<{ATOL_EXACT}, lossy-blend floor={LOSSY_BLEND_FLOOR})")
