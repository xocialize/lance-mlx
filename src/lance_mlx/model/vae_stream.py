"""Lossless temporal causal-cache streaming VAE decode (Lance / Wan2.2 VAE).

The shipped decode path is `Wan22VAEDecoder.__call__(z)` = whole-sequence: it
processes ALL latent frames at once, so a 256²×61f decode holds the full temporal
extent in memory (~14 GB true footprint; see results/decode_lossless/). The tiled
path (`decode_tiled`) caps memory but decodes each temporal tile INDEPENDENTLY with
a zero-padded causal boundary, then cross-fades overlaps — that is LOSSY (1.46-4.82
px/255 off the true decode; see DECODE_MEMORY_FINDINGS.md).

The Wan2.2 decoder is CAUSAL in time: each output frame depends only on the current
and PAST latent frames. So we can decode one latent frame (or a small chunk) at a
time while carrying a tiny per-conv boundary cache (the last CACHE_T=2 input frames
at each CausalConv3d), reproducing the whole-sequence decode BIT-IDENTICALLY. This
is the mirror of the encoder's already-shipped chunked `encode` (feat_cache/feat_idx)
— the decode path simply never had it wired.

Win: removes temporal growth. 256²×Nf decode → ~one-frame extent (~9 GB), FLAT in N,
and bit-identical to the true decode (strictly better than today's lossy blend).
This is Phase 1; spatial per-stage tiling (Phase 2) stacks on top toward ~4 GB.

Correctness gate: `decode_streaming(dec, z)` must equal `dec(z)` (whole-seq). See
tests/test_decode_stream.py — verified bit-identical for several T_lat / chunk sizes.

Design — reuse the weight-bearing submodules, reimplement only the temporal cache:
  • conv1 / ResidualBlock convs / head  : CausalConv3d already takes cache_x;
    ResidualBlock & Head22 already thread feat_cache → call them directly.
  • AttentionBlock                       : per-frame (no temporal mix) → call as-is.
  • Resample upsample2d (up2)            : no time_conv, per-frame → call as-is.
  • DupUp3D shortcut                     : per-frame; first_chunk trim only on the
                                           GLOBAL first chunk → pass first=first.
  • Resample upsample3d time_conv (up0/up1): the ONE cross-chunk temporal op — its
    causal cache is reimplemented here with the 'Rep' first-chunk sentinel that
    reproduces the whole-seq `if first_chunk and T>1` (frame 0 not doubled). The
    T_lat==1 sequence (single frame, no successor chunk) takes whole-seq's else
    branch instead — the lone frame is DOUBLED (task #48), so t2i decodes exactly.
"""

from __future__ import annotations

import gc
import logging

import mlx.core as mx

from mlx_video.models.wan_2.vae22 import (
    CACHE_T,
    AttentionBlock,
    CausalConv3d,
    DupUp3D,
    ResidualBlock,
    Resample,
    Up_ResidualBlock,
    Wan22VAEDecoder,
    _unpatchify,
)

logger = logging.getLogger(__name__)

_REP = "Rep"  # sentinel: an upsample3d slot that has only seen the first chunk


def _conv_cached(conv: CausalConv3d, x, fc, fi):
    """CausalConv3d with cross-chunk temporal cache. Mirrors
    ResidualBlockLayers._conv_with_cache exactly (CACHE_T=2 boundary frames)."""
    idx = fi[0]
    cache_x = x[:, -CACHE_T:]
    if cache_x.shape[1] < 2 and fc[idx] is not None:
        cache_x = mx.concatenate([fc[idx][:, -1:], cache_x], axis=1)
    out = conv(x, cache_x=fc[idx])
    fc[idx] = cache_x
    fi[0] += 1
    return out


def _resample_upsample3d_cached(rs: Resample, x, fc, fi, first: bool,
                                single: bool = False):
    """Streaming upsample3d: temporal time_conv with 'Rep'-sentinel cache, then the
    per-frame spatial upsample (reused from the module). Reproduces the whole-seq
    `Resample.__call__(mode='upsample3d', first_chunk=...)` chunk-by-chunk.

    `single` = the WHOLE latent sequence is a single frame (T_lat==1). Then whole-seq's
    `if first_chunk and T>1` test is False → it DOUBLES the lone frame (the else branch),
    so we must double too. The 'Rep' defer below is only correct when MORE chunks follow
    (multi-chunk stream); with no successor it would emit 1 frame where dec(z) emits the
    doubled frame. See task #48 / PHASE2_FINDINGS.md."""
    B, T, H, W, C = x.shape
    idx = fi[0]

    if first:
        # whole-seq first_chunk: frame 0 is NOT doubled; frames 1.. are time_conv'd
        # with a zero-padded causal boundary (rest = x[:,1:], frame 0 excluded).
        if T > 1:
            first_frame = x[:, 0:1]
            rest = x[:, 1:]
            tc = rs.time_conv(rest)  # cache_x=None → causal zero-pad at rest start
            tc = tc.reshape(B, T - 1, H, W, 2, C)
            inter = mx.stack([tc[:, :, :, :, 0, :], tc[:, :, :, :, 1, :]], axis=2)
            inter = inter.reshape(B, (T - 1) * 2, H, W, C)
            x = mx.concatenate([first_frame, inter], axis=1)
            # cache for next chunk = last 2 frames of the conv INPUT (rest)
            fc[idx] = rest[:, -CACHE_T:]
        elif single:
            # entire sequence is ONE latent frame: mirror whole-seq's else branch —
            # DOUBLE via time_conv (cache_x=None → causal zero-pad), exactly like
            # dec(z). No chunk follows, so the cache is moot; keep it the input
            # boundary for consistency.
            boundary = x[:, -CACHE_T:]
            tc = rs.time_conv(x)
            tc = tc.reshape(B, T, H, W, 2, C)
            x = mx.stack([tc[:, :, :, :, 0, :], tc[:, :, :, :, 1, :]], axis=2)
            x = x.reshape(B, T * 2, H, W, C)
            fc[idx] = boundary
        else:
            # single-frame first chunk but MORE chunks follow: 'Rep' — no doubling,
            # next chunk zero-pads.
            fc[idx] = _REP
        fi[0] += 1
    else:
        prev = fc[idx]
        cache_x = x[:, -CACHE_T:]
        if cache_x.shape[1] < 2 and prev is not None and prev is not _REP:
            cache_x = mx.concatenate([prev[:, -1:], cache_x], axis=1)
        if cache_x.shape[1] < 2 and prev is _REP:
            cache_x = mx.concatenate([mx.zeros_like(cache_x), cache_x], axis=1)
        cache_arg = None if prev is _REP else prev
        tc = rs.time_conv(x, cache_x=cache_arg)  # (B, T, H, W, 2C)
        fc[idx] = cache_x
        fi[0] += 1
        tc = tc.reshape(B, T, H, W, 2, C)
        x = mx.stack([tc[:, :, :, :, 0, :], tc[:, :, :, :, 1, :]], axis=2)
        x = x.reshape(B, T * 2, H, W, C)

    # --- spatial upsample (per-frame, identical to whole-seq) ---
    T2 = x.shape[1]
    xf = x.reshape(B * T2, H, W, C)
    xf = rs._upsample2x(xf)
    xf = rs._conv2d(xf)
    return xf.reshape(B, T2, xf.shape[1], xf.shape[2], C)


def _up_resblock_cached(blk: Up_ResidualBlock, x, fc, fi, first: bool,
                        single: bool = False):
    """Up_ResidualBlock with cross-chunk cache threaded through its ResidualBlocks
    and Resample, plus the per-frame DupUp3D shortcut. `single` (T_lat==1) only
    reaches the upsample3d temporal op; first=False (suffix) ignores it."""
    x_in = x
    x_main = x
    for module in blk.upsamples:
        if isinstance(module, Resample):
            if module.mode == "upsample3d":
                x_main = _resample_upsample3d_cached(module, x_main, fc, fi, first,
                                                     single)
            else:  # upsample2d: no time_conv, per-frame → reuse module as-is
                x_main = module(x_main, first)
        else:  # ResidualBlock — already supports feat_cache
            x_main = module(x_main, fc, fi)
        mx.eval(x_main)

    if blk.avg_shortcut is not None:
        x_sc = blk.avg_shortcut(x_in, first)  # DupUp3D: per-frame, trim on first only
        mx.eval(x_sc)
        return x_main + x_sc
    return x_main


# ---------------------------------------------------------------------------
# Phase 2: spatial halo-tile + CROP of the high-res upsampler suffix.
#
# Phase 1 caps the temporal extent (one-chunk floor ≈9 GB @256²). The remaining
# peak lives in the high-res suffix (up2/up3/head @64²→128²). We split the
# decoder at the first NON-temporal upsample block:
#   • PREFIX  = conv1 → middle → up0 → up1   (all upsample3d) — runs WHOLE. It is
#     cheap (≤64²) and holds the only cross-chunk temporal-upsample 'Rep' logic.
#   • SUFFIX  = up2 → up3 → head             (upsample2d / no resample) — TILED.
# Each tile pulls a REAL-neighbour halo (≥ the suffix's spatial receptive field)
# from the full prefix output, runs the suffix unchanged, then CROPS the halo
# away — no blend, no fade. A kept pixel ≥halo from the slice edge sees only the
# input whole-seq would have given it ⇒ bit-identical. (Today's decode_tiled
# zero-pads the missing neighbour + cross-fades = lossy; the crop is the whole
# difference.) Tiles compose with temporal streaming via a PER-TILE temporal
# cache: the suffix convs are temporally-causal AND spatially-local, the two axes
# are independent, and the halo bounds the spatial reach inside the cached
# boundary frames too (same 3×3 kernel) — so no extra halo is needed for time.
# ---------------------------------------------------------------------------

# Spatial receptive-field radius of the suffix in suffix-INPUT (G) pixels:
#   up2 resblocks 6 @G  +  (resample conv 1 + up3 6 + head 1 = 8 @2G → 4 @G)  = 10.
# The 8 @2G ops map to 4 @G by support-interval contraction across the nearest ×2
# upsample (a ±8 reach at 2G maps back through the nearest-2× to ±4 at G), which here
# equals ⌈8/2⌉. Verified three ways (forward index-support, impulse perturbation, and
# a halo-sweep whose bit-identity cliff is exactly at 10). The outer RF band carries
# only ~1e-8 weight, so the bit-identity GATE (max|Δ|=0.0), not an effective-RF
# threshold, is the real guarantee. Halo must be ≥ 10; default carries +2 margin.
_SUFFIX_HALO = 12


def _suffix_start(dec_decoder) -> int:
    """Index of the first upsample block with NO upsample3d (temporal) resample —
    the start of the spatially-tileable suffix. Earlier blocks carry cross-chunk
    temporal-upsample state ('Rep') and must run whole."""
    for i, blk in enumerate(dec_decoder.upsamples):
        has_temporal = any(
            isinstance(m, Resample) and m.mode == "upsample3d" for m in blk.upsamples
        )
        if not has_temporal:
            return i
    return len(dec_decoder.upsamples)


def _decoder3d_prefix(dec_decoder, x, fc, fi, first, suffix_start, single=False):
    """conv1 → middle → [upsample3d blocks], WHOLE spatial. Returns the suffix
    input, threading the shared temporal cache exactly as Phase 1. `single`
    (T_lat==1) makes the first-chunk upsample3d double the lone frame like dec(z)."""
    x = _conv_cached(dec_decoder.conv1, x, fc, fi)
    for layer in dec_decoder.middle:
        if isinstance(layer, AttentionBlock):
            x = layer(x)  # per-frame, no cache, no idx
        else:  # ResidualBlock
            x = layer(x, fc, fi)
    mx.eval(x)
    for layer in dec_decoder.upsamples[:suffix_start]:
        x = _up_resblock_cached(layer, x, fc, fi, first, single)
        mx.eval(x)
    return x


def _decoder3d_suffix(dec_decoder, x, fc, fi, suffix_start):
    """[remaining upsample blocks] → head, on a (possibly spatially-sliced) input.
    No upsample3d here, so `first_chunk` is a no-op (up2's avg_shortcut is
    factor_t=1, upsample2d ignores it); temporal causality is carried entirely by
    the CausalConv3d caches in `fc`."""
    for layer in dec_decoder.upsamples[suffix_start:]:
        x = _up_resblock_cached(layer, x, fc, fi, False)
        mx.eval(x)
    x = dec_decoder.head(x, fc, fi)  # Head22 already threads feat_cache
    return x


def _decoder3d_chunk(dec_decoder, x, fc, fi, first: bool, single: bool = False):
    """One chunk through Decoder3d, WHOLE spatial (Phase-1 path). Prefix+suffix
    share one temporal cache with a continuous feat_idx — byte-for-byte the
    original single-pass behaviour."""
    suffix_start = _suffix_start(dec_decoder)
    x = _decoder3d_prefix(dec_decoder, x, fc, fi, first, suffix_start, single)
    x = _decoder3d_suffix(dec_decoder, x, fc, fi, suffix_start)
    return x


def _tile_bounds(size: int, n: int):
    """Partition [0, size) into n contiguous, near-equal tiles (drops empties)."""
    base, rem = divmod(size, n)
    bounds = []
    s = 0
    for i in range(n):
        e = s + base + (1 if i < rem else 0)
        if e > s:
            bounds.append((s, e))
        s = e
    return bounds


def _warn_if_ineffective(g_h, g_w, n, halo):
    """No silent cap: log when the halo makes every tile span the whole axis so
    spatial tiling saves no memory at this resolution (still correct, just a no-op)."""
    rb = _tile_bounds(g_h, n)
    cb = _tile_bounds(g_w, n)
    max_h = max(min(g_h, r1 + halo) - max(0, r0 - halo) for r0, r1 in rb)
    max_w = max(min(g_w, c1 + halo) - max(0, c0 - halo) for c0, c1 in cb)
    if max_h >= g_h and max_w >= g_w:
        logger.warning(
            "decode_streaming spatial_tiles=%d at suffix grid %dx%d: halo=%d makes the "
            "largest tile span the whole grid (%dx%d) → tiling saves no memory at this "
            "resolution (output still correct, but a no-op).",
            n, g_h, g_w, halo, max_h, max_w,
        )


def _suffix_spatial_tiled(dec_decoder, x, fc_tiles, suffix_start, n_tiles, halo,
                          keep_cache: bool = True):
    """Run the suffix on x=(B,T,G_h,G_w,C) spatially tiled n×n with a real-neighbour
    halo + crop → bit-identical to the whole suffix, but with peak ≈ one tile's
    working set. fc_tiles[k] is tile k's persistent temporal cache (same spatial
    bounds every chunk → consistent). The suffix has one ×2 spatial upsample (up2),
    so its output is (B,T,2*G_h,2*G_w,12); tiles are assembled by concatenating the
    exact (non-overlapping) crops — no blend.

    keep_cache=False (single-chunk decode, e.g. t2i / any chunk_lat≥T_lat): there is
    no successor chunk, so the per-tile temporal caches are written-but-never-read.
    Each tile's cache redundantly stores its HALOED region, so retaining all n² of
    them makes the footprint grow ~(G + 2·halo·n)² — the measured ~0.09 GB/tile
    U-curve (task #49). Dropping each cache right after its tile keeps only ONE tile's
    cache live ⇒ footprint flat in n. Bit-identity is unaffected (the dropped caches
    feed nothing)."""
    B, T, G_h, G_w, C = x.shape
    rb = _tile_bounds(G_h, n_tiles)
    cb = _tile_bounds(G_w, n_tiles)

    row_strips = []
    k = 0
    for (r0, r1) in rb:
        col_pieces = []
        for (c0, c1) in cb:
            # pull a real-neighbour halo from the FULL prefix output; clamp at the
            # global edge (there the conv's own zero-pad matches whole-seq exactly).
            sr0, sr1 = max(0, r0 - halo), min(G_h, r1 + halo)
            sc0, sc1 = max(0, c0 - halo), min(G_w, c1 + halo)
            sl = x[:, :, sr0:sr1, sc0:sc1, :]
            fi = [0]
            o = _decoder3d_suffix(dec_decoder, sl, fc_tiles[k], fi, suffix_start)
            # crop the halo: input slice began at sr0 ⇒ output (×2) began at 2*sr0.
            cr0, cc0 = 2 * (r0 - sr0), 2 * (c0 - sc0)
            o = o[:, :, cr0:cr0 + 2 * (r1 - r0), cc0:cc0 + 2 * (c1 - c0), :]
            if keep_cache:
                mx.eval(o, fc_tiles[k])  # carry this tile's cache to the next chunk
            else:
                fc_tiles[k] = None       # single chunk: cache never read → free it
                mx.eval(o)
            col_pieces.append(o)
            k += 1
            gc.collect()
            mx.clear_cache()
        strip = col_pieces[0] if len(col_pieces) == 1 else mx.concatenate(col_pieces, axis=3)
        mx.eval(strip)
        row_strips.append(strip)
    return row_strips[0] if len(row_strips) == 1 else mx.concatenate(row_strips, axis=2)


def suggest_spatial_tiles(h_lat: int, w_lat: int, target_g_px: int = 16,
                          max_n: int | None = None) -> int:
    """Pick `spatial_tiles` for decode_streaming from the latent H/W. The tileable
    suffix runs on grid G = 4·lat (up1 output); we aim for ~`target_g_px` G-pixels per
    tile (16 = the 256²/n=4 sweet spot). Returns 1 for small images, where tiling would
    be a no-op. Scales n with the LARGER axis so the dominant dimension is bounded; the
    #49 single-chunk cache fix makes the resulting higher n free for images (T_lat=1).
    Examples: 256² (lat 16)→4 · 512² (32)→8 · 768² (48)→12 · 1024² (64)→16 · 64² (4)→1.

    `max_n` caps the result — REQUIRED for VIDEO (T_lat≥2): video decodes multi-chunk
    (chunk_lat<T_lat → keep_cache=True), so the #49 single-chunk cache drop does NOT
    apply and the per-tile-cache U-curve is live (~0.09 GB/tile, grows ~n²). Video
    sites pass max_n=4 and lean on chunk_lat=1 (flat-in-length temporal streaming) as
    the primary memory lever; spatial tiling just shaves the suffix. Images leave it
    None (single-chunk → cache dropped → higher n is free)."""
    g = 4 * max(h_lat, w_lat)
    n = max(1, round(g / target_g_px))
    if max_n is not None:
        n = min(n, max_n)
    return n


def decode_streaming(dec: Wan22VAEDecoder, z: mx.array, chunk_lat: int = 1,
                     spatial_tiles: int = 1, halo: int | None = None,
                     clip_output: bool = True) -> mx.array:
    """Bit-identical streaming decode — temporal (Phase 1) + optional spatial
    halo-tiling of the high-res suffix (Phase 2).

    Args:
        dec: a loaded Wan22VAEDecoder.
        z: (B, T_lat, H_lat, W_lat, z_dim) latent (ALREADY denormalized, as `dec(z)`
           expects — caller applies denormalize_latents, same as the whole-seq path).
        chunk_lat: latent frames decoded per temporal chunk (1 = minimum extent /
           lowest memory). Result is independent of this.
        spatial_tiles: n → tile the high-res suffix n×n with real-neighbour halo +
           crop. 1 (default) = Phase-1 behaviour (whole spatial). Result is
           independent of this and of `halo` (provided halo ≥ suffix RF).
        halo: suffix-input halo in G-pixels (≥ suffix RF = 10). None → _SUFFIX_HALO
           (12, +2 margin). Smaller (≥10) shrinks per-tile memory; the bit-identity
           test is the guarantee.
        clip_output: clip to [-1, 1] like `dec.__call__` (set False to compare raw).

    Returns:
        (B, T_out, H, W, 3) decoded RGB — bit-identical to `dec(z)` for ALL T_lat,
        including T_lat == 1 (the t2i / single-frame case): the lone latent frame is
        doubled at the first upsample3d exactly like whole-seq (task #48), so the
        output frame count and values match dec(z). (chunk_lat=1 carries a ~1e-6 fp32
        reorder; chunk≥2 and the spatial-tiled path are exactly 0.0 — see tests.)
    """
    if halo is None:
        halo = _SUFFIX_HALO
    z = z.astype(dec.conv2.weight.dtype)
    T = z.shape[1]
    # T_lat==1: whole-seq dec(z) DOUBLES the lone frame at the first upsample3d
    # (its `first_chunk and T>1` test is False). Signal that down so the first chunk
    # doubles instead of 'Rep'-deferring (which only suits a multi-chunk stream).
    single = T == 1
    # one chunk total ⇒ no successor reads the per-tile suffix caches; dropping them
    # keeps the tiled footprint flat in tile-count (task #49).
    keep_cache = chunk_lat < T
    suffix_start = _suffix_start(dec.decoder)
    # generous cache: one slot per cached conv (kernel-3 convs + each upsample3d).
    fc_prefix = [None] * 64
    fc_tiles = None  # lazily built once the suffix-grid size G is known (tiled path)
    outs = []
    ci = 0
    for start in range(0, T, chunk_lat):
        first = ci == 0
        zc = z[:, start:start + chunk_lat]
        xc = dec.conv2(zc)  # 1×1×1, per-frame, no temporal cache
        fi = [0]
        if spatial_tiles <= 1:
            oc = _decoder3d_chunk(dec.decoder, xc, fc_prefix, fi, first, single)
        else:
            suffix_in = _decoder3d_prefix(dec.decoder, xc, fc_prefix, fi, first,
                                          suffix_start, single)
            mx.eval(suffix_in)
            if fc_tiles is None:  # one persistent temporal cache per real tile
                G_h, G_w = suffix_in.shape[2], suffix_in.shape[3]
                n_real = len(_tile_bounds(G_h, spatial_tiles)) * \
                    len(_tile_bounds(G_w, spatial_tiles))
                fc_tiles = [[None] * 64 for _ in range(n_real)]
                _warn_if_ineffective(G_h, G_w, spatial_tiles, halo)
            oc = _suffix_spatial_tiled(dec.decoder, suffix_in, fc_tiles,
                                       suffix_start, spatial_tiles, halo, keep_cache)
        # eval the small RGB chunk + carried caches, then RELEASE the upsampler's
        # transient pool so the footprint stays FLAT in length (one-chunk extent)
        # rather than creeping to the allocator high-water mark over chunks.
        mx.eval(oc, fc_prefix)
        outs.append(oc)
        gc.collect()
        mx.clear_cache()
        ci += 1
    out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
    out = _unpatchify(out, patch_size=2)
    if clip_output:
        out = mx.clip(out, -1.0, 1.0)
    mx.eval(out)
    return out
