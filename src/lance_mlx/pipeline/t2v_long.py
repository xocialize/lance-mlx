"""Phase D — arbitrary-length t2v via chained ≤13f windows (temporal chunking).

Generates video longer than the single-shot 13-frame ceiling (Phase 2.3
loop-bound) by chaining windows: each window i>0 is conditioned on the last
``overlap_lat`` *latent* frames of window i-1 (reused DIRECTLY — no VAE
re-encode) as a CLEAN reference block, and denoises ``w_lat`` NEW frames as the
NOISY block, using Lance's two-block conditioning (CLEAN_VAE / NOISY_VAE → GEN).

Two conditioning regimes (``co_temporal`` flag, default True = Lever 1):
- CO-TEMPORAL (default, video_edit-aligned): the NOISY block OVERLAPS the CLEAN
  reference — the first k_lat noisy frames share coords with the clean frames
  ([0..k_lat) in both), so the model reconstructs the overlap to MATCH the clean
  (the regime it was TRAINED on in video_edit) and extends [k_lat..w_lat) as new
  content. The regenerated overlap is dropped; the new tail is appended. This
  pins the continuation to a coord the model has locked to real content.
- SEQUENTIAL (legacy, kept for A/B): CLEAN (k_lat past) and NOISY (w_lat future)
  take DISJOINT coords — clean [0..k_lat), noisy [k_lat..k_lat+w_lat). This OOD
  layout (the model never saw clean/noisy at disjoint coords) produced a hard
  APPEARANCE seam: window i>0's first frame didn't pixel-continue the prior tail.

Both run on a LOCAL sliding window (the lpe table caps the frame coord at 31, so
there is no absolute timeline; continuity comes from CLEAN being real prior
latents). Asymmetric generalisation of ``pipeline/video_edit.py`` (where CLEAN
and NOISY are equal-size and fully co-temporal).

Stage 1 (this file): correctness-first — a full two-block forward per window
(both experts, explicit mask), latents accumulated and decoded ONCE at the end
(spatial+temporal tiled). The free-UND optimisation (fold the step-invariant
CLEAN block into the prefill) is Stage 2.

The pure coord/lpe/mask helpers below are model-free and unit-tested
(tests/test_long_coords.py). The orchestration (generate_long) is UNVALIDATED
until the monitored 26f seam run — see PHASE_D_PLAN.md.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from lance_mlx.model.routing import PositionGroup
from .t2v import (
    MAX_LATENT_SIDE, VAE_LATENT_CHANNELS, VAE_TEMPORAL_DOWNSAMPLE,
    MAPE_ANCHOR_VIDEO_GEN, T2V_INSTRUCTION, denormalize_latents, timestep_schedule,
)


# ── pure coord / lpe / mask helpers (model-free → unit-testable) ─────────────

def long_lpe_indices(k_lat: int, w_lat: int, h_lat: int, w_grid: int,
                     co_temporal: bool = False):
    """latent_pos_embed indices for the CLEAN (k_lat frames) and NOISY (w_lat
    frames) blocks. idx = frame*64² + row*64 + col.

    SEQUENTIAL (co_temporal=False): noisy frames continue AFTER clean —
    clean 0..k_lat, noisy k_lat..k_lat+w_lat. The clean and noisy blocks never
    share a frame coord.

    CO-TEMPORAL (co_temporal=True, the video_edit-aligned regime): noisy frames
    start at 0, so the first k_lat noisy frames share their lpe index with the
    clean frames at the SAME coord (the model is trained — via video_edit — to
    denoise a co-temporal noisy frame to MATCH its clean reference). The
    remaining w_lat-k_lat noisy frames are the genuinely-new extension. The
    caller drops the first k_lat (regenerated overlap) and keeps the tail."""
    def grid(frame_offset, n_frames):
        return [
            (frame_offset + f) * (MAX_LATENT_SIDE ** 2) + r * MAX_LATENT_SIDE + c
            for f in range(n_frames) for r in range(h_lat) for c in range(w_grid)
        ]
    clean = mx.array(grid(0, k_lat), dtype=mx.int32)
    noisy = mx.array(grid(0 if co_temporal else k_lat, w_lat), dtype=mx.int32)
    return clean, noisy


def long_position_ids(
    T: int, clean_positions, noisy_positions, *,
    k_lat: int, w_lat: int, h_lat: int, w_grid: int, anchor: int,
    co_temporal: bool = False,
):
    """(3,1,T) mRoPE coords. Mirrors video_edit._build_position_ids; the t-axis of
    both blocks is MaPE-anchored to ``anchor`` and spatial coords use the
    text-position base, exactly as video_edit does.

    SEQUENTIAL (co_temporal=False): NOISY block's temporal frame is offset by
    k_lat — clean [base, base+k_lat), noisy [base+k_lat, base+k_lat+w_lat).

    CO-TEMPORAL (co_temporal=True): NOISY starts at the clean base — clean
    [base, base+k_lat), noisy [base, base+w_lat). The first k_lat noisy frames
    coincide in (t,h,w) with the clean reference (video_edit's regime); frames
    [base+k_lat, base+w_lat) are the new extension."""
    pos = np.zeros((3, 1, T), dtype=np.int32)
    seq = np.arange(T, dtype=np.int32)
    pos[0, 0, :] = seq
    pos[1, 0, :] = seq
    pos[2, 0, :] = seq

    noisy_t_offset = 0 if co_temporal else k_lat
    base = clean_positions[0]
    for idx, tp in enumerate(clean_positions):          # CLEAN: temporal f in [0, k_lat)
        f = idx // (h_lat * w_grid)
        rest = idx % (h_lat * w_grid)
        pos[0, 0, tp] = base + f
        pos[1, 0, tp] = base + rest // w_grid
        pos[2, 0, tp] = base + rest % w_grid
    for idx, tp in enumerate(noisy_positions):          # NOISY: temporal f offset by 0 (co-temporal) or k_lat
        f = idx // (h_lat * w_grid)
        rest = idx % (h_lat * w_grid)
        pos[0, 0, tp] = base + noisy_t_offset + f
        pos[1, 0, tp] = base + rest // w_grid
        pos[2, 0, tp] = base + rest % w_grid

    max_temporal = w_lat if co_temporal else (k_lat + w_lat)
    max_grid = max(max_temporal, h_lat, w_grid) - 1
    after_clean = clean_positions[-1] + 1
    before_noisy = noisy_positions[0]
    gap_len = 0
    if before_noisy > after_clean:
        gap_len = before_noisy - after_clean
        tail = base + max_grid + 1 + np.arange(gap_len, dtype=np.int32)
        pos[:, 0, after_clean:before_noisy] = tail[None, :]
    after_noisy = noisy_positions[-1] + 1
    if after_noisy < T:
        tail_start = base + max_grid + 1 + gap_len
        tail = tail_start + np.arange(T - after_noisy, dtype=np.int32)
        pos[:, 0, after_noisy:] = tail[None, :]

    first_latent_t = int(pos[0, 0, clean_positions[0]])  # MaPE re-anchor t-axis to `anchor`
    shift = anchor - first_latent_t
    for tp in list(clean_positions) + list(noisy_positions):
        pos[0, 0, tp] += shift
    return mx.array(pos)


def long_block_mask(T, clean_positions, noisy_positions, dtype):
    """Causal + bidirectional-within-each-block (identical to video_edit)."""
    i = mx.arange(T)[:, None]
    j = mx.arange(T)[None, :]
    cs, ce = clean_positions[0], clean_positions[-1] + 1
    ns, ne = noisy_positions[0], noisy_positions[-1] + 1
    bidir_clean = ((i >= cs) & (i < ce)) & ((j >= cs) & (j < ce))
    bidir_noisy = ((i >= ns) & (i < ne)) & ((j >= ns) & (j < ne))
    allowed = (i >= j) | bidir_clean | bidir_noisy
    return mx.where(allowed, mx.array(0.0, dtype=dtype), mx.array(-1e9, dtype=dtype))


# ── two-block window state + denoise (Stage 1: full forward) ─────────────────

def _prepare_two_block_state(pipe, *, prompt, k_lat, w_lat, h_lat, w_grid, anchor,
                             empty_prompt=False, co_temporal=False, verbose=False):
    n_clean = k_lat * h_lat * w_grid
    n_noisy = w_lat * h_lat * w_grid
    # Mirror t2v's "ours" template (T2V_INSTRUCTION in system, prompt in user) +
    # video_edit's two-block layout (CLEAN ref in the user turn, NOISY target in
    # the assistant turn). empty_prompt → CFG uncond arm.
    user_prompt = "" if empty_prompt else prompt
    clean_pad = "<|video_pad|>" * n_clean
    noisy_pad = "<|video_pad|>" * n_noisy
    text = (
        f"<|im_start|>system\n{T2V_INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{clean_pad}<|vision_end|>{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<|vision_start|>{noisy_pad}<|vision_end|>"
    )
    tok = pipe.processor.tokenizer
    input_ids = mx.array([tok(text, add_special_tokens=False)["input_ids"]], dtype=mx.int32)
    T = input_ids.shape[1]
    ids = input_ids[0].tolist()
    pad_positions = [i for i, v in enumerate(ids) if v == pipe.video_pad_token_id]
    assert len(pad_positions) == n_clean + n_noisy, (
        f"expected {n_clean + n_noisy} video_pad, found {len(pad_positions)}")
    clean_positions = pad_positions[:n_clean]
    noisy_positions = pad_positions[n_clean:]

    position_ids = long_position_ids(
        T, clean_positions, noisy_positions,
        k_lat=k_lat, w_lat=w_lat, h_lat=h_lat, w_grid=w_grid, anchor=anchor,
        co_temporal=co_temporal,
    )
    pg = np.full((T,), int(PositionGroup.TEXT), dtype=np.int32)
    pg[np.array(clean_positions)] = int(PositionGroup.CLEAN_VAE)
    pg[np.array(noisy_positions)] = int(PositionGroup.NOISY_VAE)
    text_embeds = pipe.lance_model.embed_tokens(input_ids)
    mask = long_block_mask(T, clean_positions, noisy_positions, dtype=text_embeds.dtype)

    # Optimized split: prefix = [text ‖ CLEAN ‖ mid-tail] (everything before the
    # NOISY block), loop = NOISY block. The prefix-KV path needs the NOISY block
    # contiguous and starting at P; the CLEAN block lives inside the prefix
    # (step-invariant: time 0, fixed z_clean) so it folds into the prefill.
    P = noisy_positions[0]
    noisy_end = noisy_positions[-1] + 1
    assert noisy_positions == list(range(P, noisy_end)), (
        "two-block prefix-KV needs a contiguous NOISY block; "
        f"got {len(noisy_positions)} spanning {P}..{noisy_end - 1}")
    if verbose:
        print(f"  window seq T={T}  clean={n_clean}@[{clean_positions[0]}..{clean_positions[-1]}] "
              f"noisy={n_noisy}@[{noisy_positions[0]}..{noisy_positions[-1]}]  P={P}")
    return {
        "text_embeds": text_embeds, "position_ids": position_ids,
        "position_group": mx.array(pg), "mask": mask,
        "clean_arr": mx.array(clean_positions, dtype=mx.int32),
        "noisy_arr": mx.array(noisy_positions, dtype=mx.int32),
        # optimized slices:
        "P": P,
        "position_ids_prefix": position_ids[:, :, :P],
        "position_group_prefix": mx.array(pg[:P]),
        "mask_prefix": mask[:P, :P],
        "position_ids_noisy": position_ids[:, :, P:noisy_end],
    }


def _two_block_velocity(pipe, *, state, z_clean, z_t, t, clean_lpe, noisy_lpe,
                        k_lat, w_lat, h_lat, w_grid):
    m = pipe.lance_model
    z_clean_flat = z_clean.reshape(1, k_lat * h_lat * w_grid, VAE_LATENT_CHANNELS)
    z_t_flat = z_t.reshape(1, w_lat * h_lat * w_grid, VAE_LATENT_CHANNELS)
    t0 = mx.zeros((1,), dtype=t.dtype)
    clean_embed = m.vae_in_proj(z_clean_flat) + m.latent_pos_embed(clean_lpe)[None, ...] \
        + m.time_embedder(t0).reshape(1, 1, -1)
    noisy_embed = m.vae_in_proj(z_t_flat) + m.latent_pos_embed(noisy_lpe)[None, ...] \
        + m.time_embedder(t.reshape(1)).reshape(1, 1, -1)

    out = np.array(state["text_embeds"].astype(mx.float32))
    out[:, np.array(state["clean_arr"]), :] = np.array(clean_embed.astype(mx.float32))
    out[:, np.array(state["noisy_arr"]), :] = np.array(noisy_embed.astype(mx.float32))
    inputs_embeds = mx.array(out).astype(state["text_embeds"].dtype)

    h = m(inputs_embeds=inputs_embeds, position_ids=state["position_ids"],
          position_group=state["position_group"], mask=state["mask"])
    v = m.llm2vae(h[:, state["noisy_arr"], :])
    return v.reshape(1, w_lat, h_lat, w_grid, VAE_LATENT_CHANNELS)


def _denoise_two_block(pipe, *, prompt, z_clean, k_lat, w_lat, h_lat, w_grid,
                       num_steps, cfg_scale, timestep_shift, seed, anchor,
                       co_temporal=False, cfg_renorm_min=0.0, verbose=False):
    """Denoise the w_lat-frame NOISY block conditioned on z_clean (k_lat frames).
    Returns the denoised noisy latents (1, w_lat, h_lat, w_grid, 48). Under
    co_temporal the first k_lat frames are the regenerated overlap (caller drops
    them); under sequential all w_lat frames are new."""
    cond = _prepare_two_block_state(pipe, prompt=prompt, k_lat=k_lat, w_lat=w_lat,
                                    h_lat=h_lat, w_grid=w_grid, anchor=anchor,
                                    co_temporal=co_temporal, verbose=verbose)
    uncond = (None if cfg_scale <= 1.0 else
              _prepare_two_block_state(pipe, prompt=prompt, k_lat=k_lat, w_lat=w_lat,
                                       h_lat=h_lat, w_grid=w_grid, anchor=anchor,
                                       co_temporal=co_temporal, empty_prompt=True))
    clean_lpe, noisy_lpe = long_lpe_indices(k_lat, w_lat, h_lat, w_grid, co_temporal=co_temporal)

    mx.random.seed(seed)
    z_t = mx.random.normal((1, w_lat, h_lat, w_grid, VAE_LATENT_CHANNELS)).astype(z_clean.dtype)
    sched = timestep_schedule(num_steps=num_steps, shift=timestep_shift)
    for step in range(num_steps):
        t, dt = sched[step], sched[step] - sched[step + 1]
        kw = dict(z_clean=z_clean, z_t=z_t, t=t, clean_lpe=clean_lpe, noisy_lpe=noisy_lpe,
                  k_lat=k_lat, w_lat=w_lat, h_lat=h_lat, w_grid=w_grid)
        v_cond = _two_block_velocity(pipe, state=cond, **kw)
        if uncond is not None:
            v_uncond = _two_block_velocity(pipe, state=uncond, **kw)
            v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
            nc = mx.sqrt(mx.sum(v_cond * v_cond, axis=-1, keepdims=True))
            nf = mx.sqrt(mx.sum(v_cfg * v_cfg, axis=-1, keepdims=True))
            velocity = v_cfg * mx.clip(nc / (nf + 1e-8), cfg_renorm_min, 1.0)
        else:
            velocity = v_cond
        z_t = z_t - velocity * dt
        mx.eval(z_t)
    return z_t


def _scatter_clean_into_prefix(pipe, state, clean_embed):
    """Fold the step-invariant CLEAN block into the text prefix and slice [0:P]."""
    out = np.array(state["text_embeds"].astype(mx.float32))
    out[:, np.array(state["clean_arr"]), :] = np.array(clean_embed.astype(mx.float32))
    return mx.array(out[:, :state["P"], :]).astype(state["text_embeds"].dtype)


def _denoise_two_block_opt(pipe, *, prompt, z_clean, k_lat, w_lat, h_lat, w_grid,
                           num_steps, cfg_scale, timestep_shift, seed, anchor,
                           co_temporal=False, cfg_renorm_min=0.0, verbose=False):
    """Stage-2 optimized window: prefill [text ‖ CLEAN] once per CFG arm (UND
    path), then loop ONLY the NOISY frames through the GEN-only stack against the
    cached prefix. The UND tower is NOT freed (the next window re-prefills). The
    gen-loop's unmasked SDPA gives noisy→{text,clean} conditioning + noisy→noisy
    bidirectionality for free — same semantics as the full two-block mask.

    Under co_temporal the NOISY block includes the regenerated overlap; the
    prefix (text ‖ CLEAN) is unchanged, only the noisy coords shift to coincide
    with the clean frames (handled inside _prepare_two_block_state / lpe)."""
    m = pipe.lance_model
    cond = _prepare_two_block_state(pipe, prompt=prompt, k_lat=k_lat, w_lat=w_lat,
                                    h_lat=h_lat, w_grid=w_grid, anchor=anchor,
                                    co_temporal=co_temporal, verbose=verbose)
    uncond = (None if cfg_scale <= 1.0 else
              _prepare_two_block_state(pipe, prompt=prompt, k_lat=k_lat, w_lat=w_lat,
                                       h_lat=h_lat, w_grid=w_grid, anchor=anchor,
                                       co_temporal=co_temporal, empty_prompt=True))
    clean_lpe, noisy_lpe = long_lpe_indices(k_lat, w_lat, h_lat, w_grid, co_temporal=co_temporal)

    # Step-invariant CLEAN embed (time 0), folded into each arm's prefix → prefill.
    z_clean_flat = z_clean.reshape(1, k_lat * h_lat * w_grid, VAE_LATENT_CHANNELS)
    dt0 = mx.zeros((1,), dtype=mx.float32)
    clean_embed = (m.vae_in_proj(z_clean_flat) + m.latent_pos_embed(clean_lpe)[None, ...]
                   + m.time_embedder(dt0).reshape(1, 1, -1))
    for st in (cond, uncond):
        if st is None:
            continue
        prefix_embeds = _scatter_clean_into_prefix(pipe, st, clean_embed)
        st["caches"] = m.prefill_prefix(
            prefix_embeds, position_ids=st["position_ids_prefix"],
            position_group=st["position_group_prefix"], mask=st["mask_prefix"])
    # NOTE: do NOT free_und_tower — the next window must re-prefill (UND path).

    mx.random.seed(seed)
    z_t = mx.random.normal((1, w_lat, h_lat, w_grid, VAE_LATENT_CHANNELS)).astype(cond["text_embeds"].dtype)
    sched = timestep_schedule(num_steps=num_steps, shift=timestep_shift)
    n_noisy = w_lat * h_lat * w_grid
    for step in range(num_steps):
        t, dt = sched[step], sched[step] - sched[step + 1]
        noisy_embed = (m.vae_in_proj(z_t.reshape(1, n_noisy, VAE_LATENT_CHANNELS))
                       + m.latent_pos_embed(noisy_lpe)[None, ...]
                       + m.time_embedder(t.reshape(1)).reshape(1, 1, -1))
        noisy_embed = noisy_embed.astype(cond["text_embeds"].dtype)

        def _vel(st):
            h = m.gen_loop_forward(noisy_embed, position_ids=st["position_ids_noisy"],
                                   caches=st["caches"])
            return m.llm2vae(h).reshape(1, w_lat, h_lat, w_grid, VAE_LATENT_CHANNELS)

        v_cond = _vel(cond)
        if uncond is not None:
            v_uncond = _vel(uncond)
            v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
            nc = mx.sqrt(mx.sum(v_cond * v_cond, axis=-1, keepdims=True))
            nf = mx.sqrt(mx.sum(v_cfg * v_cfg, axis=-1, keepdims=True))
            velocity = v_cfg * mx.clip(nc / (nf + 1e-8), cfg_renorm_min, 1.0)
        else:
            velocity = v_cond
        z_t = z_t - velocity * dt
        mx.eval(z_t)
    return z_t


def _auto_temporal_tile(total_t, requested_tile, requested_overlap):
    """Bound the temporal decode so per-tile activation stays inside the
    validated single-shot ≤13f envelope, for ANY total length.

    The whole-buffer decode peak scales with the temporal extent fed to the VAE:
    4 latent frames (≤13f) ≈ 12.5 GB at 768², but 7 latent frames (26f) hit
    17.59 GB — over the 16 GB envelope. ``decode_with_tiling`` (temporal_scale=4,
    causal_temporal=True) already splits the buffer into overlapping temporal
    tiles, decodes each as a causal first-chunk, and blends them with a
    trapezoidal ramp over the overlap — exactly the "decode-with-context" needed
    to hide the causal-padding boundary. We just have to ASK for it.

    Returns (tile_size_in_frames, overlap_in_frames) in OUTPUT-frame units for
    TemporalTilingConfig, or (None, 0) when the buffer is already ≤4 latent
    frames (one window → whole decode IS the validated profile). An explicit
    caller value is respected verbatim.

    tile_size_in_frames=16 → 16//temporal_scale(4) = 4 latent frames/tile (the
    13f profile); overlap=8 → 2 latent frames of causal context + blend ramp.
    Both satisfy TemporalTilingConfig's ≥16 / divisible-by-8 constraints.
    """
    if requested_tile is not None:
        return requested_tile, requested_overlap
    if total_t <= 4:
        return None, 0
    return 16, 8


def _decode_full(pipe, latents, *, tile_vae, vae_tile_px, vae_tile_overlap_px,
                 vae_temporal_tile, vae_temporal_overlap):
    # LOSSLESS streaming decode (bit-identical to whole-seq; replaces lossy decode_tiled).
    # The accumulated long-video buffer is the longest decode in the codebase, so the
    # chunk_lat=1 temporal stream (flat-in-length) matters most here. max_n=4 caps the
    # suffix tiling (video = multi-chunk → U-curve live). The vae_tile_px/temporal knobs
    # are now unused (chunk_lat + suggest_spatial_tiles supersede them); kept for callers.
    z = denormalize_latents(latents).astype(pipe.vae_decoder.conv2.weight.dtype)
    if not tile_vae:
        return pipe.vae_decoder(z)
    from lance_mlx.model.vae_stream import decode_streaming, suggest_spatial_tiles
    n_tiles = suggest_spatial_tiles(z.shape[2], z.shape[3], max_n=4)
    return decode_streaming(pipe.vae_decoder, z, chunk_lat=1, spatial_tiles=n_tiles)


def generate_long(
    pipe, prompt: str, total_frames: int, *,
    window_frames: int = 13, overlap_lat: int = 1,
    height: int = 768, width: int = 768, num_steps: int = 30, cfg_scale: float = 4.0,
    timestep_shift: float = 3.5, seed: int = 42, mape_anchor: int | None = None,
    tile_vae: bool = True, vae_tile_px: int = 256, vae_tile_overlap_px: int = 64,
    vae_temporal_tile: int | None = None, vae_temporal_overlap: int = 0,
    optimized: bool = False, free_und_before_decode: bool = True,
    co_temporal: bool = True, return_latents: bool = False, verbose: bool = False,
):
    """Generate `total_frames` of video by chaining ≤`window_frames` windows.

    Window 0 = plain t2v (captured as latents via return_latents). Window i>0
    conditions on the last `overlap_lat` latent frames of the running buffer.
    All latents are accumulated and decoded ONCE (spatial+temporal tiled).

    `optimized`: False = Stage 1 (full two-block mx.where forward per window).
    True = Stage 2 — fold the step-invariant CLEAN block into the prefix-KV
    prefill and gen-loop only the NOISY frames. Unlike single-shot t2v, the UND
    tower is KEPT resident across windows (free_und=False): every window
    re-prefills via the UND path, so the tower must survive between windows.
    Per-window peak ≈ the single-shot ≤13f prefill profile (~14 GB, fits 16);
    the win is the GEN-only loop, not the freeing.

    `free_und_before_decode` (default True): once the LAST window is done, no
    more prefills/gen-loops touch UND, so free it (~6.17 GB) before the final
    decode. This matters: with UND resident the bounded 768² decode peaks
    ~17.25 GB (over 16); freeing first drops resident 13.98→7.81 GB so the same
    decode transient lands ~11 GB. PERMANENT (delattr) → one generate_long per
    process. Set False only to reuse the pipe for a second run in the same
    process (e.g. a dual-anchor A/B).

    mape_anchor: t-axis anchor for the conditioned blocks (None → t2v default
    2000). The 1000-vs-2000 A/B is CLOSED — 1000≈2000, use 2000 (the default).

    `co_temporal` (default True — Lever 1): condition window i>0 the way
    video_edit was TRAINED. The NOISY block overlaps the CLEAN reference by
    `overlap_lat` frames at the SAME (t,h,w) coords; the model reconstructs the
    overlap to MATCH the clean frames (appearance lock) and extends the rest, so
    the new frames continue from a coord the model has pinned to real content
    rather than jumping off a single static reference at a disjoint coord. The
    regenerated overlap is dropped; only the new tail is appended. Costs
    `overlap_lat` new-frames of per-window budget (noisy block stays ≤ window_t).
    Set False for the legacy SEQUENTIAL regime (clean/noisy at disjoint coords —
    the OOD layout that produced the hard appearance seam; kept for A/B).
    """
    anchor = MAPE_ANCHOR_VIDEO_GEN if mape_anchor is None else int(mape_anchor)
    h_lat, w_grid = height // 16, width // 16
    window_t = (window_frames - 1) // VAE_TEMPORAL_DOWNSAMPLE + 1
    total_t = (total_frames - 1) // VAE_TEMPORAL_DOWNSAMPLE + 1
    if verbose:
        print(f"[long] anchor={anchor} window_t={window_t} total_t={total_t} "
              f"k_lat={overlap_lat} h_lat={h_lat} w_grid={w_grid}")

    # Window 0: plain t2v, latents captured (no decode). Under optimized=True we
    # keep the UND tower (free_und=False) so windows i>0 can re-prefill; under
    # Stage-1 (optimized=False) the full mx.where forward also keeps UND.
    latents = pipe.generate(
        prompt, num_frames=window_frames, height=height, width=width,
        num_steps=num_steps, cfg_scale=cfg_scale, timestep_shift=timestep_shift,
        seed=seed, optimized=optimized, free_und=False, tile_vae=tile_vae,
        return_latents=True, verbose=verbose,
    )
    produced = window_t
    _denoise = _denoise_two_block_opt if optimized else _denoise_two_block

    win = 1
    while produced < total_t:
        z_clean = latents[:, -overlap_lat:, :, :, :]
        # Co-temporal spends `overlap_lat` of the window budget on the
        # regenerated overlap, so it yields fewer NEW frames per window; the
        # NOISY block (overlap + new) still stays within the validated window_t
        # envelope. Sequential spends the whole budget on new frames.
        new_budget = window_t - (overlap_lat if co_temporal else 0)
        new_t = min(new_budget, total_t - produced)
        noisy_t = (overlap_lat + new_t) if co_temporal else new_t
        if verbose:
            print(f"[long] window {win}: clean={overlap_lat}f -> "
                  f"noisy={noisy_t}f (new={new_t}f, "
                  f"{'co-temporal overlap' if co_temporal else 'sequential'})")
        z_out = _denoise(
            pipe, prompt=prompt, z_clean=z_clean, k_lat=overlap_lat, w_lat=noisy_t,
            h_lat=h_lat, w_grid=w_grid, num_steps=num_steps, cfg_scale=cfg_scale,
            timestep_shift=timestep_shift, seed=seed + win, anchor=anchor,
            co_temporal=co_temporal, verbose=verbose,
        )
        # Co-temporal: drop the regenerated overlap (first overlap_lat frames),
        # keep only the genuinely-new tail. Sequential: every frame is new.
        z_new = z_out[:, overlap_lat:, :, :, :] if co_temporal else z_out
        latents = mx.concatenate([latents, z_new], axis=1)
        produced += new_t
        win += 1

    # Escape hatch for decode-side experiments: hand back the accumulated latent
    # buffer (1, total_t, h_lat, w_grid, 48) with UND STILL resident and no
    # decode, so a probe can decode the SAME latents multiple ways (whole vs
    # temporal-tiled) and isolate decode artifacts from generation seams.
    if return_latents:
        mx.eval(latents)
        return latents

    # Generation is complete — no more prefills or gen-loops reference the UND
    # tower. Free it (~6 GB) BEFORE decode so the bounded-decode transient runs
    # against an ~8 GB resident baseline instead of ~14 GB (UND kept resident
    # across windows costs ~2 GB more headroom than single-shot, which frees it).
    # PERMANENT (delattr): one generate_long per process — matches single-shot.
    # Disable (free_und_before_decode=False) only to keep the pipe alive for a
    # second run in the same process (e.g. a dual-anchor A/B).
    if free_und_before_decode:
        stats = pipe.lance_model.free_und_tower()
        if verbose:
            print(f"[long] freed UND pre-decode: -{stats['freed_bytes']/1e9:.2f} GB "
                  f"→ resident {stats['active_after']/1e9:.2f} GB")

    # Bound the final decode to the validated ≤13f envelope. The accumulated
    # buffer can be arbitrarily long; without temporal tiling the whole-buffer
    # decode overshoots 16 GB (7 latent frames = 17.59 GB). Auto-engage causal
    # temporal tiling (≤4 latent frames/tile, blended over a 2-frame overlap)
    # whenever the buffer exceeds one window. See _auto_temporal_tile.
    vt, vto = _auto_temporal_tile(latents.shape[1], vae_temporal_tile, vae_temporal_overlap)
    if verbose:
        print(f"[long] decode {latents.shape[1]} latent frames @ {height}² "
              f"(spatial tile {vae_tile_px}px"
              + (f", temporal tile {vt}f/ov{vto}f" if vt is not None else ", no temporal tiling")
              + ")")
    decoded = _decode_full(pipe, latents, tile_vae=tile_vae, vae_tile_px=vae_tile_px,
                           vae_tile_overlap_px=vae_tile_overlap_px,
                           vae_temporal_tile=vt, vae_temporal_overlap=vto)
    mx.eval(decoded)
    frames_np = np.array(decoded[0].astype(mx.float32))
    return ((frames_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
