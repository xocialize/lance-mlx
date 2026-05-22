# L3 — Checkerboard + VAE Corner-Clouds Audit (2026-05-22)

**Triggered by:** L3 in the polish-phase plan. User-observed residual on
Phase 5j t2v output: localized soft-cloud regions in upper-right corner
of some frames.

**Diagnostic approach:** mlx-porting skill pitfall #7 (checkerboard
spatial-op bug) and pitfall #10 (VAE numerics — color tints, gray output).

## Test 1: Codebase grep for `mx.tile` vs `mx.repeat`

Only one `mx.tile` usage in the entire pipeline:

`src/lance_mlx/model/lance_llm.py:197`:
```python
position_ids = mx.tile(position_ids, (3, 1, 1))
```

This is in the FALLBACK branch (`if position_ids is None`), broadcasting a
1D `arange(L)` to 3-axis mrope. It's effectively a broadcast — `mx.tile`
and `broadcast_to` produce equivalent output here. Not a checkerboard bug.

`mlx-video/wan_2/vae.py:294-295` uses `mx.repeat` (correct) for 2x spatial
nearest upsample in the VAE decoder. ✓

`mlx-video/wan_2/vae.py:287-288` (temporal upsample) uses `mx.stack`
followed by `reshape` — produces frame-interleaved output equivalent to
`mx.repeat` semantics. Spot-checked, correct.

**No `mx.tile`-where-`mx.repeat`-was-needed bugs found.**

## Test 2: VAE noise-path decode

`scripts/32_L3_noise_decode_audit.py` — decode pure random Gaussian
through `Wan22VAEDecoder` at the production scale (768² × 4 latent
frames → 13 video frames) across 4 seeds.

Per the skill: this isolates VAE-side bugs from LLM-side bugs. If the
VAE introduces corner artifacts on random input, the bug is downstream
of the LLM.

### Quadrant statistics

| Seed | TL mean | TR mean | BL mean | BR mean | Max deviation |
|---|---|---|---|---|---|
| 42 | 110.9 | 111.6 | 110.6 | 109.7 | **1.7%** |
| 43 | 110.7 | 110.8 | 110.1 | 110.7 | 0.6% |
| 7  | 109.8 | 110.8 | 111.6 | 111.3 | 1.6% |
| 13 | 110.0 | 111.7 | 109.0 | 109.2 | 2.4% |

All deviations well under the 5% threshold for declaring asymmetry.
Across all 4 seeds:
- Mean: 109-112 (close to grey midpoint, as expected for noise input)
- Std: 39-43 (consistent across quadrants)
- TR (upper-right) shows no systematic positive bias

**VAE produces spatially symmetric output from noise input.**

### Visual inspection

`/tmp/lance_L3_noise_decode/noise_seed42_midframe.png` and the other seeds:
all show uniform multicolored speckle noise. No checkerboard pattern at
any stride (2, 4, 8, 16). No corner-specific artifacts. No color tints
(cyan, gray, washed-out).

## Verdict

**L3 — VAE spatial ops are CORRECT.** Skill pitfall #7 (checkerboard)
and pitfall #10 (VAE numerics) are both ruled out for our pipeline.

## Implication for the corner-cloud residual

Since the VAE is symmetric, the corner-cloud artifact the user observed in
some Phase 5j t2v outputs originates UPSTREAM of the VAE — i.e., in the
LLM's velocity prediction. That makes sense given:

1. **L2 finding**: our `_build_position_ids` uses `sms=1, base=0` while
   upstream uses `sms=2, base=text_len`. The h/w mrope coords in our
   port range [0, 47] for 768² latents; upstream's range is [0, 23].
2. At the corners (positions 47), our values land further from training
   distribution than upstream's would.
3. Combined with the fact that watercolor was previously a pervasive bug
   that the Phase 5j fix mostly resolved, the residual at the spatial
   grid extremes is consistent with "position-ID is still slightly off
   compared to upstream, biggest impact at boundary positions."

## Hypothesis to test (post-v0.5)

The corner-cloud residual would likely be fixed by the **full upstream
position-ID match**:
- `sms=2` divisor on h/w in mrope construction (`_build_position_ids`)
- `base=text_len + st_idx` anchor (revert to legacy-like behavior, but
  with proper st_idx tracking from get_rope_index)
- **AND** halve the LPE index range (replace `MAX_LATENT_SIDE=64` with
  an sms-aware value, currently `64`)

Phase 5g V1 tested `sms=2` without the LPE adjustment and got subject
loss — confirming the LPE range MUST be adjusted in concert with the
mrope sms divisor.

Cleanest path: replace our `_build_position_ids` with a direct call to
`mlx_vlm.models.qwen2_5_vl.language.get_rope_index`. That's the canonical
3D-mrope builder that upstream uses. We'd need to ensure the LPE
indexing aligns to the merged-grid coords too.

## Files touched

- `scripts/32_L3_noise_decode_audit.py` — the test harness
- `notes/L3_checkerboard_vae_audit.md` — this writeup

## Status

L3 audit COMPLETE. No code change required. The polish-phase priority
list now has stronger evidence:

- ✅ L1 (done): propagation of `latent_pos_base=0` to image_edit/video_edit
  was wrong; reverted with the convention asymmetry documented
- ✅ L2 (done): documented upstream position-ID convention divergence
- ✅ L3 (done): VAE ruled out as a source of remaining residuals
- ✅ L4 (done): TimestepEmbedder t-scale matches upstream exactly

Next: L5 (republish quantized Lance-3B), L6 (DWQ 4-bit), L7 (PyPI).
The corner-cloud LLM-side fix would be a substantial refactor and is
worth doing AFTER the quantization track lands, when we can test both
configurations against the production output.
