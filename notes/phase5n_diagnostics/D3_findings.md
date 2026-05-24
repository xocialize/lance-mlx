# Phase 5n / D3 — t2v mape_anchor sweep

**Date:** 2026-05-24
**Tests:** the genuinely-novel mrope asymmetry — t2i unconditionally
applies +1000 t-axis shift (`MAPE_ANCHOR_IMAGE_GEN`), t2v applies
nothing (`mape_anchor=None` per Phase 5d). The intermediate value
1000 (matching t2i) had never been tested.

**Result:** **REFUTED.** anchor=1000 produces visually-equivalent
output to anchor=None at the Phase-5j-sensitive scale. anchor=2000
reproduces the Phase 5d slight oversaturation but does not show the
dramatic watercolor effect (that was a *different* fix, `latent_pos_base`).

## Method

Two passes:

**D3 — short-prompt scale sweep:** 256²×9f (n_lat=768), 5 anchors
(None, 500, 1000, 1500, 2000), simple prompt ("red fire truck on snowy
street"). All five produced near-identical close-up red-truck output.
No anchor sensitivity at small scale + short prompt — consistent with
Phase 5j's finding that the position-ID effect requires verbose prompts
+ larger n_lat.

**D3b — Phase-5j-sensitive conditions:** 256²×17f (n_lat=1280), 3
anchors (None, 1000, 2000), canonical 59-word Phase 5j red-panda oracle
prompt. This is the exact scale Phase 5j proved is position-ID sensitive.

## Visual verdict

`notes/phase5n_diagnostics/d3b_t2v_mape_anchor_phase5j_scale/_compare_grid.png`
(3 rows × 3 frames: early, mid, late):

- **anchor=None** (current default): naturalistic red panda with gold
  cap, recognizable subject, naturalistic blue sky / water tones.
- **anchor=1000** (match t2i): essentially equivalent — same subject,
  same composition, slightly different incidental detail (seed RNG
  through different rope rotations).
- **anchor=2000** (legacy default, Phase 5d-refuted): more saturated
  / orange tones across all frames, slight aesthetic shift toward
  illustrated. Consistent with Phase 5d's "painterly" characterization
  but mild at this scale.

None of the three shows the dramatic watercolor failure Phase 5j
documented for verbose prompts — because Phase 5j's fix
(`latent_pos_base=0`) is already on by default and is the dominant
position-ID lever.

## Conclusion

**mape_anchor is not the differential between t2i and t2v.** t2i runs
in the +1000 regime by accident of its hardcoded shift; t2v at None
produces equivalent quality. Asking t2v to "match" t2i by setting
anchor=1000 doesn't unlock any improvement. The two pipelines'
divergent t-axis treatment is structurally interesting but not
qualitatively meaningful at the tested conditions.

## What this means for the user's question

D1 (VAE temporal mode), D2 (CFG renorm at scale), and D3
(mape_anchor) — the three hypotheses extracted from the deep
pipeline analysis — are all refuted. The image-vs-video quality gap
**cannot** be cleanly explained by any of the structural divergences
I could see between the two pipelines.

Remaining live possibilities (none of which can be cheaply A/B'd
the way D1-D3 were):

1. **Training data imbalance.** Lance was trained on a substantially
   larger image corpus than video corpus. Cleanly explains both lower
   detail and lower prompt adherence in video without any code-side
   cause. The user dismissed this up front, but the diagnostic evidence
   now points back to it as the most likely cause.

2. **Wan2.2 VAE's latent space.** It's a video VAE; image latents are
   technically in-distribution but the encoder/decoder's "comfort zone"
   may bias toward video frame statistics. Hard to test without
   retraining.

3. **bf16 precision over longer sequences.** The video pipeline runs
   attention over ~5-10× more tokens per step. Even though attention's
   softmax internally promotes to fp32, the accumulated residual stream
   and layer-norm computations could drift more for longer sequences.
   Phase 5g rope_fp32 + attention_fp32 tests came back null, but those
   only covered specific spots. A wider precision audit at production
   scale is possible but expensive.

4. **Untested at production scale.** All three diagnostics ran at
   ≤ n_lat 1728 to fit the laptop time budget. Production scale is
   n_lat ~12k+. The behavior could be qualitatively different there
   without us seeing it in the small-scale tests. Particularly D2's
   CFG renorm trend was leveling off, and D3 showed no anchor effect
   at small scale — but production-scale rerun could surface effects
   too subtle for the small tests.

## Next-step options for the user to choose

- **D4 (cheap, ~2 min):** Run t2v at num_frames=1 (single frame video)
  vs t2i at the same prompt+seed. Isolates pipeline-code-side bugs from
  weight-side differences. If t2v-at-t_lat=1 ≈ t2i, the gap is in the
  multi-frame code path. If t2v-at-t_lat=1 still degrades, the gap is in
  Lance_3B_Video's weights themselves (suggests training-data answer).

- **D5 (medium, ~20 min):** Re-run D2 at production scale (768²×17f,
  n_lat=11520) to definitively settle the CFG renorm extrapolation
  question.

- **D6 (medium, ~30 min):** Run t2v with `cfg_renorm_type="none"` at
  production scale — measures the max-possible CFG signal. If the
  output is sharper and more prompt-aligned (and doesn't blow up), CFG
  renorm is costing more than we measured.

- **Accept and pivot:** Conclude the gap is training-data-driven, lower
  priority than user-perceived issues we can actually fix, and pivot to:
  - The flagged image-quality side-task (frame-0 vs frame-2 of T=1
    decode)
  - Phase 5b quantization work (DWQ-calibrated 8-bit)
  - Production-scale issue #1 (mesh artifacts at n_lat ≥ ~30k)

## Scripts and data

- `scripts/diagnostics/d3_t2v_mape_anchor_sweep.py` — small-scale sweep
- `scripts/diagnostics/d3b_t2v_mape_anchor_phase5j_scale.py` — Phase
  5j-scale sweep
- `notes/phase5n_diagnostics/d3_t2v_mape_anchor_sweep/_compare_*.png`
- `notes/phase5n_diagnostics/d3b_t2v_mape_anchor_phase5j_scale/_compare_grid.png`
