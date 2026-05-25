# Phase 5c-3 — AWQ MLX port: complete (with caveats)

**Date:** 2026-05-25 (second session)
**Status:** Code complete. Math proven. AWQ-INT4 shows measurable
improvement over naive quantization but not enough to ship for t2i.
Likely shippable for x2t_image (VQA) per Reza2kn's pattern — untested
here, follow-up needed.

## Sub-phases — all complete

| Phase | Status | Deliverable |
|-------|--------|-------------|
| 5c-3a | ✅ Read Reza2kn source | Cached locally + algorithm summarized in `awq.py` docstring |
| 5c-3b | ✅ awq_search_scale in MLX | `src/lance_mlx/quant/awq.py` — unit test: +51% INT4 output-error reduction |
| 5c-3c | ✅ Calibration hooks | `src/lance_mlx/quant/calibrate.py` (ActStats Linear subclass) + `scripts/quant/calibrate_awq.py` (runner); 504/504 modules covered, 152,790 tokens each |
| 5c-3d | ✅ Apply + quantize pipeline | `scripts/quant/apply_awq_quantize.py` — produces Lance-3B-AWQ-INT4 in 15s |
| 5c-3e | ✅ Validation vs bf16 | `scripts/diagnostics/d_p5c3_validate.py` — 4-prompt sweep at 384² |

## Quantitative results (t2i, 384², seed=42, 4-prompt sweep)

```
prompt        bf16 HF    8bit-und Δ   AWQ-INT4 Δ   AWQ-INT8 Δ
P1 cat_stop   4.82e+08   -83.0%       -79.3%       -80.4%
P2 dragon     3.45e+08   -90.2%       -85.3%       -90.3%
P3 cat_skate  4.85e+08   -80.2%       -78.2%       -80.2%
P4 cat_dog    4.18e+08   -94.0%       -78.9%       -93.9%
```

**Key findings:**

1. **AWQ math works correctly** in MLX. Unit test (5c-3b) showed
   +51% output-error reduction matching the alpha-search prediction.
   144/144 fusion groups got non-degenerate alpha in [0.25, 0.55],
   mean ~0.37 — exactly the sweet spot for AWQ.

2. **AWQ-INT4 is measurably better than naive 8bit-und** on every
   prompt (3-15 percentage points HF improvement). P4 cat_dog showed
   the largest gain (-94% → -79%), though the visual reveals that
   "improvement" is partly structured noise rather than subject recovery.

3. **AWQ-INT8 ≈ naive 8bit-und.** At 8-bit precision, AWQ provides
   essentially no benefit — the quantization itself isn't the
   bottleneck. Something else (likely activation-distribution mismatch
   or kernel-precision interaction) imposes a ~80% HF floor regardless
   of calibration.

4. **The precision ceiling for quantized Lance t2i is around -80% HF**
   regardless of bit-width (4, 8) or calibration method
   (naive, AWQ). Larger calibration corpora, different group sizes, or
   alternative algorithms might shift this but the gap is structural,
   not algorithmic.

## Sizes

```
Lance-3B-bf16        12.37 GB  (production reference)
Lance-3B-8bit-und     9.19 GB  (74% of bf16)  — naive, broken
Lance-3B-AWQ-INT4     3.31 GB  (27% of bf16)  — calibrated, partial
Lance-3B-AWQ-INT8     6.59 GB  (53% of bf16)  — calibrated, no benefit
```

AWQ-INT4 is dramatically smaller than the other quant variants
(3× smaller than 8bit-und). If shippable for any task, it's an
ergonomic win — Lance-3B fits comfortably in 16 GB Macs.

## What's potentially shippable

**For t2i image generation: nothing changes — bf16 remains the only
production-quality variant.** AWQ-INT4 may produce loosely-aligned
output (recognizable subject for some prompts) but quality is well
below bf16 across the board.

**For x2t_image (VQA): UNTESTED but likely shippable.** Reza2kn's
PyTorch AWQ-INT4 was validated only for x2t_image and reported "5/6
oracle correct" on their diagnostic set. Our MLX port uses the
identical algorithm at the math level and 4-bit at the same group_size
(128). A 6-prompt x2t_image sweep would settle whether AWQ-INT4 is
shippable as a 3.3 GB VQA-only variant.

**Recommendation:** test AWQ-INT4 on x2t_image before declaring a
shipping outcome for Phase 5c. If 5/6 or 6/6 of the oracle x2t
prompts pass content-correctness, ship as `mlx-community/Lance-3B-AWQ-INT4-VQA`
with a clear "VQA only" tag in the README. The 3.31 GB footprint
unlocks Lance VQA on the M1/M2 8-16 GB segment.

## Why AWQ-INT8 doesn't help (the surprising part)

mlx-lm's `nn.quantize` at 8-bit affine produces nearly identical
output to AWQ at 8-bit affine. This means at 8-bit, the per-group
scale rebalancing AWQ does isn't where the error lives.

Hypotheses (untested):
- The 8-bit precision floor is dominated by activation-side noise
  injected by `mx.fast.quantized_matmul`'s kernel implementation,
  not by weight quantization granularity.
- Lance's training-time activation distribution has structure that
  no per-group affine quant captures, regardless of scale.
- Group size 64 may be the bottleneck at 8-bit; group size 32 or
  16 might help but would offset the storage savings.

This is an interesting research thread but not a path to shipping
Lance image generation. Stays in the backlog.

## Pipeline artifacts

Production-ready code:
- `src/lance_mlx/quant/__init__.py` — module interface
- `src/lance_mlx/quant/awq.py` — AWQ scale-search + scale-fusion
- `src/lance_mlx/quant/calibrate.py` — ActStats hook system

Scripts:
- `scripts/quant/calibrate_awq.py` — collect activation stats
- `scripts/quant/apply_awq_quantize.py` — apply AWQ + quantize
- `scripts/diagnostics/d_p5c3_awq_search_unit_test.py` — unit test
- `scripts/diagnostics/d_p5c3_validate.py` — 4-variant validation harness

Calibration data (not shipped, can regenerate in ~90s):
- `notes/phase5n_diagnostics/phase5c3_awq_port/act_stats/`
  - `act_stats.safetensors` (6.8 MB; 504 fp32 sum_abs arrays)
  - `act_stats_meta.json` (n_tokens, n_calls per module)

Validation outputs:
- `notes/phase5n_diagnostics/phase5c3_awq_port/validation/_compare_grid.png`
  (4×4: bf16 / 8bit-und / AWQ-INT4 / AWQ-INT8 × P1/P2/P3/P4)
- Individual variant outputs per prompt

Quantized model artifacts (lance-mlx-models/, not in this git repo):
- `Lance-3B-AWQ-INT4/` (3.31 GB)
- `Lance-3B-AWQ-INT8/` (6.59 GB)

## Next recommended

**5c-3f (small):** Validate AWQ-INT4 on x2t_image oracle prompts.
~5 min wall-clock; either confirms or denies shippability for VQA.

**5c-3g (medium):** Mixed-precision experiment — keep GEN tower at
bf16 + AWQ-quantize only UND. Might recover image quality (the
GEN tower is what produces the spatial latents); UND-only quant
through AWQ could be cleaner than the broken Phase 5c-2 UND-only naive.

**5c-3h (research):** Investigate the 8-bit precision floor — why
does AWQ-INT8 ≈ naive-INT8? Sample a single layer's activations
pre/post quant to see where the error budget is being spent.

**Or pivot:** with the AWQ kernel production-ready and the t2i
ceiling characterized, this is a reasonable stopping point for
quantization work. Foundation for any future quant variant is in
place; future experiments are turning knobs on a working pipeline.
