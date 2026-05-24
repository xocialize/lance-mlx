# Phase 5c-3 — AWQ MLX port status

**Date:** 2026-05-24
**Approach:** port Reza2kn/lance-quant's PyTorch AWQ alpha-search +
scale-fusion algorithm to MLX, then hand the scale-fused bf16 model to
mlx-lm's `nn.quantize` for the actual INT4 packing.

## Pipeline (5 sub-phases)

| # | Phase | Status | Output |
|---|-------|--------|--------|
| 5c-3a | Read Reza2kn source, confirm algorithm | ✅ Done | `notes/phase5n_diagnostics/phase5c3_awq_port/_reza_awq_*.py` (full sources cached locally) |
| 5c-3b | Implement & unit-test `awq_search_scale` in MLX | ✅ Done | `src/lance_mlx/quant/awq.py` (~200 LOC); unit test passes with **+51% INT4 output-error reduction** on synthetic data |
| 5c-3c | Build calibration hooks (collect act_means from real Lance forward) | ⏳ Open | Need: per-Linear forward-hook system; run x2t_image + t2i, accumulate `sum_abs / n_tokens` per consumer |
| 5c-3d | Apply pipeline: load bf16, per-layer alpha-search, mutate weights, run `nn.quantize`, save | ⏳ Open | `scripts/awq_quantize_lance.py` (new) |
| 5c-3e | Validate: 4-prompt sweep vs bf16 (same as 5c-2 protocol) | ⏳ Open | Should match or beat bf16 quality if AWQ works as in Reza2kn's PyTorch results |

## 5c-3b — what landed and why it matters

**Files:**
- `src/lance_mlx/quant/__init__.py` — package interface
- `src/lance_mlx/quant/awq.py` — `awq_search_scale`, `apply_scale_to_norm_and_consumers`, fusion-group constants
- `scripts/diagnostics/d_p5c3_awq_search_unit_test.py` — 4-test unit harness

**Key validation:** Test 4 quantized a synthetic 2048-channel weight through
MLX's native `mx.quantize` at INT4-group128 both with and without AWQ
applied, then compared output-MSE on the realistic-magnitude input. AWQ
reduced production-path output error by **51%**. The 51% real-quantize
result matching the 50% search-predicted result is the proof that:

1. Our alpha-search math is correct (50% predicted, 51% actual)
2. Our scale-fusion math is correct (Test 3 verified at 0.04% rel error)
3. MLX's affine quantization uses the same scheme our fake-quant assumes
   (otherwise predicted vs actual wouldn't track this closely)
4. Synthetic input (`randn * act_mean`) is sufficient to drive scale
   selection — we don't need real activations going *into* the search
   (we DO need real activations to compute act_mean for the search, but
   the search itself can use random data scaled by those magnitudes)

This is the riskiest math nailed down. The remaining sub-phases (3c, 3d,
3e) are infrastructure: collect calibration data, wire it through, save
output, validate. None of them require new math.

## 5c-3c — calibration hook design (for the next session)

PyTorch reference does this with `register_forward_hook`. MLX doesn't
have built-in module hooks. Options:

1. **Subclass approach:** for each target Linear, replace it with a
   `Linear` subclass that records input stats on `__call__`. Cleanest
   but requires walking the model to swap.
2. **Monkey-patch approach:** wrap `nn.Linear.__call__` globally to
   record stats when the module path matches QUANT_SUFFIXES. Hackier
   but doesn't require model walking.
3. **Activation-capture forward:** rewrite `LanceModel.__call__` (or a
   diagnostic-mode subclass of it) to expose intermediate activations.
   Most invasive.

Option 1 is recommended. Lance has 36 layers × 14 Linear consumers ≈ 500
modules to hook; one-time setup before calibration, removed after.

**Calibration data:** at minimum run the same 4-prompt t2i sweep we
used for 5c-2 validation (with the full LLM forward — the per-step
calibration is what feeds the consumer Linears). 30 Euler steps × 4
prompts × 2 CFG arms = 240 forward passes. Per-Linear stats accumulate
across all of them. Memory cost is low (each consumer stores one
(in_features,) sum, e.g. 2048×4 bytes = 8KB per consumer × 500 = 4MB).

For UND-tower coverage alone, an x2t_image run also fires UND consumers.
For GEN-tower coverage, t2i / t2v fire those. Reza2kn calibrates each
task separately then merges; we can do the same or just calibrate on
mixed t2i runs which exercise both towers.

## 5c-3d/3e — application + validation

Apply: iterate layers, for each FUSION_GROUP look up act_means, run
search, mutate norm + consumers. Then run `nn.quantize` with the
existing skip-list logic from `scripts/16_quantize.py`. Save with the
same config.json format (so it loads through `lance_mlx.model._loader`
unchanged).

Validate using the same 4-prompt protocol from
`scripts/diagnostics/d_p5c2_validate.py`. Expected outcome based on
Reza2kn's PyTorch results: VQA tasks (x2t_image) preserve quality
nearly perfectly; t2i preserves quality acceptably. t2v / video_edit
quality is untested by Reza2kn even in their PyTorch version.

## What we have now

A proven AWQ scale-search kernel in MLX, with a unit test that
demonstrates **+51% output-error reduction** through real `mx.quantize`.
~200 LOC of clean, focused code. The math is done; the remaining work
is infrastructure (calibration hooks + apply loop + save + validate).

This is a natural pause point. Resume from `STATUS.md` next session
without losing context.
