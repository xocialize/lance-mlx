# Phase 5c-2 — naive 8-bit UND-only quantization (negative result)

**Date:** 2026-05-24
**Goal:** test whether naive 8-bit groupwise quantization on the UND
tower only (GEN tower preserved at bf16) preserves Lance image
generation quality enough to ship a usable quant variant — avoiding
the need for calibration (DWQ/AWQ).
**Result:** ❌ **Does not work.** Naive 8-bit UND-only degrades image
quality severely (HF detail down ~80% across 4 diverse prompts). And
the older Phase 5b claim that full-8-bit was "production quality" for
Lance_3B image is **also wrong** at this prompt complexity — same ~80%
degradation. Any usable Lance quant **requires** calibration.

## Critical constraint discovered

mlx-lm's `dwq_quantize` has a hardcoded `bits < 8` gate:

```python
def unfreeze(_, m):
    hasattr(m, "bits") and m.bits < 8
    m.unfreeze(keys=["scales", "biases"], recurse=False)
```

So an "8-bit DWQ" run is a no-op — 8-bit modules don't get trained. The
backlog plan for 5c-2 assumed DWQ would work at 8-bit; it doesn't with
the stock harness.

We therefore tested **naive 8-bit UND-only** as the simplest path: if
8-bit has enough precision (256 levels vs 4-bit's 16), maybe calibration
isn't needed. It isn't enough.

## Step 1 — prequant produced

`scripts/16_quantize.py --bits 8 --group-size 64 --skip-gen-tower`
- Source: `lance-mlx-models/Lance-3B-bf16` (12.37 GB)
- Output: `lance-mlx-models/Lance-3B-8bit-und/` (9.19 GB)
- 11.882 average bits/weight (mix of 8-bit UND + bf16 GEN)
- Time: 1.4 s after model load

## Step 2 — 4-prompt validation (catastrophic)

384², seed=42, 30 steps, CFG=4.0. Same prompts you'd want to ship against.

```
prompt           pix_mean_diff   pix_95p   HF bf16     HF q8       HF Δ%
P1_cat_stop      78.89          172.0      4.82e+08    8.20e+07    -83.0%
P2_dragon        35.52          101.0      3.45e+08    3.40e+07    -90.2%
P3_cat_skate     51.47          139.0      4.85e+08    9.57e+07    -80.2%
P4_cat_dog       50.45          127.0      4.18e+08    2.53e+07    -94.0%
```

Visual (`_compare_grid.png`):
- bf16 row: photographic, detailed, prompt-faithful across all 4 prompts
- 8bit-und row: cat/dragon/cat subjects mostly LOST; P4 is essentially
  pure color noise

## Step 3 — sanity check: full-8-bit vs UND-only vs 4-bit-und

The Phase 5b note claimed full-8-bit was "production quality" for
Lance_3B image; the HF README says it's broken. Tested all three
variants on P3 (the prompt 8bit-und did "least badly" on):

```
variant         HF       Δ vs bf16
bf16            4.85e+08    0.0%   (baseline — sharp cat on skateboard)
8bit-und        9.57e+07  -80.2%   (blurry cat, skateboard gone)
8bit-full       1.04e+08  -78.5%   (blurry cat, also degraded — Phase 5b note WRONG)
4bit-und        8.86e+07  -81.7%   (blurriest cat)
```

`_sanity_all_variants.png` shows all four side by side.

## Conclusions

1. **All naive quantization recipes produce ~80% HF degradation** on
   Lance_3B image generation. The "8-bit is fine for image" line in
   `notes/phase5b_quantization_findings.md` Recipe 1 was wrong (or
   tested against an easier prompt set). The HF "KNOWN BROKEN" tag on
   `mlx-community/Lance-3B-8bit` is correct.
2. **UND-only vs full quantization is roughly equivalent quality at the
   same bit width.** The Phase 5b reasoning that GEN bf16 would "save"
   image quality doesn't hold — text tokens with quantization noise
   contaminate latent tokens through shared attention regardless of
   whether GEN itself is quantized.
3. **8-bit is only modestly better than 4-bit at this naive recipe.**
   The 80%/82% HF drop difference is small — the precision floor for
   *naive* quantization is binary "broken vs not-broken" rather than a
   smooth quality-vs-bits curve. Calibration changes this; naive
   doesn't.

## What this means for Phase 5c

The "low-risk, high-reward" path 5c-2 doesn't exist. Naive 8-bit
quantization simply doesn't preserve Lance image quality at any
useful prompt complexity. Calibrated quantization is the **only** path
to a useful Lance quant variant.

## Recommended next moves

**Phase 5c-3 (AWQ port)** — port Reza2kn's AWQ alpha-search recipe to
MLX. Their pipeline successfully quantizes Lance to 4-bit AWQ-INT4 with
acceptable VQA quality. Algorithm is fully spec'd in BACKLOG.md ("from
`scripts/awq_apply.py` source, master branch"). MLX-native port is
~100 LOC since `mx.fast.quantized_matmul` provides the kernel; we'd
just write the alpha-search loop + scale-fusion. Higher initial
investment than Phase 5c-2 but it's the only validated path.

**Alternative — DWQ patch:** modify mlx-lm's `dwq_quantize` to drop
the `bits < 8` gate and enable DWQ training at 8-bit. Lower-effort
but unproven; would need its own validation cycle. Worth pursuing if
AWQ port hits a snag.

**Alternative — accept bf16-only for now:** Phase 5b deferred Reza2kn's
"first MLX 4-bit Lance with intact qk_norms" story for a reason — bf16
ships, works at production quality, and the quant gap is a
nice-to-have not a blocker. Pivot to image-side decode-frame
optimization (D1 spawned task), n_lat≥30k issue #1, or other backlog.

## Update needed elsewhere

- `notes/phase5b_quantization_findings.md` Recipe 1: revise "✅
  production quality" claim with this contradicting evidence.
- `BACKLOG.md` Phase 5c block: update with the mlx-lm bits<8 constraint
  and 5c-2 naive-8-bit empirical refutation.

## Artifacts

- `scripts/diagnostics/d_p5c2_validate.py` (4-prompt sweep)
- `scripts/diagnostics/d_p5c2_quick_full8bit_check.py` (variant sanity)
- `notes/phase5n_diagnostics/phase5c2_validation/_compare_grid.png` (4×2)
- `notes/phase5n_diagnostics/phase5c2_validation/_sanity_all_variants.png`
- Individual outputs `{bf16,8bit-und}_{P1,P2,P3,P4}.png`
- `lance-mlx-models/Lance-3B-8bit-und/` (the 8-bit prequant — keep
  for reference, don't ship)
