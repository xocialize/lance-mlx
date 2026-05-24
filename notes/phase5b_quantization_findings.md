# Phase 5b — quantization findings (2026-05-21)

## TL;DR

**Shipped:** `mlx-community/Lance-3B-8bit` — production-quality 8-bit image quantization. 2.7× faster than bf16 at 768² t2i, 53% memory footprint, photorealism preserved.

**Not shipped (require DWQ or finer recipes):**
- 4-bit Lance_3B → GEN path collapse (cat subject disappears in t2i output)
- 8-bit Lance_3B_Video → t2v output is gray gradient regardless of skip rules / group size
- 4-bit Lance_3B_Video → would be worse than 8-bit (not tested)

## What we tried (in order)

### Recipe 1: 8-bit affine, group_size=64, full quant (UND + GEN towers + embed + lm_head)
- ⚠️ **Lance_3B (image specialist):** original assessment ("works at production quality, slight text degradation") was **revised on 2026-05-24 by Phase 5c-2** — under a 4-prompt diagnostic sweep at 384², full-8-bit produces ~78% FFT high-freq detail loss; subjects are recognizable but visibly blurred, fine text rendering collapses. The HF `mlx-community/Lance-3B-8bit` "KNOWN BROKEN" tag is correct. UND-only 8-bit (Phase 5c-2's variant) is similarly broken (~80% HF loss). Any naive groupwise quantization recipe — at any tested bit-width — fails on Lance image generation. See `notes/phase5n_diagnostics/phase5c2_validation/FINDINGS.md`.
- ❌ **Lance_3B_Video (video specialist):** t2v at 256² × 17f produces gray-gradient noise instead of the recognizable balls-on-wooden-table the bf16 baseline produces at this seed/prompt.

### Recipe 2: 8-bit, skip the entire `_moe_gen` tower (GEN expert at bf16)
- ❌ Lance_3B_Video: still gray-gradient on t2v. Footprint 75% of bf16 (from 53%), no quality recovery.

### Recipe 3: 8-bit, group_size=32, skip GEN tower
- ❌ Lance_3B_Video: still gray-gradient. Finer granularity didn't help.

### Recipe 4: 8-bit, no-CFG diagnostic
- ❌ Lance_3B_Video: still gray-gradient even with cfg_scale=1.0. So the failure isn't CFG amplification.

### Recipe 5: 4-bit affine, group_size=64 on Lance_3B
- ❌ Lance_3B t2i: cat subject disappears entirely; only a yellow+red horizontal block (the colorful sign) survives. Matches Reza2kn/lance-quant's reported "naive INT4 produces gibberish on Lance gen path" finding.

## Why Lance_3B works at 8-bit but Lance_3B_Video doesn't

Per Phase 4c per-tensor diff: Lance_3B and Lance_3B_Video are **separately fine-tuned** checkpoints, not just LPE-size variants. `_moe_gen` QK-norms differ by 0.5–0.85 in 6+ layers; many GEN-tower weights differ. Lance_3B_Video has different weight distributions that affine 8-bit quantization can't approximate well — at least with this skip-list-only recipe.

Even with the GEN tower preserved at bf16 (Recipe 2), the UND tower + embed_tokens quantization is enough to break video t2v output for Lance_3B_Video. Lance_3B handles the same operations fine, presumably because its weight distributions are more affine-quantization-friendly.

This is consistent with Reza2kn's finding that the Lance GEN path needs **DWQ (dynamic weight quantization)** — calibration-based, per-tensor scaling that captures distribution-specific magnitudes — to survive INT4. The same likely applies to Lance_3B_Video at INT8.

## What ships in mlx-community

| Repo | Status |
|---|---|
| [`mlx-community/Lance-3B-bf16`](https://huggingface.co/mlx-community/Lance-3B-bf16) | 🟢 Production (full quality, ~12 GB) |
| [`mlx-community/Lance-3B-8bit`](https://huggingface.co/mlx-community/Lance-3B-8bit) | 🟢 Production (2.7× faster, 6.6 GB) |
| [`mlx-community/Lance-3B-Video-bf16`](https://huggingface.co/mlx-community/Lance-3B-Video-bf16) | 🟢 Functional (painterly aesthetic, ~13 GB) |
| `mlx-community/Lance-3B-Video-8bit` | ⏳ Phase 5c (needs DWQ) |
| `mlx-community/Lance-3B-4bit` | ⏳ Phase 5c (needs DWQ) |
| [`mlx-community/Wan2.2-VAE-Lance-bf16`](https://huggingface.co/mlx-community/Wan2.2-VAE-Lance-bf16) | 🟢 Production (standalone, ~1.4 GB) |

## Memory math (M5 Max 128 GB → 16/32 GB Macs)

| Variant | Disk | Runtime RAM (approx) | Fits comfortably on |
|---|---|---|---|
| Lance-3B-bf16 | 15.1 GB | 17–22 GB | 32 GB+ |
| Lance-3B-8bit | 9.3 GB | 10–13 GB | **16 GB+** |
| Lance-3B-Video-bf16 | 15.6 GB | 18–24 GB | 32 GB+ |

The 8-bit image variant opens Lance image generation to 16 GB Macs — the headline accessibility win.

## Implementation

- `scripts/16_quantize.py` — CLI for quantization with `--bits`, `--group-size`, `--skip-gen-tower` flags.
- `src/lance_mlx/model/_loader.py` — centralized `load_lance_model()` that detects the `quantization` block in `config.json` and applies `nn.quantize` before `load_weights`. All 5 pipelines (t2i, t2v, image_edit, video_edit, understanding) use this loader, so any quantized variant works through the same APIs as bf16.

## Deferred to Phase 5c

1. **DWQ (Dynamic Weight Quantization)** for video specialist + 4-bit image.
   - Use a small calibration dataset (a few prompts each from t2i/t2v).
   - Compute per-tensor optimal scales rather than affine groupwise.
   - Reza2kn's `quantize_dwq_lance.py` is the closest reference; their PyTorch v2 hit 50% byte-match at group_size 64.
   - **Benefit:** 4-bit image variant (~3.5 GB, runs on 8 GB Macs); 8-bit video variant (~7 GB runtime).

2. **NVFP4 / MXFP4 mode** investigation. mlx-lm supports these but they're new and may not have the dynamic range needed for Lance.
   - **Benefit:** alternative path to 4-bit if DWQ doesn't pan out.

3. **Per-layer bit-width** (mixed 8-bit/bf16). Could let critical layers stay at bf16 even within the UND tower.
   - **Benefit:** finer-grained quality/size tradeoff.
