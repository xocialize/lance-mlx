# Phase 5n / D2 — CFG renorm scale logger

**Date:** 2026-05-24
**Tests:** L2 audit open hypothesis H3 ("CFG renorm at higher token counts
— not yet tested").
**Result:** **REFUTED** at all tested scales. CFG renorm is not the cause
of the image-vs-video quality gap, and is not silently suppressing
t2v's prompt adherence at small-medium scales.

## Method

Reimplemented the Euler loop outside the pipeline (`scripts/diagnostics/
d2_cfg_renorm_logger.py`) to capture per-step CFG renormalization stats
without modifying production code. Both pipelines use the **channel**
renorm path (Phase 5m default): per-spatial-cell L2 norm, clamped at
`min(ratio, 1.0)`. Cost: ~3 min per pipeline including model load.

Logged per step: `norm_cond`, `norm_cfg`, `ratio = norm_cond / norm_cfg`,
`scale = clip(ratio, 0, 1)`, and `frac_clipped = (ratio < 1).mean()`.

## Headline data

```
label              n_lat   scale_avg   step-0   first_half   last_half   ratio_med
t2i 256×256          256       0.887     0.719        0.847       0.927       0.908
t2i 384×384          576       0.874     0.630        0.814       0.934       0.895
t2i 512×512         1024       0.888     0.596        0.809       0.968       0.903
t2v 256×256×9f       768       0.951     0.846        0.924       0.978       0.973
t2v 256×256×17f     1280       0.905     0.496        0.845       0.966       0.912
t2v 384×384×9f      1728       0.891     0.579        0.814       0.969       0.901
```

## Observations

1. **CFG renorm is not differentially suppressing t2v.** At comparable
   n_lat, t2v's `scale_avg` is at or above t2i's. At n_lat=768 t2v is
   0.951 vs t2i at n_lat=576 0.874. At n_lat=1728 t2v is 0.891 vs t2i
   at n_lat=1024 0.888.

2. **Both pipelines show the same temporal pattern.** Step 0 (high noise)
   sees heaviest clamping (~0.5-0.85 scale). Late denoising steps
   approach scale ≈ 1.0 (renorm largely passive). The CFG signal where
   it matters most (high-noise steps where prompt-following is set) is
   clamped *similarly* between pipelines.

3. **Mild downward trend with n_lat in t2v.** scale_avg goes from 0.951
   (n_lat=768) → 0.905 (n_lat=1280) → 0.891 (n_lat=1728). The drop is
   *decelerating*, suggesting a leveling-off near 0.88-0.89 rather than
   collapse. Extrapolating to production n_lat=11520 the scale would
   plausibly sit at 0.85-0.88 — non-trivial 12-15% velocity reduction
   but consistent with what the Phase 5m channel renorm was designed to
   permit.

4. **`frac_clipped` runs 0.7-0.9 throughout both pipelines.** Most cells
   experience *some* clamping at most steps. This is just how channel
   renorm works — it's a regularizer that's always slightly active, not
   a gate that "fires" or "doesn't fire".

## Verdict

**H3 (CFG renorm at scale → quality degradation) — REFUTED for n_lat ≤ 1728.**

The CFG renorm operates similarly across both pipelines and the gentle
downward trend with n_lat is consistent with intentional behavior, not
a silent suppression. Prompt-adherence degradation in t2v cannot be
explained by differential CFG renorm.

**Caveat for production scale:** We did NOT test t2v at production
n_lat (11520, i.e., 768²×17f). If H3 is going to re-emerge anywhere it
would be there. But the leveling-off pattern in the small/medium data
makes this unlikely to be the dominant cause.

## What this points to next

By elimination, the remaining live candidate for the t2i > t2v gap is:

- **mrope t-axis asymmetry** (D3 next). t2i unconditionally applies
  +1000 t-axis shift via `MAPE_ANCHOR_IMAGE_GEN`. t2v applies NOTHING
  (`mape_anchor=None` per Phase 5d). The two pipelines feed structurally
  different position-IDs to the same shared mrope kernel, and this has
  never been A/B'd directly because t2i never exposed the knob.

Other ruled-out:
- D1: VAE temporal-mode asymmetry — refuted, VAE favors video regime.
- D2: CFG renorm at scale — refuted as above.

## Open complementary work (lower priority)

- **Test t2v at production n_lat (11520) once a faster runner exists.**
  Would either confirm CFG renorm stays well-behaved at production scale
  or surface a late-emerging H3 effect. ~10-20 min runtime expected.
- **Test cfg_renorm_type="none" on t2v at production scale.** Would
  give an upper bound on how much (if anything) renorm is costing. Risk:
  unrenormalized CFG with scale=4.0 will likely blow up — but the test
  is informative even if it fails.

## Scripts and data

- `scripts/diagnostics/d2_cfg_renorm_logger.py` — the custom-loop logger
- `notes/phase5n_diagnostics/d2_cfg_renorm/_run_t2i.log` — full t2i trace
- `notes/phase5n_diagnostics/d2_cfg_renorm/_run_t2v.log` — full t2v trace
- `notes/phase5n_diagnostics/d2_cfg_renorm/summary.json` — machine-readable
