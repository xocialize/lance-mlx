# Phase 5n / D5 + D6 — multi-frame bug localization

**Date:** 2026-05-24
**Combined goal:** find what specifically about t_lat>1 introduces the
quality / prompt-adherence degradation.

## D5 — num_frames sweep: CLIFF, not gradient

t_lat ∈ {1, 2, 3, 4} on the D4 cat-STOP-sign prompt, 384², seed=42.

```
t_lat  num_frames  n_lat   FFT_HF       Δ_vs_t1
1      1           576     3.92e+08      0.0%
2      5           1152    2.81e+08    -28.4%      ← big cliff
3      9           1728    2.87e+08    -26.8%      ← actually slight recovery
4      13          2304    2.41e+08    -38.5%
```

Visual: text on the STOP sign goes from "STO" English (t_lat=1) to
abstract/Asian-style glyphs (t_lat=2+) to fully illegible (t_lat=4).

**Verdict:** The damage is concentrated in the t_lat=1 → t_lat=2 step,
not in continued sequence growth. Cause class: "anything triggered by
having ANY second latent frame".

## D6 — position-IDs vs LPE disambiguation: RULES OUT POSITION

At t_lat=2 with surgical overrides of the t2v state:
- A: baseline (varying t-axis mrope + varying LPE per frame)
- B: flatten t-axis (mrope sees all latents at t=0)
- C: flatten LPE indices to f=0 (all latents get the LPE entries of frame 0)
- D: both flattened (frames effectively indistinguishable)

```
variant       flatten_t  flatten_f   FFT_HF       Δ_vs_A   recovery_to_t1
A_baseline    no         no          2.81e+08     +0.0%     0%
B_tflat       YES        no          3.34e+08    +18.9%    47.7%
C_lpe_fflat   no         YES         3.34e+08    +18.8%    47.6%
D_both_flat   YES        YES         3.07e+08    +9.3%     23.6%
```

**Surprises:**

1. **B ≈ C** (47.7% vs 47.6%) — flattening the t-axis OR flattening LPE
   gives the same numerical recovery. Both are equivalent ways to
   remove "frame distinction".

2. **D worse than B or C alone** — when BOTH signals are flattened, the
   model can no longer distinguish frames and the output regresses.
   The model needs SOMETHING to tell frames apart.

3. **Visual ground truth contradicts the numerical signal.** Even with
   B or C's +47% HF recovery, the text rendering on the STOP sign
   does NOT recover to English. Different degraded glyphs, but still
   degraded. The FFT_HF metric improves; the prompt-adherence (text)
   does not.

## What this rules in/out (cumulative through D6)

```
Ruled OUT as cause of multi-frame quality drop:
  - Weights diverging          (D4: Lance_3B_Video @ t_lat=1 ≈ Lance_3B)
  - VAE temporal mode          (D1: VAE favors video regime)
  - CFG renorm at scale        (D2: similar between pipelines)
  - mape_anchor                (D3: anchor sweep all equivalent)
  - Position-IDs at t > 0      (D6: flattening t-axis doesn't restore text)
  - LPE indexing at f > 0      (D6: flattening LPE doesn't restore text)

Ruled IN — narrower set of candidates:
  - Mask construction at the 2× larger latent block (most likely)
  - Attention behavior with larger bidirectional region (related)
  - Something else multi-frame-specific we haven't enumerated
```

## Next-test options

**D7 (cheap, ~3 min) — mask experiment:** at t_lat=2, override the
mask so each frame's latents only see their own frame's latents (no
cross-frame bidirectional attention). If text recovers → cross-frame
bidi mask is the bug. If not → mask isn't it.

**D8 (medium, ~5-10 min) — attention instrumentation:** log
attention-entropy per layer per step at t_lat=1 vs t_lat=2. Identifies
WHERE in the 36-layer stack the multi-frame behavior diverges from
single-frame.

**Stop and document:** narrowed enough that future work has a clean
starting point. Mark as a known foundation issue, pivot.

## Scripts

- `scripts/diagnostics/d5_num_frames_sweep.py`
- `scripts/diagnostics/d6_position_ids_vs_lpe.py`

## Per-variant artifacts

```
notes/phase5n_diagnostics/d5_num_frames_sweep/
  _compare_grid.png
  nf=1_tlat=1/  nf=5_tlat=2/  nf=9_tlat=3/  nf=13_tlat=4/

notes/phase5n_diagnostics/d6_position_ids_vs_lpe/
  _compare_grid.png
  A_baseline/  B_tflat/  C_lpe_fflat/  D_both_flat/
```
