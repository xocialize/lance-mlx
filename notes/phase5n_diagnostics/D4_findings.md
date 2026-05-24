# Phase 5n / D4 — pipeline-code vs weights isolation

**Date:** 2026-05-24
**Tests:** is the image-vs-video gap caused by (a) multi-frame code,
(b) Lance_3B_Video weights diverging from Lance_3B, or (c)
training-data imbalance manifesting at t_lat>1?

**Result:** **Multi-frame code path is the cause.** Lance_3B_Video at
t_lat=1 produces image-comparable output; degradation appears
specifically at t_lat>1. This **refutes the training-data hypothesis**
as the dominant cause and **localizes the bug to multi-frame code**.

## Method

Three runs, same prompt (38-word t2i oracle cat-STOP-poster), same
seed, same scale (384²), 30 Euler steps:

- **A.** Lance_3B + t2i pipeline (production image baseline)
- **B.** Lance_3B_Video + t2v pipeline at num_frames=1 (t_lat=1)
- **C.** Lance_3B_Video + t2v pipeline at num_frames=9 (t_lat=3,
  production-like multi-frame)

## Visual verdict

`notes/phase5n_diagnostics/d4_pipeline_isolation/_compare_grid.png`

| Panel | Subject | Sign | Letters | Garden | Quality |
|-------|---------|------|---------|--------|---------|
| A (t2i)               | cat ✓ | red ✓ | clear English "STO" ✓ | ✓ | Photographic |
| B frame 0 (t2v 1f)    | cat ✓ | red ✓ | slightly distorted but recognizable | ✓ | Photographic |
| B last (t2v 1f)       | cat ✓ | red ✓ | same as frame 0 | ✓ | Photographic |
| C mid (t2v 9f)        | cat ✓ | red ✓ | **illegible Asian-style glyphs** | ✓ | Slightly painterly |

**Critical visual finding:** t2v at t_lat=1 (panel B) produces output
visually comparable to t2i. The English text on the sign is preserved
(with minor RNG-path variation). At t_lat=3 (panel C), text rendering
collapses from English letters to abstract glyphs — exactly matching the
user's "prompt adherence is lower for video" symptom.

## Numerical verdict

Pixel-level mean absolute difference vs panel A (t2i baseline):

```
A vs B (t2v 1f, frame 0):     mean diff = 55.21  (95p = 159.0)
A vs B (t2v 1f, last frame):  mean diff = 51.09  (95p = 149.0)
A vs C (t2v 9f, mid frame):   mean diff = 69.82  (95p = 180.0)
B vs C:                       mean diff = 62.62  (95p = 181.0)
```

Going from t_lat=1 to t_lat=3 increases divergence from t2i baseline
by ~30%, and B→C alone adds 63 units of mean per-pixel difference.
The damage is real and concentrated in the multi-frame step.

## What this rules in and out

**Rules OUT:**
- **Training-data imbalance.** If video weights had been undertrained
  relative to image weights, they'd produce degraded output at
  *any* t_lat, including t_lat=1. They don't. The weights are
  semantically equivalent at the equivalent compute regime.
- **VAE temporal-mode bias hurting image.** D1 already refuted; D4
  reconfirms (t_lat=1 path matches t2i, both go through the same
  T_decoded=3 VAE regime).
- **Bug in t2v's general control flow.** If it were broken at t_lat=1
  too, B would show it. It doesn't.

**Rules IN — narrow candidate list:**
1. **Position-IDs at t > 0** — only with t_lat>1 do latent tokens get
   varying t-axis values. At t_lat=1 every latent token has t=0
   (post-base-offset), trivially mrope-compatible with t2i. The
   variation only emerges at t_lat>1.
2. **Mask construction at larger latent blocks.** The bidirectional-
   within-latent-block region grows quadratically. At 384²×9f the
   latent block is 1728 tokens (3× the t_lat=1 case). May degrade
   attention quality.
3. **LPE indexing into f>0 entries.** Phase L2's H2 audit confirmed
   the LPE is sinusoidal (uniform stats across frame slots), so this
   is the least likely. But worth ruling out — the indexing math at
   `f * 64² + r * 64 + c` puts t>0 indices well above 4096 (the
   image-LPE max), so any sinusoidal-vs-trained mismatch at higher
   indices could matter.
4. **Attention dilution** at the longer sequence. With 3× more tokens
   per query, softmax becomes flatter. Could reduce semantic
   precision (e.g., the ability to attend specifically to text tokens
   when rendering letters).

## Implications

This is a **much better outcome than the training-data answer** because
it's actionable. The bug is in code, not in data. The remaining work
is to bisect *which* of the multi-frame-specific differences causes
the damage.

The most diagnostic next test (D5 candidate): run t2v with
`num_frames` ∈ {1, 5, 9, 13} on the same prompt+seed. If there's a
sharp cliff at t_lat=2 (num_frames=5), the bug is purely "anything
that introduces a t>0 axis"; if there's a gradient with t_lat, it's
more likely sequence-length-related (attention dilution / mask growth).

## Scripts and data

- `scripts/diagnostics/d4_pipeline_isolation.py`
- `notes/phase5n_diagnostics/d4_pipeline_isolation/_compare_grid.png`
- Individual decoded frames per variant under same dir
