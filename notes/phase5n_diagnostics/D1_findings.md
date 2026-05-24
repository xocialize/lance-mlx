# Phase 5n / D1 — VAE temporal-mode comparison

**Date:** 2026-05-24 (laptop / clean-room session)
**Goal:** Determine whether the shared Wan2.2 48-ch VAE decoder produces
materially different output for `T_latent=1` (t2i) vs `T_latent≥2` (t2v)
at the same first-frame latent content — and if so, whether that
asymmetry explains the video < image quality gap.

**Cost:** 5 s VAE-only load + ~5 s of forward passes. No LLM needed.

## TL;DR

- **The VAE has a code-path switch at `T_latent=1`.** Single-latent decode
  produces 3 output frames (not 1 per `(T-1)*4+1`); frame 0 has
  **3.6× less high-frequency detail and a measurable DC offset**
  compared to frame 0 of any `T_latent≥2` decode of the same first-frame
  latent content.
- **For `T_latent∈{2, 3, 5}` frame 0 is byte-identical.** Causal behavior
  confirmed: subsequent latent frames cannot affect frame 0 of the decode.
  So this is **not** "more video frames → better" — it's strictly **T=1
  vs T≥2 is two different code paths in the decoder**.
- **This eliminates the VAE as the cause of the video < image quality
  gap.** Video uses the "good" VAE regime (`T_latent≥2`); image uses
  the special `T=1` regime. If anything the VAE *favors* video.
- **Side finding worth flagging separately:** `t2i.py:286` takes
  `decoded[0, 0]` of the 3-frame `T_latent=1` decode. The 3 frames have
  visibly different statistics (frame 2 brighter and more detailed than
  frame 0). Worth a real-latent A/B on whether frame 0 is the right
  choice for image generation, independent of the video question.

## Data

### D1 — paired comparison, identical noise

```
seed=42, latent grid 24×24×48  (→ 384² output)

input shape         output shape         f0 mean   f0 std   f0 HF
(1, 1, 24, 24, 48)  (1,  3, 384,...)     +0.1100   0.2890   9.20e+05
(1, 5, 24, 24, 48)  (1, 17, 384,...)     +0.0627   0.5246   3.35e+06
```

Same first-frame latent content (sliced from the same noise tensor).
Different decoder output. Ratio of HF energy: T5 / T1 = **3.64×**.

### D1b — scan across `T_latent ∈ {1, 2, 3, 5}`

```
T_latent  T_decoded  f0 std   f0 FFT_HF   f0 mean
1         3          0.2890   9.20e+05    +0.1100   ← t2i regime
2         5          0.5246   3.35e+06    +0.0627
3         9          0.5246   3.35e+06    +0.0627
5         17         0.5246   3.35e+06    +0.0627   ← t2v regime
```

Frame 0 byte-identical across `T_latent ≥ 2`. Step change at `T=1`.

### Visual confirmation

Grid: `notes/phase5n_diagnostics/d1b_vae_frame_indexing/_grid_T1_vs_T5_noise_decode.png`

- `T_lat=1` frames 0, 1, 2: all blue/cyan-tinted, dim, low contrast
- `T_lat=5` frame 0: vivid, full color range, clear texture detail

The cyan cast is not "noise-textured Gaussian"; it's structurally
different output.

## Within-T=1 frame variation

t2i ships `decoded[0, 0]` but the 3 frames differ:

```
T_lat=1 frame 0:  mean +0.1100  std 0.2890  HF 9.20e+05   ← ships
T_lat=1 frame 1:  mean +0.3761  std 0.3623  HF 1.14e+06
T_lat=1 frame 2:  mean +0.4068  std 0.4349  HF 1.65e+06
```

None of them match `T_lat≥2` frame 0 (mean +0.063, std 0.525, HF 3.35e6).
Frame 2 of T=1 is closer in std/HF than frame 0, but still ~half the
detail of the T≥2 regime.

## Interpretation against user's symptom

User reports **image > video** quality. The VAE finding suggests the
**opposite structural bias**: VAE favors video. Therefore:

1. **VAE is not the cause of video < image.** Ruled out.
2. The image quality user perceives is good *despite* the VAE running in
   its degraded T=1 regime, presumably because:
   - Lance trained its image GEN tower against the T=1 decode behavior
     (so the LLM produces latents calibrated to this VAE regime).
   - Lance trained for image substantially more than for video, so the
     latents themselves are tighter for images.
3. The video gap lives **upstream of the VAE** — LLM-side (mrope,
   attention, position drift) or in CFG renorm.

## Action items

**Closes:**
- VAE-temporal-mode-asymmetry hypothesis for video < image gap. Ruled
  out as the cause.

**Opens (separately, lower priority — image-quality optimization, not
the user's current question):**
- A/B real-prompt t2i with `decoded[0, 0]` vs `decoded[0, 1]` vs
  `decoded[0, 2]`. If a later frame produces visibly sharper/better
  images, t2i.py:286 is leaving quality on the table.
- A/B real-prompt t2i with synthetic `T_latent=2` (append a copy of
  the latent or a noise frame) then take `decoded[0, 0]`. If this
  improves output, the VAE's "good" regime is reachable for images
  with a one-line decode-time tweak.

**Promotes (for video < image gap):**
- D2 (CFG renorm logger) — directly tests open hypothesis H3 from L2
  audit: "CFG renorm at higher token counts — not yet tested
  (~16 min t2v rerun)".
- D3 (mrope asymmetry) — t2i has unconditional +1000 t-axis shift,
  t2v has none. Never been A/B'd because t2i never exposed the knob.

## Scripts

- `scripts/diagnostics/d1_vae_temporal_mode.py` — paired T=1 vs T=5 test
- `scripts/diagnostics/d1b_vae_frame_indexing.py` — scan across T_latent
