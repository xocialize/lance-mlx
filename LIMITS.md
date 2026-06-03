# Honest limits — what to expect running Lance on your Mac

These optimizations (lossless streaming VAE decode, relay/deferred tower loading, a
mapped resolution×length frontier) are **real wins** — they're what makes a 3B dual-tower
image/video diffusion model run on consumer Apple-silicon RAM at all. But they do **not**
make the model small, and they hit hard walls. This page is the unvarnished version so you
aren't surprised. **We default to under-promising.**

There are **two VAE decode modes**, and the limits differ:

- **LOSSLESS** (`decode_streaming`, bit-identical to the reference decode) — the **default**
  for both image and video.
- **LOSSY** (`decode_tiled`, trapezoidal blend, ~1.5–4.8 / 255 off true pixels — imperceptible,
  but **not** bit-identical) — the **fallback** for the 768² video case where lossless exceeds RAM.

Toggle per call: `generate(..., lossless_decode=True|False)` (default `True`).

All memory numbers are **true process footprint** (`ri_phys` — what the OS actually commits, what
your RAM must hold), measured on a 16 GB M-series Mac, `lance-3b-video-bf16`, relay mode, 2026-06.
They are **~2× larger than older internally-quoted numbers**, which used MLX's `get_peak_memory()`
(an allocation counter that under-reports). We quote the honest, larger numbers. Tags:
**[M]** measured / **[P]** predicted (model under-predicts high res) / **[U]** unverified.

---

## "Generates" ≠ "losslessly viewable"

Two independent memory walls, which can land on opposite sides of your RAM:

1. **The denoise LOOP** makes the latent: ≈ `8.0 GB + 0.23 GB × (tokens/1000)` [P], `tokens = (H/16)·(W/16)·T_lat`.
2. **The VAE DECODE** turns the latent into pixels — and **at high resolution the lossless decode costs MORE than the loop.**

Example on 16 GB: **768²×25f** — LOOP fits (12.83 GB [M], you can generate it) but the LOSSLESS decode does **not** fit (it exceeds our 16 GB box; the only number is a watchdog abort at **>~21 GB** [M lower bound], true peak unmeasured); only the LOSSY decode (16.31 GB + ~3 GB swap [M]) fits 16 GB. **"My machine generated it" does not mean "my machine can losslessly decode it."**

---

## The 16 GB frontier — LOSSLESS vs LOSSY (all decode numbers [M], ri_phys)

| Output | LOSSLESS decode (bit-identical) | LOSSY decode (blend) | What to use on 16 GB |
|---|---|---|---|
| **256² video → 121 frames** | **8.05 GB** ✅ clean | 12.47 GB ✅ | **Lossless** — lighter *and* exact. Flat in length. |
| **512² video → 33 frames** | **12.55 GB** ✅ (+1.3 swap) | — | **Lossless** |
| **512² video → 61 frames** | **12.64 GB** ✅ (+2.3 swap) | **>20 GB ❌** | **Lossless — the *only* path that fits** (lossy blows up here) |
| **1024×1024 image** | **12.17 GB** ✅ clean | 12.53 GB ✅ | **Lossless** — ~same memory, exact. (Model's max image.) |
| **768² video, 13 frames** | **>~21 GB** ❌ (lower bound) | **13.17 GB** ✅ | Lossy on 16 GB; lossless: 24 GB+ [P] |
| **768² video, 25 frames** | **>~21 GB** ❌ (lower bound, peak unmeasured) | 16.31 GB (+3.1 swap) | Lossy on 16 GB; lossless: 24 GB+ [P] |

**The counterintuitive part:** *lossless is usually the LIGHTER path.* At 256²/512² the lossless
causal-cache streaming uses less RAM than the lossy blend (and is exact). **Lossy only earns its
keep at 768² video**, where it's the *only* mode that fits 16 GB — at the cost of blend artifacts.

End-to-end (loop + fresh-subprocess decode = max of the two): 256² ≈ 9.8 GB, 512² ≈ 12.6 GB
(decode-bound), 1024² image ≈ 12.2 GB, 768² video = decode-bound (lossless >~21 GB → exceeds 16 GB, 24 GB+ [P] / lossy 13.2–16.3 GB + ~3 swap → fits 16 GB).

---

## What the streaming decode actually buys (vs the naive whole `dec(z)`)

The honest baseline for "did this improve anything?" is **not** the lossy blend — it's the only
*other* bit-identical option, the plain whole `vae(z)`. Streaming is lighter than it in **every**
measured config, at zero pixel cost (all [M], ri_phys, same method; raw in
`results/decode_lossless/whole/`):

| config | whole `dec(z)` | streaming | win | mechanism |
|---|---|---|---|---|
| 256² × 49f  | 12.18 GB | **7.99 GB** | **−4.2 GB** | temporal streaming |
| 256² × 121f | 15.41 GB | **8.05 GB** | **−7.4 GB** | temporal streaming |
| 1024² image | 15.60 GB | **12.17 GB** | **−3.4 GB** | spatial halo-tiling |

Two mechanisms, both confirmed: **temporal streaming makes decode flat in length** (whole grows
12.18 → 15.41 across 49 → 121 frames; streaming barely moves, 7.99 → 8.05, so the win *widens*
with length), and **spatial halo-tiling wins even with no frames to stream** (1024² image, one
frame: 15.60 → 12.17). So the improvement is real for video *and* images — and it is the default
(`lossless_decode=True`), bit-identical (50 cases, `max|Δ|=0`). It does **not**, however, make
768² video fit 16 GB (the absolute peak there is still ~20 GB — see the frontier table).

---

## Per-RAM-tier expectations

| RAM | Loads? | Max IMAGE | Max VIDEO | Feel |
|---|---|---|---|---|
| **8 GB** | **NO** [P, firm] | — | — | Hard failure at **load** (relay transient ~7.5 GB + smallest video loop 8.24 GB each exceed usable RAM). **Do not attempt.** |
| **16 GB** | Yes [M] | 1024² **lossless** (~12.2 GB) | **256²→121f lossless** (~9.8 GB) · **512²→~61f lossless** (~12.6 GB, tight) · **768² → LOSSY ONLY** (13f 13.2 GB / 25f 16.3 GB + ~3 swap; lossless >~21 GB doesn't fit 16 GB) | Comfortable at 256². Tight at 512²/768² (decode-bound). 256²×121f gen ≈ **23 min** [M]. Free RAM before long/high-res runs. |
| **24 GB** | Yes | 1024² (model-capped) | 256²/512² lossless + **768²×25f LOSSLESS likely fits [P]** (decode >~21 GB lower bound; *not* measured on 24 GB — we have no such box) | Roomy; high-res video lossless decode is **inferred** to fit, unverified. |
| **32 GB** | Yes | 1024² (model-capped) | Same as 24 GB — **no larger/longer** | Extra headroom; output is already maxed by the model caps below. |
| **64 GB** | Yes [P] | 1024² (model-capped) | Same as 32 GB — **no larger/longer** | Surplus RAM wasted; output capped by the model. |

**Above ~24 GB, more RAM buys nothing for output size** — the lossless 768²×25f decode (>~21 GB lower bound) is *expected* [P] to fit 24 GB, and model caps bind (below). Note: we only have 16 GB hardware, so the 24 GB rows are inferences, not measurements.

---

## The harsh limits

1. **The loop's O(n) wall is fundamental and unfixed.** Loop memory grows with both resolution and frame count. These optimizations lowered the *base* (~8 GB), not the slope.

2. **High-res video DECODE exceeds the GENERATION wall, starting at 768²×13f (lossless).** You can generate clips your machine cannot *losslessly* decode. At 768², the lossless decode exceeds 16 GB for any length ≥13 frames — we only have a lower bound (>~21 GB, watchdog abort; true peak unmeasured, no 24 GB box); only lossy fits 16 GB there. A bigger-RAM machine (24 GB+) is inferred [P], not confirmed.

3. **Video decode is bit-identical by default** (`decode_streaming`), and at 256²/512² it is *also lighter* than the old lossy blend. The lossy `decode_tiled` blend is **only** the fallback for 768² video, where lossless decode exceeds 16 GB (see the frontier table) — and even then it differs by an imperceptible ~1.5–4.8 / 255, not a free pass. If you need bit-identical 768² video, you need a bigger-RAM machine than 16 GB (24 GB+ [P], inferred from the >~21 GB lower bound — the true peak is unmeasured) rather than the lossy fallback.

4. **Our older decode numbers were ~2× too low** — they used `mx.get_peak_memory()`. Everything here is true `ri_phys`. ("768² image decode 6.5 GB" was the under-count; honest figure is **12.2 GB**.)

5. **The loop model UNDER-PREDICTS at high resolution** (768²×25f measured 12.83 vs model 11.71, +10%). Treat 512²/768² *loop* numbers as optimistic; decode numbers above are measured.

6. **"Fits" ≠ "fast."** 768²×25f took **57 min** [M] on a box with ~9 GB free (compression + swap + weight re-paging → ~5× drag) vs ~15 min projected clean [P]. **Free your RAM before long/high-res runs.** Even the safe 256² path can lightly swap on a busy machine.

7. **bf16 only — no quantization escape hatch.** No wired quant path to trade precision for memory and clear the walls.

8. **All loop budgets assume flash-SDPA attention** (else O(n²); limits don't hold).

9. **Model caps bind before RAM above 16 GB:** **121 output frames** (31 latent) and **1024×1024** images (64×64 latent). No RAM exceeds these.

10. **Some levers were dead ends** (recorded so we don't claim them): temporal VAE tiling does **not** extend single-shot length; a deeper decode-tiling variant we evaluated *raised* the floor (not shipped); fast samplers aren't a free video speedup (video DPM is a different sample; `cfg_interval` ~3% @256², ~6–7% @768² [P]).

---

## Practical 16 GB cheat-sheet

- ✅ **Lossless, clean:** 256² video to 121 frames; 1024×1024 images.
- ⚠️ **Lossless, tight (light swap):** 512² video to ~61 frames — reboot/idle first. (Lossy does *not* fit here.)
- 🟡 **768² video: lossy only** (13f ~13.2 GB, 25f ~16.3 GB + ~3 swap). It generates, but you get blend-decoded pixels, not bit-identical. The lossless 768² decode doesn't fit 16 GB (>~21 GB lower bound, true peak unmeasured); for bit-identical 768² video you need a bigger-RAM machine (24 GB+ [P], unverified).
- **Rule of thumb:** the loop fitting tells you nothing about whether the *decode* fits. At 768²+ video, the decode is the wall — and only the lossy blend fits 16 GB.

*Measured numbers + method: `results/decode_lossless/DECODE_FOOTPRINT_SWEEP.md` (lossless + lossy + whole-`dec(z)` baseline sweeps, true ri_phys; whole-decode raw in `results/decode_lossless/whole/`).*
