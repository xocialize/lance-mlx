# Lossless decode footprint sweep — TRUE ri_phys (2026-06-03)

Measures the **real process footprint** (`ri_phys_footprint` via `proc_pid_rusage`) of the
shipped lossless `decode_streaming` path, per resolution × length. Method: REAL VAE decoder
weights (`vae.safetensors`) + a random latent of the target shape (decode memory is
shape-determined, value-independent) + the production decode call
(`decode_streaming(z, chunk_lat=1, spatial_tiles=suggest_spatial_tiles(...))`), each in a
FRESH subprocess (so the MLX pool starts empty — no inherited loop pool). Tool:
`scripts/_t2v_render_split.py decode` with `V_RANDOM=1`. Box: 16 GB M-series, clean (~13 GB
reclaimable). Watchdog aborts >16 GB RAM + 5 GB swap (= "EXCEEDS ~21 GB").

## Why this supersedes the old decode numbers
The previously-quoted decode peaks (e.g. "768² image decode 6.47 GB", "256²×121f decode
6.71 GB") were `mx.get_peak_memory()` — MLX's internal allocation counter, which
**under-reports the true footprint by ~2×** (it omits Metal command buffers + the
allocator's reserved pool). Calibration: 768² image decode = **12.17 GB** ri_phys vs the old
6.47 GB mx number. The numbers below are what the OS actually commits — what a user's RAM
must hold.

## Results

| config | res | output frames | T_lat | n_tiles | **decode ri_phys** | swap used | verdict |
|---|---|---|---|---|---|---|---|
| image | 1024² | 1 | 1 | 16 | **12.17 GB** | 0.14 | FITS (clean) |
| video | 256² | 49 | 13 | 4 | **7.99 GB** | 0.0 | FITS (clean) |
| video | 256² | 121 | 31 | 4 | **8.05 GB** | 0.0 | FITS (clean) — flat in length |
| video | 512² | 33 | 9 | 4 | **12.55 GB** | 1.31 | FITS (light swap) |
| video | 512² | 61 | 16 | 4 | **12.64 GB** | 2.33 | FITS (moderate swap, near edge) |
| video | 768² | 13 | 4 | 4 | **> ~21 GB** | — | **EXCEEDS 16 GB** |
| video | 768² | 25 | 7 | 4 | **> ~21 GB** | — | **EXCEEDS 16 GB** |

## What it means
- **256² video decode is light (~8 GB) and FLAT in length** (49f ≈ 121f ≈ 8 GB) — the temporal
  causal-cache streaming works exactly as designed. Lossless at any length up to the 121-frame
  model cap fits 16 GB with room.
- **512² video decode ~12.6 GB** — fits, but on the edge: it needs 1.3–2.3 GB of swap even on a
  clean box, and is near-flat in length (33f ≈ 61f). Call it "fits, may lightly swap."
- **1024² image decode = 12.17 GB** — fits clean. (Supersedes the old "14.26 GB" free-mode/pre-relay
  figure and the "6.47 GB" mx under-count.)
- **768² VIDEO decode EXCEEDS ~21 GB at *any* length ≥ 13 frames** — the lossless decode does NOT
  fit 16 GB. This includes 768²×13f, the historically "single-node-validated" size (which was
  validated for the LOOP and with the OLD lossy decode, not lossless). The 768² wall is the
  *decode*, and it begins at 13 frames.

## End-to-end (what a 16 GB machine actually needs)
With the production fresh-subprocess decode (t2v_long path), end-to-end peak = max(loop, decode):

| config | loop ri_phys | decode ri_phys | end-to-end (max) | 16 GB lossless? |
|---|---|---|---|---|
| 256²×49f | 8.63 (measured) | 7.99 | ~8.6 GB | ✅ clean |
| 256²×121f | 9.80 (measured) | 8.05 | ~9.8 GB | ✅ clean |
| 512²×33f | ~10 (predicted) | 12.55 | ~12.6 GB | ✅ tight (decode-bound, +1.3 swap) |
| 512²×61f | ~11.7 (predicted) | 12.64 | ~12.6 GB | ⚠️ tight (decode-bound, +2.3 swap) |
| 768²×13f | ~10 (predicted) | **>~21 (lower bound)** | **>16 GB** | 16 GB: no · 24 GB+ [P] |
| 768²×25f | 12.83 (measured) | **>~21 (lower bound)** | **>16 GB** | 16 GB: no · 24 GB+ [P] |
| 1024² image | — | 12.17 | ~12.2 GB | ✅ clean |

For 256²/512² video and 768²+ images the **loop and decode are comparable**; for 768² VIDEO the
**decode is the hard wall**. The 768² lossless decode does **not** fit 16 GB — the only
measurement is the watchdog abort above (**>~21 GB**, a lower bound; the exact peak is
**unmeasured**, since it exceeds our 16 GB test hardware and we have no 24 GB box). So "needs a
24 GB-class machine" is an **inference [P]** from the >~21 GB footprint, not a measured fact. (An
earlier draft of this doc quoted "20.29 GB [M]" here — that figure was never backed by a raw run
and has been dropped.)

---

## LOSSY decode_tiled (blend) footprint — same method, ri_phys (added 2026-06-03)

The lossy path is `Wan22VAEDecoder.decode_tiled` (trapezoidal-blend tiling, ~1.5–4.8/255 off
true pixels). It is the fallback for the 768² video case where lossless exceeds RAM. Measured the same way:

| config | res | frames | T_lat | **LOSSY decode ri_phys** | swap | vs LOSSLESS |
|---|---|---|---|---|---|---|
| image | 1024² | 1 | 1 | 12.53 GB | 0.29 | ≈ lossless (12.17) |
| video | 256² | 121 | 31 | **12.47 GB** | 0.0 | **WORSE** than lossless (8.05) |
| video | 512² | 61 | 16 | **EXCEEDS ~20 GB** | — | **WORSE** — lossless (12.64) fits, lossy does NOT |
| video | 768² | 13 | 4 | **13.17 GB** | 0.0 | **BETTER** — lossy fits, lossless (>21) does not |
| video | 768² | 25 | 7 | 16.31 GB | 3.13 | **BETTER** — lossy fits (tight), lossless (>21) does not |

**Key correction to the intuition "lossless costs more":** at 256²/512² the lossless
causal-cache streaming is *lighter AND exact* — the lossy blend tiling holds more (it
temporal-tiles+blends rather than streaming). Lossy only wins at **768²**, where it's the
only path that fits 16 GB. So the best 16 GB strategy is **lossless everywhere it fits
(256²/512² video, all images), lossy only at 768² video** — which is exactly what the
default `decode_streaming` does, falling back to lossy `decode_tiled` only where lossless
exceeds RAM.

---

## WHOLE `dec(z)` baseline — what streaming improves ON (added 2026-06-03)

The lossy comparison above answers "is the new default lighter than #6's blend?". A separate
question is "did streaming improve on the only *other* bit-identical option — the naive whole
`vae(z)`?" Re-ran whole `dec(z)` (no streaming, no tiling) at matching shapes, same method
(`V_RANDOM=1 V_WHOLE=1`, true ri_phys, fresh subprocess). Raw JSON in `whole/`.

| config | res | frames | T_lat | **whole `dec(z)`** | streaming | **win** |
|---|---|---|---|---|---|---|
| v256x49  | 256²  | 49  | 13 | 12.18 GB | 7.99 GB  | **−4.19** |
| v256x121 | 256²  | 121 | 31 | 15.41 GB | 8.05 GB  | **−7.36** |
| img1024  | 1024² | 1   | 1  | 15.60 GB | 12.17 GB | **−3.43** |

**Streaming wins in every measured config**, via two mechanisms: temporal streaming makes the
video decode *flat in length* (whole 12.18 → 15.41 over 49 → 121 frames; streaming 7.99 → 8.05,
so the win widens with length), and spatial halo-tiling wins even on the single 1024² image with
no temporal extent (15.60 → 12.17). Bit-identical throughout (`test_decode_stream.py`, 50 cases,
`max|Δ|=0`). This is the backing for the "−7.4 GB at 256²×121f" figure; it is the *whole-decode*
baseline, not a guessed "32 GB" (which was never measured and has been dropped).
