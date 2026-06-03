# Whole `dec(z)` baseline vs streaming — the backing for PR #1's claim (2026-06-03)

Re-ran the **prior lossless option** (plain whole `vae(z)`, no streaming/tiling) at the
SAME shapes as the committed streaming sweep, same method (true `ri_phys`, REAL VAE weights,
random latent of the target shape, fresh subprocess). Tool: `scripts/_t2v_render_split.py
decode` with `V_RANDOM=1 V_WHOLE=1`. Box: 16 GB M-series, clean (~84% free). This is the
comparison that was previously UNcommitted (only in PR_PLAN.md).

## Measured (all true ri_phys, GB)

| config | shape | WHOLE `dec(z)` [NEW, M] | streaming (committed) | **streaming win vs whole** | swap (whole) |
|---|---|---|---|---|---|
| 256²×49f  | [1,13,16,16,48] | **12.18** | 7.99  | **−4.19** | 1.37 |
| 256²×121f | [1,31,16,16,48] | **15.41** | 8.05  | **−7.36** | 3.54 |
| 1024² image | [1,1,64,64,48] | **15.60** | 12.17 | **−3.43** | 2.39 |

## What it backs (definitively, measured both arms, same method)

1. **Streaming is lighter than the naive whole decode in EVERY measured config** (−3.4 to −7.4 GB).
   The "−7.4 GB" headline (256²×121f) reproduces exactly: whole **15.41** vs streaming **8.05**.

2. **Two independent mechanisms, both confirmed:**
   - **Temporal streaming → flat in length.** Whole GROWS with frames: 12.18 (49f) → 15.41 (121f),
     +3.23 GB over +72 frames. Streaming is flat: 7.99 → 8.05, +0.06 GB. The win WIDENS with length.
   - **Spatial halo-tiling → wins even with ZERO temporal extent.** 1024² single image: whole
     **15.60** vs streaming **12.17** (16 tiles) = −3.43 GB. No frames to stream; the win is pure
     spatial tiling.

3. **CORRECTION to an earlier claim:** I had said "no memory win at a single image." That was true
   only against the *lossy blend* (1024²: streaming 12.17 ≈ lossy 12.53). Against the *whole decode*
   (the naive baseline) streaming wins at 1024² image too, by 3.4 GB, via spatial tiling.

## Bit-identity (the correctness arm, already committed)
`tests/test_decode_stream.py` — 50 cases, `max|Δ| == 0.0` vs whole-seq `dec(z)`. So every GB
above is saved with zero pixel change.

## Raw per-run JSON
`results/decode_lossless/whole/whole_256x49.mp4.json`, `whole_256x121.mp4.json`,
`whole_1024img.mp4.json` (decode_peak_gb / swap / compressor / secs).
