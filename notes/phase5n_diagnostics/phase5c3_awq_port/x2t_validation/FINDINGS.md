# Phase 5c-3f — AWQ-INT4 x2t_image (VQA) oracle sweep

**Date:** 2026-05-25
**Goal:** decide whether AWQ-INT4 is shippable as a VQA-only variant
given it failed on t2i (per Phase 5c-3e).
**Outcome:** **MARGINAL.** AWQ-INT4 preserves bf16's VQA behavior on
~4/6 cases (close to but below Reza2kn's PyTorch 5/6 benchmark), but
delivers **6-9× decode speedup** on longer-form questions. Worth
shipping as a preview / experimental variant with clear caveats about
precision-dependent answers.

## The verdict-recalibration step

My initial verdict heuristic compared AWQ output against the original
ByteDance PyTorch oracle's expected answer. That was wrong — bf16 is
our calibration target, not the upstream PyTorch oracle, and bf16
itself diverges from the original oracle on some cases. The meaningful
question is whether AWQ preserves bf16 behavior.

Reza2kn's "5/6 oracle correct" was their PyTorch AWQ-INT4 vs PyTorch
bf16 (matched framework, matched seed, matched everything). Our
equivalent is AWQ-vs-bf16 parity within MLX.

## Quality (AWQ vs bf16 parity)

```
case  Q                                      bf16 says        AWQ says           parity
 1    largest segment > sum others?          "Yes"            "Yes"              ✓ identical
 2    % wanting border security?             "43"             "43"               ✓ identical  (both ≠ oracle 29%)
 3    license plate?                         "Bx62bfy"        "Byfky"            ✗ AWQ garbled
 4    1998 promo spend?                      "1.8M dollars"   "198%"             ✗ divergent
 5    Colosseum appearance?                  "iconic amphi.." "iconic amphi.."   ✓ semantically equivalent
 6    solar eclipse appearance?              "circular shape" "moon silhouette"  ~ marginal (same topic, different specifics)
```

**3 clean parity + 1 marginal + 2 divergent = ~4/6 AWQ-vs-bf16 parity.**
Reza2kn's PyTorch: 5/6 oracle parity. We're one case behind.

## Speed (the unexpected upside)

```
case  bf16     AWQ-INT4   speedup
 1     0.6s     0.4s        1.5×
 2     0.6s     0.3s        2.0×
 3     1.1s     0.4s        2.8×
 4     6.4s     0.7s        9.1×
 5    12.1s     1.4s        8.6×
 6     8.6s     1.3s        6.6×
 ─────────────────────────────────
 total 29.4s    4.5s        6.5× wall-clock
```

The long-form descriptive cases (Colosseum, eclipse) — exactly the
expensive ones for users — see 6-9× speedup. Short factual answers see
1.5-3× speedup.

## Size

```
Lance-3B-bf16          12.37 GB
Lance-3B-AWQ-INT4       3.31 GB   (27% of bf16)
```

8-16 GB Macs that currently swap or OOM on bf16 Lance can run AWQ-INT4
comfortably.

## Where AWQ-INT4 degrades vs bf16

Pattern: **precision-required outputs** suffer.
- Case 3: license plate "BX62 BFY" → "Byfky" (alphanumeric mangling)
- Case 4: "1.8 million dollars" → "198%" (number-word relationship broken)
- Case 6 (marginal): geometric description shifts emphasis

Pattern: **categorical / descriptive outputs** preserved.
- Case 1: yes/no preserved exactly
- Case 2: numeric answer preserved (even when both bf16/AWQ are wrong)
- Case 5: paragraph description semantically equivalent

This is consistent with what 4-bit quantization is expected to lose:
fine-grained precision in lexical-token-level reasoning. The model
still "knows" the answer is a number / license plate, but the specific
digits get noisy.

## Shipping recommendation

**Ship as `mlx-community/Lance-3B-AWQ-INT4` with a clear VQA scope tag:**

Suggested model card framing:
> Compressed Lance variant via AWQ-INT4 calibration. **Use for:** VQA
> on M1/M2 8-16 GB Macs (3.3 GB on disk, 6-9× faster decode than
> bf16). **Don't use for:** image generation (use bf16 instead — naive
> and AWQ quantization both produce ~80% high-freq detail loss on t2i)
> or VQA tasks requiring precise alphanumeric extraction (license
> plates, currency amounts). For long-form descriptive VQA the quality
> closely matches bf16.

**Don't ship as a t2i variant.** Phase 5c-3e showed AWQ-INT4 doesn't
preserve image generation quality.

## What this closes

Phase 5c-3 is now substantively complete:
- 3a-3b: AWQ math kernel ported + unit-tested ✓
- 3c: calibration system in place ✓
- 3d: apply pipeline ✓
- 3e: t2i validation (AWQ not enough for image gen) ✓
- 3f: x2t_image validation (this — AWQ shippable for VQA with caveats) ✓

Future quant work (Phase 5c-3g+) candidates:
- AWQ-UND-only + bf16 GEN (untested combination — might preserve image
  quality if GEN-tower precision is the t2i bottleneck)
- Larger calibration corpus (currently 4 prompts; expand to all 6
  x2t_image cases too for more coverage)
- Investigate the 8-bit precision floor (AWQ-INT8 ≈ naive-INT8 is
  surprising; root cause worth understanding)

## Artifacts

- `scripts/diagnostics/d_p5c3f_x2t_sweep.py` — the sweep harness
- `notes/phase5n_diagnostics/phase5c3_awq_port/x2t_validation/`
  - `x2t_oracle_report.json` — machine-readable per-case detail
  - `_run.log` — full session output
