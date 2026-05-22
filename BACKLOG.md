# Deferred work — post-v0.5

Items deliberately deferred. Each carries a **Benefit** line so future
cost/benefit decisions can be made without re-deriving the rationale.

## L7 — PyPI release

**Status:** Deferred (2026-05-22, v0.5.1-polish).

**Blocker:** `pyproject.toml` pins specific git commits of `mlx-vlm` and
`mlx-video` (Qwen2.5-VL mRoPE has been churning upstream — multiple commits
per month). PyPI rejects packages with direct URL dependencies (PEP 508 +
PyPI policy).

**Unblock paths (whichever lands first):**
1. Blaizzy tags PyPI releases of `mlx-vlm` + `mlx-video` containing our
   pinned commits → just update pins to versions.
2. Validate our code against the latest PyPI releases of both — likely
   requires API-drift fixes given the upstream churn.
3. Vendor the small parts we use under `vendor/` (last-resort; high
   maintenance cost).

**Benefit:** `pip install lance-mlx` ergonomics for users — eliminates the
git-URL install step from quickstart docs. Lowers friction for community
adoption + makes the package discoverable on pypi.org. Cost is real but
contained to a deliberate bump-and-validate cycle.

**Trigger:** when a side-stream project needs Lance MLX as a clean
`pip install`able dep, or when upstream pins stabilize for ~2 weeks
without churn.

**Current install (works fine):**
```bash
pip install git+https://github.com/xocialize/lance-mlx@v0.5.1-polish
```

---

## Phase 5c — DWQ-calibrated quantization

**Status:** Deferred (2026-05-22, after L6 negative finding).

**Blocker:** Standard mlx-lm `quantize_model` (affine, group_size=64) destroys
Lance's MoE-gen tower quality. Reza2kn/lance-quant evidence + our own L6
test (`mlx-community/Lance-3B-8bit` now flagged KNOWN BROKEN) confirm Lance
needs per-tower calibration, not naive groupwise quantization.

**Approach:**
1. Per-tower calibration data (text + image latent activations for UND
   and GEN respectively)
2. DWQ (Dynamic Weight Quantization) per Reza2kn pattern: `group_size=64`
   uniform per-tower, calibrated against real activation statistics
3. Mixed-precision options: 4-bit UND + 8-bit GEN, or 8-bit UND + bf16 GEN
4. Re-validate against `tests/fixtures/results/t2i_sample_*` oracle on
   full bf16 baseline first (gate: optimize bf16, then quantize)

**Benefit:** Lance-3B on 16 GB Macs (currently borderline-OOM in bf16),
and ~2-3× inference speedup. Significant user-base expansion to the
M1/M2/M3 8-16 GB segment. The current `mlx-community/Lance-3B-8bit` is
broken and a working quantized variant would close the regression.

**Trigger:** After bf16 baseline optimization is complete (L2-followup,
motion-direction audit). Quantizing a still-buggy bf16 baseline wastes
the calibration effort.

---

## L2-followup — closer upstream-replica position-IDs

**Status:** Deferred until empirical test lands (L2-impl, in progress).

**Findings from L2 audit (notes/L2_upstream_position_ids_audit.md):**
- Our `_build_position_ids` uses sms=1 + base=0 (Phase 5j fix)
- Upstream `get_rope_index` uses sms=2 + base=text_len + st_idx tracking

**Open question:** Phase 5g V1 tested sms=2 + base=text_len at 256²×17f
and got "subject loss." But at 256² subjects are barely resolved even at
baseline; the test was inconclusive. Worth re-testing at 768²×17f where
subjects render cleanly.

**Benefit:** Closer-to-upstream port = closer chance of (a) addressing
the user-observed corner-cloud residual in some t2v outputs, (b)
addressing the motion-direction-at-short-clips observation, (c) making
the port more diff-friendly against upstream for future deep dives. Cost
is low — kwargs already plumbed; just needs a controlled empirical run.

**Trigger:** Currently in progress (L2-impl).

---

## Issue #1 — n_lat ceiling at 768²×≥17f

**Status:** Deferred separately from issue #2 (closed).

**Symptom:** t2v at n_lat ≥ ~11,520 (768²×17f+) shows partial degradation;
n_lat = 29,952 (768²×50f, Lance reference scale) is pure noise.

**Phase 5j fix did NOT address this** — different bug class (n_lat
ceiling, not the watercolor / position-ID drift).

**Benefit:** Would unlock the full Lance reference scale (768²×50f, ~4s
clips) — currently capped at 768²×13f (~1s) on production. Likely the
single biggest user-visible win in the t2v polish track.

**Trigger:** Post-L2-followup + post-DWQ. Needs deeper investigation
than a one-line fix.

---

## x2t_video — full 6/6 oracle sweep

**Status:** Deferred. 2/6 cases validated (verbatim + content-correct).

**Missing locally:** vqa-01 (counting), vqa-02 (repeated-actions),
vqa-04 (time-manipulation narrative), caption-long-01 (butterfly+bee).

**Benefit:** Completeness — would close the Phase 0 oracle suite for
video understanding. Likely no surprises given the 2/6 already match
(VQA verbatim, captioning content-correct).

**Trigger:** If we need a complete oracle pass for a paper/writeup.
