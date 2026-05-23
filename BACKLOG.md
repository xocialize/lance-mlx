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

**Recipe pinned (2026-05-23, after surveying Reza2kn's HF artifacts):**

*Understanding path (4-bit) — direct recipe transfer:*
```bash
mlx_lm.convert --hf-path Lance_3B_Video \
    --mlx-path Lance_3B_Video-MLX-4bit-prequant \
    -q --q-bits 4 --q-group-size 64
mlx_lm.dwq --model Lance_3B_Video \
    --quantized-model Lance_3B_Video-MLX-4bit-prequant \
    --mlx-path Lance_3B_Video-MLX-4bit-DWQ \
    --bits 4 --group-size 64 --num-samples 256
```
Reza2kn dropped `qk_norm` weights to fit mlx-lm's stock `qwen2` class. **Our
port carries qk_norms in `LanceMoTAttention` natively** — same DWQ recipe
should yield strictly better quality at the same bit budget. (Differentiator
for any future publish.)

*Generation path (4-bit) — needs activation-aware calibration:*
`Reza2kn/Lance-3B-Video-AWQ-INT4` (sibling repo, custom AWQ outside
mlx-lm) is the first public evidence the `_moe_gen` tower can be 4-bit
quantized coherently. Their recipe: AWQ alpha grid-search ∈ {0.0,…,1.0}
per fusion group, MSE-min against synthetic Gaussian, scale fused as
`norm.weight /= s; consumer.weight *= s`. group=128, 504 Linears (360
AWQ, 144 plain). Calibrated on 6 x2t_image + 11 t2i samples (108.5M
tokens). **Critical caveat:** they tested x2t_image only (5/6 oracle
correct) — never validated t2i / t2v / image_edit / video_edit at 4-bit.
GEN-path STRUCTURE survives; GEN-path GENERATION QUALITY is unknown.
Their PyTorch inference is ~10× slower than bf16 (per-forward dequant);
we sidestep this in MLX via `mx.fast.quantized_matmul`.

**Approach (updated):**
1. **Step 1 (low-risk, high-reward):** mlx-lm DWQ for Lance-3B understanding
   path (the recipe above) — validates the DWQ pipeline + ships a usable
   4-bit x2t_image variant. We win on qk_norm parity vs Reza2kn out of
   the gate.
2. **Step 2 (the interesting science):** mlx-lm DWQ on GEN tower. Use
   distillation against bf16 teacher with a per-tower split. If DWQ alone
   can recover GEN quality, we have the first MLX 4-bit Lance t2i.
3. **Step 3 (fallback if DWQ insufficient):** port Reza2kn's AWQ recipe
   to MLX. Their alpha-search + scale-fusion is library-agnostic. Use
   their calibration corpus (or generate our own t2i denoising samples)
   and validate against `tests/fixtures/results/t2i_sample_*`.
4. Mixed-precision options if uniform 4-bit fails: 4-bit UND + 8-bit GEN,
   or 8-bit UND + bf16 GEN.
5. **Always gate on bf16 oracle parity first** (optimize bf16, then quantize) —
   quantizing a still-buggy baseline wastes calibration effort.

**Open question:** Reza2kn's AWQ doesn't mention qk_norm handling at all.
For step 3, our preserved qk_norms might interact with the alpha-search
fusion logic in non-obvious ways. Worth a small isolated test before
committing.

**Benefit:** Lance-3B on 16 GB Macs (currently borderline-OOM in bf16),
and ~2-3× inference speedup. Significant user-base expansion to the
M1/M2/M3 8-16 GB segment. The current `mlx-community/Lance-3B-8bit` is
broken and a working quantized variant would close the regression. Bonus:
"first MLX Lance with intact qk_norms" is a real publish-worthy story if
step 1 lands.

**Trigger:** After bf16 baseline optimization is complete (L2-followup,
motion-direction audit, issue #1 pure-noise regime). Quantizing a
still-buggy bf16 baseline wastes the calibration effort.

**References:**
- `Reza2kn/Lance-3B-Video-und-MLX-4bit` — DWQ recipe template (UND only)
- `Reza2kn/Lance-3B-Video-AWQ-INT4` — first public Lance GEN quant (CUDA-only)
- `Reza2kn/Lance-3B-AWQ-INT4` / `Lance-3B-NVFP4` / `Lance-3B-Video-NVFP4` — sibling variants
- `github.com/Reza2kn/lance-quant` — reproduction toolkit

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
