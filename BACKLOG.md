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

## Phase 5c — Calibrated quantization (RESEARCH-CLOSED, 2026-05-26)

**🎓 Final status: closed as research effort.** The full arc through
5c-1 (DWQ), 5c-2 (naive), 5c-3a-h (AWQ + investigation) concluded
that Lance image generation cannot be quantized below bf16 to
production quality with currently available MLX quantization
primitives. The 80% HF floor is architectural (per-step error
compounding across 2,160 forward-pass evaluations per image),
not algorithmic — AWQ's math works correctly per-Linear (Phase 5c-3h
empirical confirmation) but per-layer gains don't compound through
the flow-matching integrator. **No active development planned.**

Shipping outcome:
- `mlx-community/Lance-3B-bf16` ships for t2i / image_edit / x2t_image
- `mlx-community/Lance-3B-AWQ-INT4` ships for x2t_image VQA only
  (3.31 GB LLM, 6-9× faster decode, ~4/6 oracle parity with bf16)
- `mlx-community/Lance-3B-8bit` superseded by AWQ-INT4 (kept for
  historical reproducibility)

Code artifacts (kept for future use if upstream MLX gains new quant
primitives or if downstream `mlxEngine` work picks up):
- `src/lance_mlx/quant/awq.py` — AWQ scale-search + scale fusion
- `src/lance_mlx/quant/calibrate.py` — ActStats hook system
- `scripts/quant/calibrate_awq.py` + `apply_awq_quantize.py` + `publish_awq_int4.py`

Forward pointer: `notes/mlx_engine_quant_notes.md` (Lance-specific
constraints + speculative paths if quant becomes important again).

Below is the chronological investigation record for reference.

---

**Phase 5c-1 attempted (2026-05-23)** — 4-bit UND-only DWQ
produces 1-of-4 acceptable outputs on the diagnostic sweep, NOT shipped.
**Phase 5c-2 attempted (2026-05-24)** — naive 8-bit UND-only fails
catastrophically (~80% HF detail loss across 4-prompt sweep). Also
discovered mlx-lm's `dwq_quantize` has a hardcoded `bits < 8` gate, so
"8-bit DWQ" with the stock harness is a no-op. Full writeup:
`notes/phase5n_diagnostics/phase5c2_validation/FINDINGS.md`.
**Phase 5c-3 COMPLETED + SHIPPED (2026-05-26)** — full AWQ port to MLX,
end-to-end validated, and published as `mlx-community/Lance-3B-AWQ-INT4`
(VQA-scoped variant; 5.65 GB repo, 3.31 GB LLM). All sub-phases 3a-3g
delivered:

- 3a/3b: AWQ math kernel ported + unit-tested (+51% output-error reduction)
- 3c: calibration system (ActStats hook → 504/504 module coverage)
- 3d: apply pipeline (AWQ scale-fusion + nn.quantize); produces
  Lance-3B-AWQ-INT4 (3.31 GB, 27% of bf16) in ~15s
- 3e: t2i validation — REFUTED. AWQ-INT4 still has ~-80% HF detail loss
  on image gen, no improvement at INT8. **Bf16 remains only production
  t2i variant.**
- 3f: x2t_image (VQA) validation — **MARGINAL SHIPPABLE.** AWQ-INT4
  preserves ~4/6 cases vs bf16 with **6-9× decode speedup**. Caveats:
  precision-required outputs degrade (license plates, currency, exact
  numbers); long-form descriptive VQA closely matches bf16.
- 3g: AWQ-UND-only experiment — REFUTED. Identical VQA quality to
  AWQ full quant, 2.2× larger, 3-4× slower long decodes. No
  shipping case; AWQ full quant is strictly better. See
  `phase5c3_awq_port/PHASE_5C3G_FINDINGS.md`.

**Shipped artifact: [`mlx-community/Lance-3B-AWQ-INT4`](https://huggingface.co/mlx-community/Lance-3B-AWQ-INT4)**
(3.31 GB LLM; full repo 5.65 GB incl. VAE + ViT). For VQA on 8-16 GB
Macs. Honest VQA-scoped model card with caveats. Full writeup:
`notes/phase5n_diagnostics/phase5c3_awq_port/PHASE_5C3_COMPLETE.md`
and `.../x2t_validation/FINDINGS.md`.

Production code: `src/lance_mlx/quant/{awq,calibrate}.py`, CLI tools
under `scripts/quant/`.

Surprising finding worth investigating later: AWQ-INT8 ≈ naive 8bit-und
quality. At 8-bit the quantization scheme isn't the bottleneck — some
other systematic source imposes ~80% HF floor regardless of calibration.

**Phase 5c-1 empirical result (4-bit UND + bf16 GEN + DWQ):**
- Script: `scripts/17_dwq_und_4bit.py` (mlx-lm DWQ wrapped around
  LanceTextLogitsWrapper for text-only forward).
- Recipe: 256 samples, batch=4, LR=1e-6, Adam(bias_correction=True),
  temperature=2.0, 64 steps. Final val loss 0.161 vs initial 0.192.
- 4-prompt sweep verdict: 🟢 panda portrait, ⚠️ landscape, ❌ dragon+
  castle (loses castle), ❌ cat+"STOP" poster (text lost).
- Root cause: UND-tower QKV at int4 corrupts image generation through
  *shared* attention (text tokens cross-attend with latent tokens
  → corrupted text-side projections poison the SDP). Even with GEN at
  bf16, UND-only quantization isn't sufficient for general t2i.
- NOT pushed to mlx-community. Artifact at
  `/tmp/lance_phase5c/Lance-3B-4bit-und-DWQ/`.

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
Reza2kn drops `qk_norm` only in `extract_und_to_qwen.py` (the UND-only
repackaging step that forces Lance into stock mlx-lm's `qwen2` class).
Their AWQ pipeline ITSELF preserves qk_norms (the AWQ `FUSION_GROUPS`
touch `input_layernorm` + `post_attention_layernorm` only, not the qk
norms inside attention). So no qk_norm-parity advantage for us at the
AWQ stage — both pipelines are equivalent here.

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

**Algorithm pinned (from `scripts/awq_apply.py` source, master branch):**

```python
# Per fusion group:
#   group ∈ {(input_layernorm, [q_proj, k_proj, v_proj]),
#            (input_layernorm_moe_gen, [q/k/v_proj_moe_gen]),
#            (post_attention_layernorm, [mlp.gate_proj, mlp.up_proj]),
#            (post_attention_layernorm_moe_gen, [mlp_moe_gen.gate_proj, .up_proj])}
#
# 1. act_mean = per-channel mean(|activations|), averaged across consumers
# 2. w_max   = per-channel max(|weight|), averaged across consumers
# 3. for alpha in {0/20, 1/20, ..., 20/20}:
#        s = (act_mean^alpha / w_max^(1-alpha)).clamp(min=1e-5)
#        s = s / sqrt(s.max() * s.min())                 # geomean ≈ 1
#        x = randn(512, in_features) * act_mean          # synthetic input
#        for w in consumers:
#            w_scaled    = w * s.unsqueeze(0)
#            w_dequant   = quant->dequant per-group asym INT4(w_scaled)
#            err        += mean((x/s @ w_dequant.T - x @ w.T)^2)
#        track best alpha by err
# 4. norm.weight        /= s                             # absorb into preceding norm
#    consumer.weight    *= s.unsqueeze(0)                # per-column scale
# 5. quantize_per_group(consumer.weight, n_bit=4, group_size=128)
#
# Non-fused: o_proj, o_proj_moe_gen, mlp.down_proj, mlp_moe_gen.down_proj
#   → plain per-group asymmetric INT4 (no AWQ).
#
# lm_head: kept in bf16. "inference_lance asserts on its .weight pointer."
```

MLX-native port should be ~100 LOC: `mx.fast.quantized_matmul` already
provides the asymmetric per-group quant kernel, so we'd only need the
alpha-search loop + scale-fusion. No PyTorch dequant overhead → no ~10×
slowdown problem.

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

## Issue #1 — narrowed: structured-but-degraded mesh artifacts at n_lat ≥ ~30k

**Status:** Narrowed scope after Phase 5m + manual 49f verification
(2026-05-23). Originally "t2v collapses to pure noise at n_lat ≥ ~30k."
Phase 5m partially addressed it; new symptom is milder + more
actionable.

**Pre-Phase-5m symptom:** t2v at n_lat ≥ ~11,520 (768²×17f+) silently
degraded; at n_lat = 29,952 (768²×49f) collapsed to pure random noise.

**Post-Phase-5m envelope** (`cfg_renorm_type="channel"` default, v0.5.2):
- n_lat ≤ 9,216 (768²×13f):  🟢 Production
- n_lat = 11,520 (768²×17f): 🟢 Production (was degraded)
- n_lat = 16,128 (768²×25f): 🟢 Production (verified — bus + Big Ben)
- n_lat ≥ ~30k (768²×49f):  ❌ structured-but-degraded mesh artifacts
  (was pure noise; partial fix)

**Numerical signature of the residual failure (49f bus, seed=43):**
- Final std=0.623 vs ~0.88 for clean runs (17f/25f)
- Channel renorm clamps too aggressively at late timesteps once n_lat
  hits ~30k, pushing latents outside the VAE's trained distribution
- VAE outputs colored geometric mesh tiles overlaid on a barely-visible
  scene attempt (Big Ben silhouette + sky colors recognizable; bus lost)

**Open candidates for Phase 5n / future fix:**
1. **n_lat-aware renorm threshold** — currently constant; scaling the
   per-channel cap with n_lat magnitude may avoid over-clamping at scale
2. **cfg_interval=[0.4, 1.0]** — disable CFG entirely in the last steps
   (Phase 5d Cand 1b tested this at small scales; worth re-testing at
   49f specifically)
3. **Late-timestep VAE-distribution probe** — sample latent stats at the
   point of breakdown, compare against VAE input distribution from
   normal-scale runs to confirm the OOD hypothesis
4. **(longer-term)** VAE decoder retrained on Phase-5m-style latents

**Benefit:** Would unlock the full Lance reference scale (768²×49f
~4s clips; 480×848×121f Lance default ~10s clips). Currently capped at
768²×25f (~2s) on production. The narrower failure mode (degraded vs
noise) suggests a one-or-two-parameter fix is plausible — much more
tractable than the original pure-noise scope.

**Trigger:** Post-Phase-5c-DWQ (quantization gating). Or sooner if
someone wants to take a focused 1-2 day swing at the four candidates
above.

---

## x2t_video — full 6/6 oracle sweep

**Status:** Deferred. 2/6 cases validated (verbatim + content-correct).

**Missing locally:** vqa-01 (counting), vqa-02 (repeated-actions),
vqa-04 (time-manipulation narrative), caption-long-01 (butterfly+bee).

**Benefit:** Completeness — would close the Phase 0 oracle suite for
video understanding. Likely no surprises given the 2/6 already match
(VQA verbatim, captioning content-correct).

**Trigger:** If we need a complete oracle pass for a paper/writeup.

---

## x2t_image — PyTorch-parity gap on chart-value reads (cases 02/04)

**Status:** Deferred (2026-06-10). Revisit AFTER T2i + T2V land.

**The defect:** the Python MLX port misses 2/6 of the x2t_image oracle
cases vs the PyTorch capture — precise chart-value reads only:
case-02 answers "43" (expected "29%"), case-04 answers "1.8 million"
(expected "1.3 billion"). Binary/scene/OCR cases (01/03/05/06) pass.
That is a 33% failure rate on the image-understanding oracle; not
acceptable as a Lance-level pass.

**What's already ruled out (via the Swift port's 15-run L1 hunt):**
this is NOT a Swift issue — the Swift port reproduces the Python port
byte-identically on 5/6 cases (greedy), same wrong values on 02/04.
Both towers of the *MLX* implementation agree; the divergence is
MLX-port-vs-PyTorch, present since the initial port (the documented
~95% functional parity). Start from the original port: dump PyTorch
reference activations for case-02 and per-stage diff (resampler /
ViT window attention / mRoPE conventions / decoder) the same way the
Swift E6/E7 hunt worked — first stage below ~0.999 names the op.

**Benefit:** closes the last 2/6 of the Phase 0 image oracle; whatever
op is wrong here is shared by the Swift port (it inherits the fix) and
may also affect t2i/x2t_video quality on fine-detail reads.

**Trigger:** T2i + T2V complete. Accepted for MLXEngine integration in
the meantime (lance-mlx-swift v0.1.0 = MLX-parity, explicitly NOT a
PyTorch-parity claim).

**Cross-check vs RockTalk/Lance-3B-MLX (2026-06-11):** drove their bundled
`lance_mlx` X→T code + their F32 weights byte-as-published (driver:
`/tmp/rocktalk-code/run_x2t_oracle.py`, archived below) on all 6 oracle
cases, same AutoProcessor preprocessing + Lance prompt template as ours.
Result: case-02 **"43" — identical miss**; case-04 wrong AND less coherent
than ours ("28% … 300-500" word salad); cases 01/03 identical to ours;
case-05 **verbatim-identical** to our 461-char output; case-06 shares a
long prefix then near-tie tail split. Three independent MLX implementations
(our Python, our Swift, RockTalk F32) converge to near-identical greedy
outputs. This RULES OUT: F32-vs-bf16 precision, our MoT/decoder/prefill
code, KV-cache. Remaining shared suspects for the PyTorch gap:
(a) **preprocessing resolution** — chart reads are resolution-sensitive and
all MLX runs used AutoProcessor smart-resize defaults (grid 40×58 → 20×29
merged); check what min/max_pixels + resolution the upstream PyTorch Lance
capture used — PRIME suspect; (b) mlx_vlm's Qwen2.5-VL vision tower
numerics; (c) the prompt template. Start the deferred investigation at (a).

**ROOT CAUSE IDENTIFIED (2026-06-11) — preprocessing GEOMETRY, hypothesis
confirmed by direct test.** Upstream's x2t_image capture (`image_768res`)
does NOT use HF smart-resize. Its ViT stream is
`VideoTransform(resolution_vit=672, mode="bucket")`:
BucketResize(max_area=672², AR buckets [21:9,16:9,4:3,1:1,3:4,9:16],
stride 16 — torchvision RandomResizedCrop(scale=(1,1)) ⇒ deterministic
center-crop to bucket AR, bicubic resize to bucket dims) →
DivisibleCrop(28) → CLIP mean/std (normalization matches HF; geometry does
not). For the 800×557 chart this center-crops ~50-60 px of width and lands
on 756×560 — vs our native-AR 812×560 smart-resize.
**Test (`scripts/45_bucket_geometry_test.py`): replicating this geometry in
PIL and feeding OUR pipeline flips case-02 "43" → "29%" — EXACT oracle
match.** But the same geometry perturbs the other knife-edges (03 plate
last-char, 05/06 drift) — the PIL replication is not yet byte-exact to
torchvision and the model is extremely sensitive to input geometry.
**Deferred-phase plan, now well-scoped:** implement upstream's vit-stream
preprocessing torchvision-exact (the E6 lesson — byte-gate the preprocessed
pixels against the upstream transform, don't threshold), make it the
default x2t preprocessing in the port, re-gate all 6, then propagate to the
Swift port (lance-mlx-swift LancePILResize → bucket geometry).
Normalization, ViT numerics, decoder, and prompt template are all
EXONERATED for case-02; resolution per se was a red herring
(downsample_only — image is never upscaled).

**RESOLVED (2026-06-11, commit 0af739e) — systematic defect fixed; residual
reclassified as backend numerics.** Upstream-exact preprocessing implemented
byte-exact (`pipeline/upstream_und_preprocess.py`; pixels max|diff|=0.0 vs
upstream's verbatim torchvision code). Case-02 "29%" EXACT at every
resolution preset. Full verification chain vs upstream PT: position ids
exact (HF get_rope_index), prompt token-equal, attention semantics matched
(vision span bidirectional), no connector (vit_type qwen_2_5_vl_original),
logit vocab-mask added, ViT algorithmically EXACT (CPU-stream bisect:
1.000000 every stage; the GPU run's 0.886 worst-token was M5 fp32 matmul
accumulation noise — see memory note on M5 GPU matmul precision).
Residual: 3 greedy knife-edge trajectory splits (03 one char / 04 one
digit / 06 early flip) — the A100 capture (CUDA flash-attn, bf16 autocast)
is a third numerics point not reproducible locally even by PyTorch.
Open decision: A100 re-capture w/ recorded config + activation dumps
(definitive, ~$5) vs broader semantic eval vs defer Lance.
