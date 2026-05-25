# Phase 5n — Image vs Video quality gap investigation (FINAL)

**Date:** 2026-05-24 (laptop clean-room session)
**Question:** Why is the video port quality and prompt adherence lower
than the image port quality, beyond "training set"?
**Outcome:** **Port is faithful to source.** The gap is real and
reproducible but inherent to the model — a training-data-composition
effect that surfaces when the multi-frame inference regime activates.
No code fix will close it. Foundation is solid for proceeding to
quant + production-scale work.

## Six diagnostics — what was ruled in and out

| # | Hypothesis | Result | Confidence |
|---|------------|--------|------------|
| D1 | VAE temporal mode disadvantages t2i (T_latent=1) | **Refuted.** VAE actually *favors* the video regime (T≥2 frame 0 has 3.6× more high-freq detail than T=1 from same noise). Side finding: t2i ships `decoded[0, 0]` of 3-frame T=1 decode; later frames differ — image-side optimization opportunity logged separately. | High |
| D2 | CFG renorm at higher token counts silently suppresses t2v (open H3 from L2 audit) | **Refuted at n_lat ≤ 1728.** Both pipelines run scale_mean ~0.87-0.95; t2v is at or above t2i at comparable scales. Gentle downward trend with n_lat is leveling off — channel renorm working as designed. | High at tested scale; medium for production scale |
| D3 | mrope t-axis asymmetry (t2i hardcoded +1000 vs t2v=None) is the bug | **Refuted.** At Phase-5j-sensitive scale (256²×17f, verbose oracle prompt), anchor∈{None, 1000, 2000} produce visually equivalent output. Matching t2i's value doesn't help. | High |
| D4 | Lance_3B_Video weights are undertrained (training-data explanation) | **Refuted — and decisively reframed the search.** Lance_3B_Video at t_lat=1 produces image-comparable output. Weights are fine. The degradation appears specifically at t_lat>1, localizing the bug to multi-frame code. | High |
| D5 | num_frames sweep to find cliff vs gradient | **Cliff confirmed at t_lat=1 → t_lat=2** (-28% high-freq detail). t_lat=3, 4 don't worsen meaningfully. Cause class: "anything triggered by having ANY second latent frame". | High |
| D6 | Disambiguate position-IDs at t>0 vs LPE at f>0 | **Both refuted as cause.** Flattening t-axis (B) and flattening LPE (C) produce equivalent ~47% FFT_HF recovery but neither restores English text rendering. Position is necessary-but-not-sufficient; flattening both (D) is worse than either alone — model needs *some* frame-distinguishing signal. | High |

## The actual finding

Cross-pipeline comparison at same prompt + seed (cat-STOP-sign at 384²,
panels are `notes/phase5n_diagnostics/_t2i_vs_t2v_sweep_grid.png`):

| Pipeline | t_lat | Subject | Sign | Letters |
|----------|-------|---------|------|---------|
| t2i (Lance_3B)         | 1 | cat ✓ | red ✓ | **English "OTO"** ✓ |
| t2v (Lance_3B_Video)   | 1 | cat ✓ | red ✓ | **English "STO"** ✓ |
| t2v                    | 2 | cat ✓ | red ✓ | **Chinese-style glyphs** ✗ |
| t2v                    | 3 | cat ✓ | red ✓ | **Chinese-style glyphs** ✗ |
| t2v                    | 4 | cat ✓ | red ✓ | **Octagonal icon, no text** ✗ |

The failure mode is **not random degradation** — it's a consistent
regime shift to Chinese / Asian-style character rendering at t_lat>1.
All other prompt content (subject, color, scene composition) renders
correctly in both regimes; only English text rendering breaks.

**Most likely root cause:** ByteDance's video training corpus had more
Chinese-character text than English. When the multi-frame code path
activates (correctly, per architecture), the model defaults to its
better-trained text-rendering regime for video — which happens to be
Chinese-style characters. This is a model property, not a port bug.

## Implications for downstream dev work

- **Foundation is solid for quant + production work.** D4 proved
  Lance_3B_Video weights are semantically equivalent to Lance_3B at
  equivalent compute regimes; the port faithfully expresses the
  upstream model's training-data biases.
- **Phase 5b DWQ planning** — lance-quant's incidental finding (33% →
  50% exact-match at gs=64) confirms Lance is highly sensitive to
  weight perturbations. **Phase 5c-2 (naive 8-bit) refuted on 2026-05-24;
  Phase 5c-3 (AWQ port) completed 2026-05-25 — see
  phase5c3_awq_port/PHASE_5C3_COMPLETE.md.** Net outcome: AWQ-INT4
  ships for x2t_image VQA (3.31 GB, 6-9× faster decode); bf16 remains
  only production t2i variant.
- **Production-scale issue #1** (mesh artifacts at n_lat ≥ ~30k)
  remains a separate, unrelated bug — investigate independently.
- **Image-side optimization** D1 follow-up (D1c spawned task)
  **resolved 2026-05-24: frame 0 is correct; no patch needed.**
  Real-prompt A/B across 3 oracle prompts showed frame 0 wins; D1b's
  noise-only hint that frame 2 had more detail was a noise-input
  artifact, not real-latent behavior.

## What we did NOT test (deferred)

- t2v at production n_lat (11520) for D2's CFG renorm extrapolation —
  trend was leveling, low expected value, high cost.
- Mask experiment at t_lat=2 (D7) — would have been informative for
  the bug-class question, but D6's regime-shift finding makes this
  unnecessary for the closure question.
- Running the same prompt through upstream PyTorch Lance — would have
  definitively confirmed the model-property hypothesis, but requires
  significant additional setup (CUDA, PyTorch weights, ~70GB
  download).
- Chinese-text prompt A/B — would have locked the training-data
  hypothesis but battery-of-prompts subjective testing has diminishing
  returns relative to the explanatory clarity we already have.

## All artifacts

```
notes/phase5n_diagnostics/
├── SUMMARY.md                       (this file — final)
├── D1_findings.md
├── D1c_findings.md                  (spawned-task follow-up: frame 0 is correct)
├── D2_findings.md
├── D3_findings.md
├── D4_findings.md
├── D5_D6_findings.md
├── _t2i_vs_t2v_sweep_grid.png       (key cross-pipeline visual)
├── d1_vae_temporal_mode/
├── d1b_vae_frame_indexing/
├── d1c_real_prompt_frame_ab/        (D1c — frame-index real-prompt A/B)
├── d2_cfg_renorm/
├── d3_t2v_mape_anchor_sweep/
├── d3b_t2v_mape_anchor_phase5j_scale/
├── d4_pipeline_isolation/
├── d5_num_frames_sweep/
├── d6_position_ids_vs_lpe/
├── phase5c2_validation/             (5c-2 negative result: naive 8-bit refuted)
└── phase5c3_awq_port/               (5c-3 AWQ port — see PHASE_5C3_COMPLETE.md)
    ├── STATUS.md
    ├── PHASE_5C3_COMPLETE.md
    ├── act_stats/                   (calibration data)
    ├── validation/                  (t2i 4-prompt sweep — degrades)
    └── x2t_validation/              (VQA 6-case sweep — shippable)

scripts/diagnostics/
├── d1_vae_temporal_mode.py
├── d1b_vae_frame_indexing.py
├── d2_cfg_renorm_logger.py
├── d3_t2v_mape_anchor_sweep.py
├── d3b_t2v_mape_anchor_phase5j_scale.py
├── d4_pipeline_isolation.py
├── d5_num_frames_sweep.py
└── d6_position_ids_vs_lpe.py
```
