# Lance-MLX t2v residual quality — research brief

**Audience:** A researcher or ML practitioner with experience in Wan2.2-family video diffusion models, MLX/Metal numerical precision, or Apple Silicon LLM inference. The deliverable we want is *actionable diagnosis* of what's left in our port, not a fresh implementation.

**Date:** 2026-05-21
**Repo:** https://github.com/xocialize/lance-mlx
**HF:** https://huggingface.co/collections/mlx-community/lance-mlx-6a0f3cd5648a74f8283fc8a4

---

## What we've built and confirmed

We've ported ByteDance's [Lance unified multimodal model](https://huggingface.co/bytedance-research/Lance) to MLX for Apple Silicon. Lance is a 3B-active MoT (Mixture-of-Transformer-Experts) model with two parallel expert towers (`LLM_UND` for understanding/text, `LLM_GEN` for generation), modality-deterministic routing, flow-matching velocity prediction, and a bundled 48-channel Wan2.2 3D causal VAE.

**Confirmed working at PyTorch reference quality:**
- `t2i` (text → image, Lance_3B): photorealistic, prompt-aligned
- `image_edit` (instruction-based): identity + style + signature preserved
- `x2t_image` / `x2t_video` (VQA): content-correct vs Phase 0 oracle

**Confirmed working but residual fine-detail gap:**
- `t2v` (text → video, Lance_3B_Video): **after our Phase 5d MaPE-shift fix**, output is photorealistic 3D-cinematic — clear subject, correct props, correct composition. But water surfaces, fine textures, paws, and surfboard detail are softer / show watercolor-flavored artifacts vs the PyTorch oracle.

**The model itself can produce photorealism — that's confirmed.** The Phase 0 PyTorch oracle outputs (in `tests/fixtures/results/t2v_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_091630/`) are unambiguously photoreal, generated with the exact weights we converted and the exact sampler config we use. So the residual gap is in our MLX port, not the underlying weights or training.

---

## The question we want answered

What specific port-side numerical or routing deviation(s) cause the residual fine-detail loss in our MLX t2v vs the PyTorch reference?

We've solved the gross-level deviation already (MaPE shift). We're now hunting subtler deviations — likely a combination of bf16 numerical precision in RoPE/attention and one or two additional details we've missed.

**Sub-questions** (any of these would be valuable):

1. **Is there a known bf16-vs-fp32 quality floor for Wan2.2-family video diffusion at inference time?** If yes, what's its character (smearing, color shift, detail loss, frame-to-frame drift)? Does it match what we see (watercolor artifacts in water, fine textures)?

2. **Has anyone else ported Lance or Wan2.2 to MLX/Apple Silicon and reported similar artifacts?** Specifically [RockTalk/Lance-MLX](https://huggingface.co/RockTalk/Lance-3B-Video-MLX) — they shipped a Lance MLX port one day before our public release. If their t2v at 256²×9f looks clean, comparing their pipeline code against ours pinpoints the gap. If their output has the same artifacts, that's evidence of a bf16 fundamental.

3. **Are there documented best practices for porting Qwen2.5-VL-based diffusion models from PyTorch to MLX?** mlx-vlm has its own bf16 RoPE handling; Lance specifically uses a more sensitive `_moe_gen` expert tower for the generation path.

4. **Is there a specific Wan2.2 VAE decode artifact pattern** (e.g., temporal-causal-cache state drift, RMSNorm precision, 3D-conv channel-bleed) that matches the "water gets watercolor-flavored" artifact we observe?

---

## What we've already ruled out

We've tested or eliminated these candidates ourselves. **Please don't repeat them**; they're useful as priors only:

- **Wrong weights**: SHA256 confirms our converted Lance_3B_Video matches upstream byte-for-byte for embed/lm_head; weight diff shows expected `_moe_gen` deltas, no corruption.
- **Wrong sampler config**: same `validation_timestep_shift=3.5, cfg_text_scale=4.0, validation_num_timesteps=30, seed=42` as the Phase 0 PyTorch oracle.
- **MaPE temporal shift to 2000**: confirmed bug (removed, Phase 5d fix). Output went from painterly-impressionistic to photoreal-but-soft.
- **`cfg_interval=[0.4, 1.0]`**: tested, no visible difference vs always-on CFG at our test scales.
- **Prompt-content interpretation issue**: real but separate — we addressed it in Phase 4e.
- **Painterly aesthetic "by design"**: false; refuted by oracle data.
- **48-channel vs 16-channel Wan2.2 VAE**: we use the correct bundled 48-channel; VAE roundtrip MAD ~7/255 on real photo at 768².
- **Per-tower QK-norm differences (`_moe_gen`)**: real but unrelated to the painterly aesthetic per the bisect.

---

## Candidates we have NOT yet tested but suspect

In rough priority order:

1. **fp32 RoPE rotation in the generation path.** mlx-vlm's `Qwen2RotaryEmbedding` (in `mlx_vlm/models/qwen2_5_vl/language.py:48-73`) computes cos/sin in fp32 but casts to `x.dtype` (bf16) before `apply_multimodal_rotary_pos_emb`. The actual `q * cos + rotate_half(q) * sin` rotation runs in bf16. At our latent token positions this could accumulate precision loss the PyTorch reference doesn't have. Concrete patch: wrap the rotation to upcast to fp32, then cast q/k_embed back.

2. **Position-grid scaling by `spatial_merge_size`**. Our `t2v.py::_build_position_ids` uses raw h_lat, w_lat for h/w positions. Upstream Qwen2.5-VL's `get_rope_index` divides h, w by `spatial_merge_size=2` for visual tokens. We never tested if our VAE-latent positions should follow this convention.

3. **VAE encoder/decoder temporal-cache state**. Wan2.2 VAE decode is causal-temporal-chunked (1+4+4+... pattern with per-conv `feat_cache`). If we're resetting the cache between chunks differently than PyTorch, fine-detail temporal coherence (water flow, fur edges) could degrade.

4. **bf16 attention softmax precision** at our long sequence lengths (4k-10k+ tokens). mlx-vlm's `scaled_dot_product_attention` runs in bf16. The dynamic range of the softmax may be losing the resolution PyTorch keeps in fp32.

5. **TimestepEmbedder sigma scaling**. We pass timesteps in `[0, 1]` directly. PyTorch reference may scale by 1000 internally. We hit a similar bug for image (Phase 3d) but never explicitly verified t2v inherited the right convention.

---

## Where the gap is visually (concrete description for any researcher)

Open these two videos side-by-side. Both: same prompt "A red panda surfing on a bright seaside wave, wearing a gold-trimmed cap and travel satchel..."; same seed 42; same sampler config (30 steps, CFG=4.0, shift=3.5).

- **PyTorch oracle** (`tests/fixtures/results/t2v_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_091630/000000.mp4`): photorealistic 3D-cinematic. Clean smooth water with realistic refraction and foam. Sharp red panda fur. Crisp surfboard. Premium cinematic lighting.
- **MLX (this port, post-Phase-5d-fix)** (`tests/fixtures/lance_vs_ltx_pre_fix/p00_oracle_panda_480x704_noshift_BREAKTHROUGH.mp4`): photorealistic 3D-cinematic for SUBJECT (red panda + gold cap rendered well). But water surface has watercolor-flavored swirls instead of clean refraction. Paws and surfboard outline are slightly softened. Overall composition correct, fine detail off.

The artifact pattern is **localized to high-frequency texture regions** (water, paws, fine fur edges), not global. That's why we suspect numerical-precision issues rather than position/routing — position bugs tend to break globally.

---

## Useful repos and sources to investigate

- **Our repo (the port)**: https://github.com/xocialize/lance-mlx
  - `src/lance_mlx/pipeline/t2v.py` — generation loop
  - `src/lance_mlx/model/lance_llm.py` — MoT layer / attention
  - `src/lance_mlx/model/mape.py` — MaPE positional shift (now mostly unused)
  - `src/lance_mlx/model/latent_pos_embed.py` — learned LPE
  - `notes/phase5d_breakthrough.md` — full record of what we fixed and how
  - `notes/phase4c_findings.md` — earlier analysis (now superseded but contains useful empirical data)

- **Upstream Lance reference (PyTorch)**: https://github.com/bytedance/Lance
  - `modeling/lance/lance.py` — the main forward function (`validation_gen` is the inference path we care about)
  - `data/common.py::shift_position_ids` — the position-handling code we ported (this is the gate that misled us in Phase 4)
  - `data/data_utils.py::create_sparse_mask` — attention mask construction
  - `modeling/vae/wan/vae2_2.py` — VAE encode/decode (chunked causal temporal)

- **RockTalk's MLX port (parallel attempt)**: https://github.com/RockTalk/Lance-MLX and https://huggingface.co/RockTalk/Lance-3B-Video-MLX
  - They published one day before us; their t2v sample is verified at 256²×9f. **Highest-value comparison target** — if their output is sharp, the gap is something specific to our impl. If their output also has watercolor artifacts, it's bf16 fundamental.
  - Specifically inspect: their `qwen2_navit_mlx.py` (routing), their VAE port, how they call attention/RoPE.

- **Reza2kn/lance-quant** (quantization toolkit): https://github.com/Reza2kn/lance-quant
  - Not directly a port but contains numerical-sensitivity analysis of Lance's GEN tower. Their finding: naive INT4 produces gibberish; v2 at group_size=64 reaches 50% byte-match. This is a corroborator that the GEN expert is numerics-sensitive — and at bf16 we may be hitting a softer version of the same issue.

- **mlx-vlm**: https://github.com/Blaizzy/mlx-vlm
  - `models/qwen2_5_vl/language.py:Qwen2RotaryEmbedding` — the RoPE we inherit
  - `models/qwen2_5_vl/language.py:apply_multimodal_rotary_pos_emb` — the rotation application
  - Worth checking their open issues for bf16 quality complaints.

- **mlx-video**: https://github.com/Blaizzy/mlx-video
  - `models/wan_2/vae22.py` — the Wan2.2 VAE we use (Lance ships a re-trained 48-channel variant; we load Lance's into mlx-video's class)
  - Worth checking issues for "watercolor", "smearing", or "low quality bf16" reports.

---

## What we'd find most useful as research output

In rough order of preference:

1. **A specific patch or pointer**: "Your `_build_position_ids` is wrong because upstream uses XYZ" or "RockTalk's port uses fp32 RoPE here and it cleans up the water artifacts."

2. **A ranked list of remaining candidates**: not just our list above, but new ones we haven't thought of, prioritized by their estimated contribution to the visible gap.

3. **A definitive "bf16 floor" determination**: if the gap is fundamentally what you get when porting Wan2.2-family models to MLX bf16, knowing that lets us stop chasing and ship the current state as the practical ceiling.

4. **A note on what to AVOID**: things we'd be tempted to try that won't help.

We're not asking for a fresh implementation. We want enough information to know whether to spend another week iterating fixes or to declare the port at its practical ceiling.

---

## How to reproduce our current state

```bash
git clone https://github.com/xocialize/lance-mlx
cd lance-mlx && uv sync

# Download the converted Lance_3B_Video weights:
HF_HUB_DISABLE_XET=1 uv run huggingface-cli download mlx-community/Lance-3B-Video-bf16

# Run our best-looking config:
HF_HUB_DISABLE_XET=1 uv run python scripts/10_t2v_demo.py \
  --prompt "A red panda surfing on a bright seaside wave with foam spray." \
  --lance-weights ~/.cache/huggingface/hub/models--mlx-community--Lance-3B-Video-bf16/snapshots/*/ \
  --vae-weights   ~/.cache/huggingface/hub/models--mlx-community--Lance-3B-Video-bf16/snapshots/*/vae.safetensors \
  --num-frames 17 --height 480 --width 704 \
  --steps 30 --cfg-scale 4.0 --seed 42 \
  --output-mp4 /tmp/lance_t2v.mp4
```

Compare against `tests/fixtures/results/t2v_sample_*/000000.mp4` (the PyTorch oracle on disk).

Or compare the saved MLX output: `tests/fixtures/lance_vs_ltx_pre_fix/p00_oracle_panda_480x704_noshift_BREAKTHROUGH.mp4`.

---

## Contact

GitHub issue for tracking: https://github.com/xocialize/lance-mlx/issues/2
