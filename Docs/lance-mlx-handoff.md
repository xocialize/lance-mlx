# Coding Handoff — Lance (ByteDance) → MLX Port

**Owner:** Dustin (MVS Collective)
**Date:** 2026-05-19
**Repo target:** `~/dev/lance-mlx/` (new) — GitHub `xocialize/lance-mlx`
**Hardware:** M5 Max, 128 GB unified memory (macOS 26.2+ required to exploit Neural Accelerators)

---

## Context

Educational deep-dive into unified multimodal modeling on Apple Silicon, building on prior MLX TTS/LLM/video work (RosettaCast, DubKit, Mel-RoFormer, VoxCPM2, the LTX-2.3 evaluation). First unified omni-modal generator on Apple Silicon if shipped.

**What Lance is, in one paragraph:**
ByteDance Intelligent Creation Lab's "Lance: Unified Multimodal Modeling by Multi-Task Synergy," paper arXiv:2605.18678 (submitted 2026-05-18), weights `bytedance-research/Lance` on HuggingFace under Apache 2.0. Dual-stream Mixture-of-Transformer-Experts: two Qwen2.5-VL-3B–initialized expert towers (`LLM_UND` and `LLM_GEN`) share an attention substrate but route tokens by modality — text/ViT-semantic tokens to `LLM_UND` (autoregressive next-token via LM head), Wan2.2-VAE-latent tokens to `LLM_GEN` (flow-matching velocity prediction via flow head). 3B active parameters; ~12B total LLM weight at bf16. Six task heads: `t2i`, `t2v`, `image_edit`, `video_edit`, `x2t_image`, `x2t_video`. Self-contained — no external Gemma-style text encoder.

**Key architectural pieces and their MLX prior art:**
- Understanding ViT — Qwen2.5-VL ViT — **already fully supported in `Blaizzy/mlx-vlm`**
- Wan2.2 3D causal VAE (16× spatial × 4× temporal) — **already in `Blaizzy/mlx-video`**, plus three standalone ports (`osama-ata/Wan2.2-mlx`, `Armanoide/Wan2.2.mlx`, `kryptx/Wan2.2-mlx`)
- Flow-matching sampler with CFG — **already in `Blaizzy/mlx-video`** for Wan2.x/LTX-2
- Dual-expert MoT routing + MaPE (modality-aware RoPE offset) — **net new**; first MoT in MLX
- Flow head over LLM hidden states — **net new**; small; few hundred LOC

This port sits at the intersection of `mlx-vlm` (understanding) and `mlx-video` (generation). Prince Canuma is the natural collaborator.

---

## Naming & identity

Decision recorded here so it stops mattering:

- **Python package / GitHub repo:** `lance-mlx` (snake_case import: `lance_mlx`)
- **mlx-community weight repos:** `mlx-community/Lance-3B-bf16`, `-8bit`, `-4bit`, plus `-Video-bf16`, `-Video-8bit`, `-Video-4bit`
- **README disambiguation:** every README starts with a one-liner clarifying *"Lance (ByteDance unified multimodal model, arXiv:2605.18678) — not to be confused with Lance/LanceDB columnar format."*

Rejected alternatives: `bytelance-mlx` (clunky), `lance-omni-mlx` (overspecific), `mlx-lance` (mlx-community convention is `<model>-mlx` for personal namespaces and `mlx-community/<Model>-<quant>` for org weights).

---

## Goals (priority order)

1. **Educational** — understand dual-expert MoT routing, MaPE, flow-matching over VAE latents, and how unified models orchestrate AR + flow heads in one pass. Each phase produces `notes/phaseN.md`.
2. **MVP: T2I + image understanding** — ship `mlx-community/Lance-3B-bf16` and `lance-mlx generate --task t2i / --task x2t_image` working end-to-end on M5 Max within ~2 weeks.
3. **Full feature parity** — add t2v, image_edit, video_edit, x2t_video over weeks 3–5. Ship 4-bit and 8-bit quants.
4. **Upstream** — coordinate with Prince Canuma on whether Lance lives as a standalone `lance-mlx` package depending on `mlx-vlm` + `mlx-video`, or gets folded into one of those.

**Non-goals (this iteration):**
- Swift port — deferred. `mlx-swift-examples` integration is Phase 6 stretch goal.
- Training / fine-tuning.
- Distillation / few-step variants (Lance-Lightning style).
- ComfyUI nodes.

---

## Hardware & environment

- **M5 Max 128 GB** unified memory, macOS 26.2+ for Neural Accelerator support
- **Memory bandwidth:** M5 Max is in the ~546 GB/s range; M5 Neural Accelerators per Apple ML Research's Nov 2025 blog give **~4× speedup vs M4 baseline for LLM TTFT** and **~3.8× faster FLUX-dev-4bit T2I vs M4** — Lance T2I should land in the 30–60s range at 768² bf16, possibly faster at int4. T2V at 50 frames / 480p budget 4–8 minutes per clip pre-optimization.
- Python ≥3.12 via `uv`
- HuggingFace CLI with write access to `mlx-community` org (request via Awni/Pedro/Prince before Phase 5)
- ffmpeg for video output
- **Disk:** ~80 GB across upstream Lance weights (60 GB) + MLX conversions (bf16 + 8bit + 4bit ≈ 25 GB per checkpoint × 2 checkpoints)
- **Cloud GPU for Phase 0 parity oracle:** rent an A100 or H100 (RunPod ~$1.50/hr, Lambda ~$1.10/hr) for 4–8 hours to capture reference outputs from ByteDance's PyTorch inference code. **Do not skip this step.** Without a parity oracle, you can't tell whether your MLX outputs are "different but valid" or "buggy."

---

## Phase 0 — Verify + capture parity oracle (½–1 day)

This is the highest-value pre-port investment. The goal is to confirm Lance works as advertised AND to capture reference outputs at fixed seeds so every later phase has something to diff against.

```bash
# Rent A100 or H100. RunPod or Lambda. ~4-8 hours total.
git clone https://github.com/bytedance/Lance ~/work/Lance
cd ~/work/Lance
pip install -r requirements.txt   # CUDA 12.4+, flash-attn 2.6.3, triton 3.1.0
huggingface-cli download bytedance-research/Lance --local-dir ./checkpoints/Lance
bash inference_lance.sh   # uses config/examples/*.json
```

Reference outputs to capture (save to your local repo's `tests/fixtures/`):
- 3× T2I prompts at seed 42, 30 steps, CFG 4.0, 768²
- 3× T2V prompts at seed 42, 30 steps, CFG 4.0, 480p × 50 frames
- 1× image_edit (e.g., "remove the person")
- 1× video_edit
- 1× x2t_image (VQA — use the official pie-chart / Colosseum demos)
- 1× x2t_video (caption a held-out clip)

For each: save the input, the exact prompt JSON, the seed/CFG/step count, and the full output (images as PNG, video as MP4, text as JSON). These are your **parity oracles**.

**Validation gate:** All 10 captures succeed; outputs are visually plausible; nothing obviously broken in the official PyTorch pipeline. If the official code doesn't work cleanly, file issues upstream and pause the port until resolved.

---

## Phase 1 — Repo bootstrap + weight conversion (3–5 days)

```bash
cd ~/dev/lance-mlx
uv init --python 3.12
uv add --dev pytest ruff
uv add mlx mlx-lm
# mlx-vlm: Qwen2.5-VL substrate
uv add "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git"
# mlx-video: Wan2.2 VAE + flow-matching sampler substrate
uv add "mlx-video @ git+https://github.com/Blaizzy/mlx-video.git"
uv add huggingface_hub safetensors numpy pillow imageio imageio-ffmpeg psutil tqdm

# Clone reference impl read-only for Phase 0 fixtures + key-name diffing
git clone https://github.com/bytedance/Lance ./vendor/Lance
```

### 1a. Inspect upstream weights

Run `scripts/01_inspect_keys.py` against the local downloaded Lance checkpoints to:
- Dump full key topology of `Lance_3B/model.safetensors` (24.7 GB) and `Lance_3B_Video/model.safetensors` (28.4 GB)
- Classify keys by component: `vit.*`, `llm_und.*`, `llm_gen.*`, `vae.*` (if present), `flow_head.*`, `lm_head.*`, `qk_norm_und.*`, `qk_norm_gen.*`, `connector.*`, `mape.*`
- Confirm the **untied LM head** (Lance unties lm_head from input embeddings while Qwen2.5-3B ties them — verify by shape diff between `model.embed_tokens.weight` and `lm_head.weight`)
- Confirm whether MaPE `Δ_m` offsets are learned (present in weights) or hard-coded (absent)
- Confirm flow head structure (MLP? thin projection? small DiT block?)

### 1b. Implement the modality-routed dual-expert model

Fork mlx-vlm's Qwen2.5-VL `LanguageModel` into `lance_mlx/model/lance_llm.py`:

```python
# Pseudocode for the routing pattern
class LanceMoTLayer(nn.Module):
    def __init__(self, config):
        # shared attention substrate
        self.attn = Attention(config)  # shared QKV proj
        # per-expert QK-norms (Lance-specific)
        self.qk_norm_und = QKNorm(...)
        self.qk_norm_gen = QKNorm(...)
        # two FFNs, two output projections
        self.ffn_und = SwiGLU(...)
        self.ffn_gen = SwiGLU(...)

    def __call__(self, h, modality_mask, ...):
        # modality_mask: (B, T) int — 0 = UND (text/ViT), 1 = GEN (VAE latent)
        # Apply attention with per-expert QK-norm
        attn_und = self.attn(h, qk_norm=self.qk_norm_und, ...)
        attn_gen = self.attn(h, qk_norm=self.qk_norm_gen, ...)
        attn_out = mx.where(modality_mask[..., None] == 0, attn_und, attn_gen)
        # Route FFN per-token
        ffn_und_out = self.ffn_und(h)
        ffn_gen_out = self.ffn_gen(h)
        return mx.where(modality_mask[..., None] == 0, ffn_und_out, ffn_gen_out)
```

This is "static routing" — no learned router, deterministic from token modality metadata. Much simpler than a Mixtral/DeepSeek-style MoE.

### 1c. Implement MaPE (modality-aware RoPE)

One-line modification to mlx-vlm's Qwen2.5-VL 3D RoPE:

```python
def apply_mape_rope(q, k, position_ids, modality_group, delta_m_offsets):
    # position_ids: (B, T, 3) — (t, h, w) coordinates
    # modality_group: (B, T) int in {0: ViT-semantic, 1: clean VAE, 2: noisy VAE}
    # delta_m_offsets: (3,) — temporal offset per modality group
    t_offset = delta_m_offsets[modality_group]  # (B, T)
    position_ids = position_ids.copy()
    position_ids[..., 0] = position_ids[..., 0] + t_offset
    return qwen25_3d_rope(q, k, position_ids)
```

### 1d. Vendor the Wan2.2 VAE from `Blaizzy/mlx-video`

Don't re-port. Use mlx-video's `WanVAE` directly:

```python
from mlx_video.models.wan_2.vae import WanVAE
vae = WanVAE.from_pretrained("~/models/lance/Wan2.2_VAE.safetensors")
# Note: upstream ships as .pth (pickled); convert to .safetensors first
```

If the Lance `Wan2.2_VAE.pth` has architectural deltas vs. the stock Wan2.2 VAE that mlx-video supports, document in `notes/phase1_vae_diff.md` and either patch in-place or upstream a Lance-specific variant to mlx-video. **The published Wan2.2 channel-count footgun** (T2V uses 16-channel latents but the 48-channel VAE in `wan2.2_vae.safetensors` will produce shape-mismatch errors; correct VAE for T2V is `wan_2.1_vae.safetensors`) **must be verified before going further.** First step: print `Wan2.2_VAE.pth` state-dict input/output channel counts and compare to `LLM_GEN`'s flow head output projection.

### 1e. Write the converter

`scripts/02_convert.py` — translates HF safetensors keys → MLX module tree, applies dtype cast (bf16 default), saves single-file `model.safetensors` in MLX format.

**Validation gates for Phase 1:**
- Inspection report confirms architectural facts in this handoff
- Converter produces loadable MLX safetensors
- Dummy forward pass on a 768² latent (random init) produces correct-shape output
- Wan2.2 VAE round-trip (encode → decode an image) produces visually identical output

---

## Phase 2 — Understanding pipeline (x2t_image, x2t_video) (2–3 days)

The easy half. Re-use mlx-vlm's autoregressive decode loop with the new dual-expert model where text+ViT tokens route entirely to `LLM_UND`.

```bash
uv run python -m lance_mlx generate \
  --task x2t_image \
  --image ./tests/fixtures/colosseum.png \
  --prompt "What is the structure shown in the image and what is its historical significance?" \
  --weights ~/models/mlx/Lance-3B-bf16
```

**Validation gates:**
- Match PyTorch reference VQA outputs token-for-token at greedy decode (do_sample=False) on the Phase 0 fixtures
- ≥ 95% token agreement vs. PyTorch reference

---

## Phase 3 — Image generation (t2i, image_edit) (5–7 days)

Build the flow-matching denoising loop:
- 30 steps, linear interpolant `x_t = t·x_1 + (1-t)·x_0`
- Velocity target `x_1 - x_0`
- Timestep shift = 3.5
- CFG-text-scale = 4.0
- Optional CFG renorm (knobs match Wan2.2: `cfg_renorm_type`, `cfg_renorm_min`, `cfg_interval`)

```bash
uv run python -m lance_mlx generate \
  --task t2i --prompt "A red fox in tall grass at golden hour" \
  --seed 42 --steps 30 --cfg 4.0 --resolution 768 \
  --weights ~/models/mlx/Lance-3B-bf16
```

For `image_edit`: VAE-encode the input image (clean latents), concatenate with noisy target latents and edit-instruction text tokens, denoise the noisy half conditional on the clean half + instruction.

**Validation gates:**
- T2I at seed 42 / CFG 4.0 / 30 steps matches PyTorch reference fixtures via FID (CLIP or DINOv2 features) < 0.05 and CLIPScore agreement within 0.005
- image_edit matches reference on the Phase 0 edit fixture

---

## Phase 4 — Video generation (t2v, video_edit) (5–10 days)

Extend the flow loop to handle the temporal dimension:
- Wan2.2 3D causal VAE produces (T/4, H/16, W/16, 16) latents
- Flow head predicts velocity on the full 4D latent
- 3-frame batched VAE decode (per DrawThings's `WanVAE.swift` pattern) keeps peak memory in check on 120-frame outputs

```bash
uv run python -m lance_mlx generate \
  --task t2v --prompt "..." --seed 42 \
  --frames 50 --fps 12 --resolution 480 \
  --weights ~/models/mlx/Lance-3B-Video-bf16
```

**Validation gates:**
- 50-frame 480p generation at seed 42 from a Phase 0 fixture
- Mean per-frame LPIPS vs PyTorch reference < 0.02
- VBench Total Score within 1.5 points of paper's 85.11 on a 15–30 prompt subsample (full VBench is overkill for parity)

---

## Phase 5 — Quantization + packaging + publish (3–4 days)

### 5a. Quantize

Per-expert quantization is a knob worth exploring — UND and GEN towers can have different bit widths if quality demands it. Suggested defaults:

```bash
uv run python scripts/05_quantize.py --mlx-path ~/models/mlx/Lance-3B-bf16 \
  --output ~/models/mlx/Lance-3B-8bit --q-bits 8 --q-group-size 32
uv run python scripts/05_quantize.py --mlx-path ~/models/mlx/Lance-3B-bf16 \
  --output ~/models/mlx/Lance-3B-4bit --q-bits 4 --q-group-size 32
```

Validate each quantization tier matches Phase 3/4 fixtures within tolerance — relax thresholds at 4-bit (FID < 0.10, LPIPS < 0.04).

### 5b. Coordinate before publishing

Same checklist pattern as the LTX-2.3 handoff:

1. Open issue on `Blaizzy/mlx-vlm` AND `Blaizzy/mlx-video` discussing the Lance port — propose either: (a) standalone `lance-mlx` package depending on both, or (b) folding into one. Let Prince decide; he is effectively mlx-community for VLM and video.
2. Open PRs for any code changes needed in `mlx-vlm` (likely: support for untied lm_head on Qwen2.5-VL variants, per-expert QK-norm) and `mlx-video` (likely: Lance-specific WanVAE variant if it differs).
3. Email `mengqi.huang@bytedance.com` / `jianzhu.guo@bytedance.com` (paper corresponding authors) with a heads-up about the port. ByteDance research has historically been receptive; informal architectural clarifications save days.
4. Verify Apache 2.0 attribution requirements: include verbatim LICENSE in every uploaded repo, include NOTICE crediting ByteDance Intelligent Creation Lab.
5. Convert `Wan2.2_VAE.pth` to safetensors before any HF upload (clears HF's pickle-security flag).

### 5c. Publish

Run `scripts/06_publish_hf.py --commit` after the coordination checklist passes. Targets:
1. `mlx-community/Lance-3B-bf16`
2. `mlx-community/Lance-3B-8bit`
3. `mlx-community/Lance-3B-4bit`
4. `mlx-community/Lance-3B-Video-bf16`
5. `mlx-community/Lance-3B-Video-8bit`
6. `mlx-community/Lance-3B-Video-4bit`

Each repo: README with architecture summary, benchmarks on M5 Max, conversion provenance, parity-check results, LICENSE, NOTICE.

### 5d. Announce

- Launch blog post in `notes/launch_post.md`
- Cross-post r/LocalLLaMA, X, HN — "First unified omni-modal generator running on Apple Silicon"
- Tag the Lance paper authors + Prince Canuma + Awni Hannun for visibility

---

## Open questions / decision points

1. **Standalone package or upstream into mlx-vlm/mlx-video?** Decide in Phase 5a based on Prince's preference. Default: standalone `lance-mlx` depending on both upstream packages.
2. **Wan2.2 VAE version mismatch (16-channel vs 48-channel).** Resolve in Phase 1d — print state dict shapes and compare to flow head output dim. Lance's bundled `Wan2.2_VAE.pth` may be a custom variant.
3. **Per-expert quantization.** Worth a short experiment: quantize UND tower at q4 (understanding is robust to quantization) and GEN tower at q8 (image quality is sensitive). Mixed-precision packages are unusual but legitimate.
4. **MaPE Δ_m offsets — learned or hard-coded?** Inspect the safetensors key list in Phase 1a. If learned, they must round-trip through conversion. If hard-coded, read them from `llm_config.json` or from the source code.
5. **Coordinate with `dgrauet` (LTX-2.3) and other ByteDance-port authors.** This is a different model family, so probably less coordination needed than LTX-2.3, but worth a shout in the launch post.

---

## References

- Lance paper: https://arxiv.org/abs/2605.18678
- Lance HF: https://huggingface.co/bytedance-research/Lance
- Lance code: https://github.com/bytedance/Lance
- Lance project page: https://lance-project.github.io/
- mlx-vlm: https://github.com/Blaizzy/mlx-vlm
- mlx-video: https://github.com/Blaizzy/mlx-video
- Wan2.2 reference: https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B
- Apple M5 NA blog: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- BAGEL (architectural sibling, fallback target): https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

---

## Notes for the coding agent

- **Phase 0 parity oracle is non-negotiable.** Without PyTorch reference outputs at fixed seeds, you cannot tell port bugs from "different but valid" outputs.
- **Read `vendor/Lance/modeling/*.py` end-to-end before writing MLX code.** Especially `lance.py`, `qwen2.py`, `vae/wan/model.py`, `vit/qwen2_5_vl_vit.py`. The whole source is ~3,000 LOC.
- **Vendor, don't re-port, when MLX prior art exists.** Wan2.2 VAE → use `mlx-video`'s. Qwen2.5-VL ViT → use `mlx-vlm`'s. Flow-matching sampler → use `mlx-video`'s. Only re-implement what's genuinely Lance-novel: dual-expert routing, MaPE, flow head, the six task pipelines.
- **Every phase produces a `notes/phaseN.md`.** Educational value is the primary deliverable; the published weights are the bonus.
- **Capture timing and memory for every generation.** Reuse `lance_mlx.bench.RunRecord` (same pattern as the LTX project's `ltx_mlx_eval.bench`).
- **Don't auto-publish.** Every HF push goes through `scripts/06_publish_hf.py --commit` with explicit confirmation. Dry-run is the default.
- **License compliance:** Apache 2.0 everywhere (Lance, Wan2.2, Qwen2.5-VL all Apache). Include LICENSE + NOTICE in every uploaded repo with full attribution.
- **The model card on HF mis-classifies architecture due to bundled Qwen 3.5 GGUF in `prompt_enhancer/`** — same trap as the Sulphur/qwen35 false positive in the LTX work. Pass `--architecture lance` explicitly to any converter that auto-detects from HF metadata.
