# lance-mlx

> **Note:** "Lance" here refers to **ByteDance Intelligent Creation Lab's unified multimodal model** ([paper](https://arxiv.org/abs/2605.18678), [weights](https://huggingface.co/bytedance-research/Lance)), **not** [Lance/LanceDB](https://github.com/lancedb/lance) (the columnar data format).

MLX port of **Lance** for Apple Silicon. Lance is a 3B-active / ~12B-total parameter dual-stream Mixture-of-Transformer-Experts model that unifies image and video understanding, generation, and editing in a single framework. This package brings Lance to Apple Silicon via MLX, with weights hosted on the `mlx-community` HuggingFace organization.

## 📦 Weights on Hugging Face (`mlx-community`)

All repos live in the **[Lance MLX collection](https://huggingface.co/collections/mlx-community/lance-mlx-6a0f3cd5648a74f8283fc8a4)** for one-click browsing.

### Recommended by task (TL;DR)

| Task | Use | Why |
|---|---|---|
| **`t2i`** (text→image) | `Lance-3B-bf16` | Only validated path; quant variants all degrade — see research notes below |
| **`image_edit`** | `Lance-3B-bf16` | Same precision requirement as t2i |
| **`t2v`** (text→video) | `Lance-3B-Video-bf16` | Production within n_lat ≤ 16,128 |
| **`video_edit`** | `Lance-3B-Video-bf16` | Same envelope as t2v |
| **`x2t_image`** (VQA) on 32 GB+ Macs | `Lance-3B-bf16` | Full bf16 quality |
| **`x2t_image`** (VQA) on 8-16 GB Macs | `Lance-3B-AWQ-INT4` | 27% size, 6-9× faster decode, ~4/6 oracle parity with bf16 |
| **`x2t_video`** (VQA) | `Lance-3B-Video-bf16` | No video-VQA quant variant — bf16 only |
| Wan2.2 VAE (standalone) | `Wan2.2-VAE-Lance-bf16` | Shared by image + video pipelines |

### Full repo list

| Repo | Status | Notes |
|---|---|---|
| [`mlx-community/Lance-3B-bf16`](https://huggingface.co/mlx-community/Lance-3B-bf16) | 🟢 Production | `t2i`, `image_edit`, `x2t_image` (full quality, ~15 GB) |
| [`mlx-community/Lance-3B-AWQ-INT4`](https://huggingface.co/mlx-community/Lance-3B-AWQ-INT4) | 🟢 Production (VQA only) | `x2t_image` on 8-16 GB Macs (5.65 GB repo, 3.31 GB LLM, 6-9× faster long-form decode). **Don't use for t2i** — quant research notes below. |
| [`mlx-community/Lance-3B-8bit`](https://huggingface.co/mlx-community/Lance-3B-8bit) | ⚠️ Superseded | Naive 8-bit broken on image gen; `Lance-3B-AWQ-INT4` is the working compressed variant for VQA. Kept for historical reproducibility. |
| [`mlx-community/Wan2.2-VAE-Lance-bf16`](https://huggingface.co/mlx-community/Wan2.2-VAE-Lance-bf16) | 🟢 Production | 48-ch Wan2.2 VAE (standalone, shared by image + video pipelines) |
| [`mlx-community/Lance-3B-Video-bf16`](https://huggingface.co/mlx-community/Lance-3B-Video-bf16) | 🟢 Production through 768²×25f | `t2v` (photoreal after Phase 5j + 5m fixes), `x2t_video`, `video_edit` |

### Quantization research — closed as RESEARCH (Phase 5b → 5c-3h, May 2026)

**Bottom line: bf16 is the only production-grade variant for image generation. AWQ-INT4 ships for VQA only.** We are not actively developing further quantization variants.

What we tried (each linked to a notes writeup):

- **Naive groupwise 4-bit + 8-bit, full and UND-only tower configurations** ([phase5b_quantization_findings.md](notes/phase5b_quantization_findings.md), [phase5c2_validation/FINDINGS.md](notes/phase5n_diagnostics/phase5c2_validation/FINDINGS.md)) — all produce ~80% high-frequency detail loss on Lance image generation. Text-only forward (VQA) less affected. `mlx-community/Lance-3B-8bit` is from this lineage.
- **DWQ (4-bit UND-only with mlx-lm's distillation harness)** ([phase5c-1 in BACKLOG.md](BACKLOG.md)) — 1/4 prompts acceptable; mlx-lm DWQ has a hardcoded `bits < 8` gate so no 8-bit variant possible.
- **AWQ (Reza2kn-style alpha-search + per-channel scale fusion, ported to MLX)** ([phase5c3_awq_port/PHASE_5C3_COMPLETE.md](notes/phase5n_diagnostics/phase5c3_awq_port/PHASE_5C3_COMPLETE.md)) — code lives in [`src/lance_mlx/quant/`](src/lance_mlx/quant/). Produces shippable VQA variant at 4-bit; doesn't close the image-gen gap.

What we learned ([phase5c3_awq_port/PHASE_5C3H_FINDINGS.md](notes/phase5n_diagnostics/phase5c3_awq_port/PHASE_5C3H_FINDINGS.md)):

- **AWQ math is working correctly per-Linear.** Weight-level introspection at 6 representative Linears confirms AWQ reduces per-layer output MSE by 28% on average at 8-bit, 20% at 4-bit.
- **Per-layer gains don't compound into end-to-end image quality** because Lance t2i runs **2,160 forward-pass evaluations per image** (36 layers × 30 Euler steps × 2 CFG arms). Errors at each Euler step feed the next step's input via the flow-matching integrator. Per-step quant improvements average out over this long path.
- **Middle layers (around layer 18) are AWQ's blind spot** — their activations don't have the strong per-channel outlier pattern AWQ assumes, and middle-layer AWQ regressions partially cancel peripheral-layer gains.
- **The 80% HF floor is architectural, not algorithmic.** k-quants from llama.cpp, NVFP4, custom Metal kernels would all face the same compounding bottleneck. No quant scheme tested or hypothesized would close this floor without changing Lance's architecture itself.

Forward pointer for downstream `mlxEngine` work: [`notes/mlx_engine_quant_notes.md`](notes/mlx_engine_quant_notes.md) — captures the Lance-specific constraints and the two speculative paths that target compounding directly (hybrid precision per layer position, or step-conditional AWQ calibration) rather than per-layer quant precision.

For now, **bf16 is the recommended path for any image generation use case, on any Apple Silicon hardware that can fit it.** AWQ-INT4 unlocks Lance VQA on smaller Macs.

## Status

🟢 **Image and video both production-quality through n_lat ≤ 16,128 (2026-05-23).** Image pipelines (t2i, image_edit, x2t_image) reproduce the bf16 PyTorch reference. Video pipelines (t2v, video_edit, x2t_video) landed two production fixes: **Phase 5j position-ID fix** (`latent_pos_base=0`, closed [issue #2](https://github.com/xocialize/lance-mlx/issues/2)) and **Phase 5m CFG-renorm fix** ([v0.5.2](https://github.com/xocialize/lance-mlx/releases/tag/v0.5.2-phase5m-cfg-renorm), `cfg_renorm_type="channel"`). **Production envelope: n_lat ≤ 16,128** (256² through 768²×25f, 480×704×17f) — verified across two diagnostic prompts (panda surfing, bus + Big Ben). Phase 5m root cause: the legacy `"global"` CFG-renorm cap computed a single scalar L2 over the full velocity tensor, so at ~2× n_lat the same cap silently over-suppressed high-frequency detail. `"channel"` clamps per-channel without dragging the aggregate. Within the envelope, n_lat → quality is stochastic seed × scale, not a monotonic degradation curve. At **n_lat ≥ ~30k (768²×49f, 480×848×121f)** Phase 5m partially closes the original pure-noise failure to a milder "structured-but-degraded with mesh artifacts" regime — model attempts the scene but VAE outputs colored geometric tiles overlaid throughout (verified manually 2026-05-23, final std=0.623 vs ~0.88 for clean runs — channel renorm clamps too aggressively at late timesteps once n_lat hits ~30k, pushing latents outside the VAE's trained distribution). Narrowed scope of [issue #1](https://github.com/xocialize/lance-mlx/issues/1).

| Capability | Status |
|---|---|
| Convert HF safetensors → MLX bf16 (both checkpoints + Wan2.2 VAE) | ✅ `scripts/02_convert.py`, `scripts/06_convert_wan_vae.py` |
| Load `Lance_3B` + `Lance_3B_Video` into `LanceModel` | ✅ 0 missing keys, dummy forward verified |
| **x2t_image VQA (image → text answer)** | **✅ Production. Content-correct across all 6 oracle cases.** |
| KV cache for fast autoregressive decode | ✅ 1.7×–2.8× speedup on long generations |
| **t2i (text → image generation)** | **✅ Production. Photorealistic, prompt-aligned output.** |
| **image_edit (instruction-based)** | **✅ Production. "Remove hat" preserves identity + style + signature; "Add pearl necklace" leaves rest intact.** |
| **t2v (text → video)** | ✅ **Production. Photoreal/CGI-quality output** at n_lat ≤ 16,128 (256–768² × ≤25f; 480×704×17f; 640²×17f) after Phase 5j position-ID fix + Phase 5m CFG-renorm fix. Verified across two diagnostic prompts (panda surfing, bus + Big Ben). ❌ Degraded with mesh artifacts at n_lat ≥ ~30k (768²×49f, 480×848×121f) — [issue #1](https://github.com/xocialize/lance-mlx/issues/1) narrowed from "pure noise" to "structured-but-degraded mesh artifacts". |
| **x2t_video (video VQA)** | **✅ Validated against Phase 0 oracle.** Cooking video → kitchen+pan+spatula+tomato+meat all content-correct in 17.5 s. |
| **video_edit (instruction-based)** | ✅ Same envelope as t2v: works at ≤9,216 latent tokens after Phase 5d fix. |
| 8-bit + 4-bit quants + HF community variants | ⏳ Phase 5b |

**Try it:**
```bash
# Install
git clone https://github.com/xocialize/lance-mlx && cd lance-mlx && uv sync

# Download production-ready image MVP (~15 GB):
HF_HUB_DISABLE_XET=1 uv run huggingface-cli download mlx-community/Lance-3B-bf16

# t2i — photorealistic text-to-image:
HF_HUB_DISABLE_XET=1 uv run python scripts/07_t2i_demo.py \
    --prompt "A photorealistic tabby cat holding a colorful STOP sign." \
    --lance-weights ~/.cache/huggingface/hub/models--mlx-community--Lance-3B-bf16/snapshots/*/ \
    --vae-weights   ~/.cache/huggingface/hub/models--mlx-community--Lance-3B-bf16/snapshots/*/vae.safetensors

# image_edit — instruction-based editing:
HF_HUB_DISABLE_XET=1 uv run python scripts/13_image_edit_demo.py \
    --input-image my_photo.jpg \
    --instruction "Remove the hat from the painting." \
    --lance-weights .../Lance-3B-bf16 --vae-weights .../vae.safetensors

# x2t_image — image VQA:
HF_HUB_DISABLE_XET=1 uv run python scripts/04_x2t_image_demo.py \
    --case 03 \
    --lance-weights .../Lance-3B-bf16 \
    --vit-weights   .../Lance-3B-bf16/vit.safetensors
```

## Schedulers

Two schedulers ship in this repo:

### DPM-Solver++(2M) — ~2.4× faster, 12 steps
```bash
HF_HUB_DISABLE_XET=1 uv run python scripts/08_t2i_demo.py \
    --prompt "A photorealistic tabby cat holding a colorful STOP sign." \
    --lance-weights ~/.cache/huggingface/hub/models--mlx-community--Lance-3B-bf16/snapshots/*/ \
    --vae-weights   ~/.cache/huggingface/hub/models--mlx-community--Lance-3B-bf16/snapshots/*/vae.safetensors \
    --scheduler dpm --steps 12
```

### Default Euler — quality baseline, 30 steps
```bash
HF_HUB_DISABLE_XET=1 uv run python scripts/08_t2i_demo.py \
    --prompt "A photorealistic tabby cat holding a colorful STOP sign." \
    --lance-weights ~/.cache/huggingface/hub/models--mlx-community--Lance-3B-bf16/snapshots/*/ \
    --vae-weights   ~/.cache/huggingface/hub/models--mlx-community--Lance-3B-bf16/snapshots/*/vae.safetensors \
    --scheduler euler --steps 30
```

| Mode | Steps | Generate time | vs baseline |
|---|---|---|---|
| DPM-Solver++(2M) | 12 | ~78s | **2.4×** |
| Default Euler | 30 | ~196s | — |

_Benchmarked on Lance-3B-bf16, 768², seed=42, cfg_scale=4.0, M-series Apple Silicon._

#### Related projects

[`lance-mlx-studio`](https://github.com/ianscrivener/lance-mlx-studio) is a downstream wrapper that adds `mx.compile()` optimisation around the transformer forward pass (`FastTextToImagePipeline`), giving a further ~5% wall-clock reduction on top of the Euler baseline. It is not part of this repo and requires a separate install.

---

See [HANDOFF.md](./HANDOFF.md) for the phased roadmap (start with the **⚠ Verified findings (2026-05-19)** section — it supersedes earlier guesses). Phase 0 parity-oracle capture runbook lives at [Docs/RUNPOD_PHASE0.md](./Docs/RUNPOD_PHASE0.md). Per-phase technical notes in [notes/](./notes/).

## Quick start (after PyPI release)

```bash
uv pip install lance-mlx
# Image generation
lance-mlx generate --task t2i --prompt "..." --weights mlx-community/Lance-3B-bf16
# Image editing
lance-mlx generate --task image_edit --image foo.jpg --instruction "..." --weights mlx-community/Lance-3B-bf16
# Image understanding (VQA)
lance-mlx generate --task x2t_image --image foo.png --prompt "What is this?"
# Video generation (alpha)
lance-mlx generate --task t2v --prompt "..." --weights mlx-community/Lance-3B-Video-bf16
```

## Tasks supported

- `t2i` — text-to-image (768²)
- `t2v` — text-to-video (480p, 12 fps, ≤121 frames)
- `image_edit` — instruction-based image editing
- `video_edit` — instruction-based video editing
- `x2t_image` — image understanding / VQA / captioning
- `x2t_video` — video understanding / VQA / captioning

## Architecture

- **Two expert towers** (`LLM_UND`, `LLM_GEN`), each initialized from Qwen2.5-VL-3B-Instruct, with per-expert FFN, output projection, and QK-norm
- **Modality-deterministic routing:** text + Qwen2.5-VL ViT semantic tokens → `LLM_UND` (autoregressive next-token); Wan2.2 3D causal VAE latent tokens → `LLM_GEN` (flow-matching velocity prediction)
- **MaPE** — modality-aware RoPE with per-modality temporal offset
- **Wan2.2 3D causal VAE** (16× spatial / 4× temporal compression, **48-channel** latent — Lance bundles its own VAE; do NOT use the public 16-ch `wan2.2_vae.safetensors`)
- **Untied LM head**

## Building blocks reused

- [`Blaizzy/mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) for the Qwen2.5-VL ViT and autoregressive decode infrastructure
- [`Blaizzy/mlx-video`](https://github.com/Blaizzy/mlx-video) for the Wan2.2 VAE and flow-matching sampler

## Hardware

- **Minimum:** Apple Silicon Mac with 16 GB unified memory (4-bit quantized image only)
- **Recommended:** 32 GB+ for bf16 image, 64 GB+ for video
- **Reference platform:** M5 Max 128 GB (macOS 26.2+ for Neural Accelerator support)

## Layout

```
.
├── HANDOFF.md                 phased port plan (this is the spec)
├── pyproject.toml             uv-managed
├── src/lance_mlx/
│   ├── __init__.py
│   ├── __main__.py            CLI entry point
│   ├── bench.py               Timer + RunRecord + JSONL logging
│   ├── io.py                  image/video IO + muxing
│   ├── model/
│   │   ├── lance_llm.py       dual-expert MoT backbone
│   │   ├── mape.py            modality-aware RoPE
│   │   ├── flow_head.py       velocity prediction head
│   │   └── routing.py         token modality routing
│   ├── pipeline/
│   │   ├── t2i.py             text-to-image flow loop
│   │   ├── t2v.py             text-to-video flow loop
│   │   ├── image_edit.py
│   │   ├── video_edit.py
│   │   └── understanding.py   x2t_image + x2t_video AR decode
│   └── convert.py             HF → MLX weight conversion
├── scripts/
│   ├── 00_capture_oracle.py   Phase 0 PyTorch reference capture (runs on cloud GPU)
│   ├── 01_inspect_keys.py     Phase 1a weight topology audit
│   ├── 02_convert.py          Phase 1e weight conversion
│   ├── 03_run_understanding.py Phase 2 x2t pipeline
│   ├── 04_run_t2i.py          Phase 3 T2I
│   ├── 05_quantize.py         Phase 5a quantization
│   └── 06_publish_hf.py       Phase 5c HF upload (dry-run default)
├── prompts/
│   ├── t2i_eval.json
│   ├── t2v_eval.json
│   └── understanding_eval.json
├── tests/
│   ├── fixtures/              Phase 0 PyTorch reference outputs
│   ├── test_routing.py
│   ├── test_mape.py
│   ├── test_vae_roundtrip.py
│   └── test_parity_t2i.py
├── notes/                     phase-by-phase educational notes
└── vendor/                    read-only reference clones
```

## License

This MLX port: **Apache 2.0**.

Lance model weights: Apache 2.0 (ByteDance Intelligent Creation Lab).
Wan2.2 VAE: Apache 2.0 (Alibaba).
Qwen2.5-VL: Apache 2.0 (Alibaba).

See `LICENSE` and `NOTICE` for full attribution.

## Citation

```bibtex
@article{fu2026lance,
  title={Lance: Unified Multimodal Modeling by Multi-Task Synergy},
  author={Fu, Fengyi and Huang, Mengqi and Wu, Shaojin and others},
  journal={arXiv preprint arXiv:2605.18678},
  year={2026}
}
```
