# lance-mlx

> **Note:** "Lance" here refers to **ByteDance Intelligent Creation Lab's unified multimodal model** ([paper](https://arxiv.org/abs/2605.18678), [weights](https://huggingface.co/bytedance-research/Lance)), **not** [Lance/LanceDB](https://github.com/lancedb/lance) (the columnar data format).

MLX port of **Lance** for Apple Silicon. Lance is a 3B-active / ~12B-total parameter dual-stream Mixture-of-Transformer-Experts model that unifies image and video understanding, generation, and editing in a single framework. This package brings Lance to Apple Silicon via MLX, with weights hosted on the `mlx-community` HuggingFace organization.

## 📦 Weights on Hugging Face (`mlx-community`)

All three repos live in the **[Lance MLX collection](https://huggingface.co/collections/mlx-community/lance-mlx-6a0f3cd5648a74f8283fc8a4)** for one-click browsing.

| Repo | Status | Use for |
|---|---|---|
| [`mlx-community/Lance-3B-bf16`](https://huggingface.co/mlx-community/Lance-3B-bf16) | 🟢 Production | `t2i`, `image_edit`, `x2t_image` |
| [`mlx-community/Wan2.2-VAE-Lance-bf16`](https://huggingface.co/mlx-community/Wan2.2-VAE-Lance-bf16) | 🟢 Production | 48-ch Wan2.2 VAE (standalone, shared by image + video pipelines) |
| [`mlx-community/Lance-3B-Video-bf16`](https://huggingface.co/mlx-community/Lance-3B-Video-bf16) | 🟢 Functional | `t2v` (painterly aesthetic by design), `x2t_video` + `video_edit` (implemented, validation pending) |

## Status

🟢 **Feature-complete on Apple Silicon as of 2026-05-21 — all 6 Lance task families validated end-to-end.** Image (t2i, image_edit, x2t_image) is production-quality crystal-clear; video (t2v, video_edit, x2t_video) is functional with Lance_3B_Video's intentional painterly aesthetic.

| Capability | Status |
|---|---|
| Convert HF safetensors → MLX bf16 (both checkpoints + Wan2.2 VAE) | ✅ `scripts/02_convert.py`, `scripts/06_convert_wan_vae.py` |
| Load `Lance_3B` + `Lance_3B_Video` into `LanceModel` | ✅ 0 missing keys, dummy forward verified |
| **x2t_image VQA (image → text answer)** | **✅ Production. Content-correct across all 6 oracle cases.** |
| KV cache for fast autoregressive decode | ✅ 1.7×–2.8× speedup on long generations |
| **t2i (text → image generation)** | **✅ Production. Photorealistic, prompt-aligned output.** |
| **image_edit (instruction-based)** | **✅ Production. "Remove hat" preserves identity + style + signature; "Add pearl necklace" leaves rest intact.** |
| **t2v (text → video)** | **✅ Functional across the scale envelope. 17f and 25f at 768² both produce recognizable, prompt-aligned painterly content. [Issue #1](https://github.com/xocialize/lance-mlx/issues/1) closed as prompt-content misinterpretation.** |
| t2v at very high frame counts (≥49f at 768²) | ⚠️ Functional but ~2¼ hr/clip — impractical without KV cache (Phase 5b) |
| **x2t_video (video VQA)** | **✅ Validated against Phase 0 oracle. Cooking video → kitchen+pan+spatula+tomato+meat all content-correct in 17.5 s.** |
| **video_edit (instruction-based)** | **✅ Functional. "Change all the balls to a deep red color." → balls correctly recolored, composition preserved. 17 frames × 256² in 81.6 s.** |
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
