# lance-mlx

> **Note:** "Lance" here refers to **ByteDance Intelligent Creation Lab's unified multimodal model** ([paper](https://arxiv.org/abs/2605.18678), [weights](https://huggingface.co/bytedance-research/Lance)), **not** [Lance/LanceDB](https://github.com/lancedb/lance) (the columnar data format).

MLX port of **Lance** for Apple Silicon. Lance is a 3B-active / ~12B-total parameter dual-stream Mixture-of-Transformer-Experts model that unifies image and video understanding, generation, and editing in a single framework. This package brings Lance to Apple Silicon via MLX, with weights hosted on the `mlx-community` HuggingFace organization.

## Status

🚧 **Pre-alpha — port in progress.** See [HANDOFF.md](./HANDOFF.md) for the phased roadmap (start with the **⚠ Verified findings (2026-05-19)** section — it supersedes earlier guesses). Phase 0 parity-oracle capture runbook lives at [Docs/RUNPOD_PHASE0.md](./Docs/RUNPOD_PHASE0.md).

## Quick start (after Phase 5)

```bash
uv pip install lance-mlx
# Image generation
lance-mlx generate --task t2i --prompt "..." --weights mlx-community/Lance-3B-bf16
# Image understanding (VQA)
lance-mlx generate --task x2t_image --image foo.png --prompt "What is this?"
# Video generation
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
