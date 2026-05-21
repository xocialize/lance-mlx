"""x2t_image / x2t_video — VQA via Lance MLX.

Phase 2 MVP: end-to-end pipeline composing:
  1. mlx-vlm's `Qwen2_5_VLProcessor` (chat template + image preprocessing + tokenizer)
  2. mlx-vlm's `VisionModel` (Qwen2.5-VL ViT, loaded from `vit.safetensors`)
  3. Our `LanceModel` (loaded from `model.safetensors`)
  4. Greedy decode loop (no KV cache — Phase 2.1 follow-up)

Public API: `UnderstandingPipeline.from_pretrained(...).generate(image, question)`.

Position-ID construction (`_compute_position_ids`) is adapted from
mlx-vlm's `LanguageModel.get_rope_index`. The image-grid handling is
load-bearing for mRoPE to encode 2D positions correctly inside images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_vlm.models.qwen2_5_vl.config import TextConfig, VisionConfig
from mlx_vlm.models.qwen2_5_vl.vision import VisionModel
from PIL import Image

from lance_mlx.model import LanceModel
from lance_mlx.model.routing import PositionGroup


# ---------------------------------------------------------------------------
# Position-ID construction (adapted from mlx-vlm get_rope_index)
# ---------------------------------------------------------------------------

def _compute_position_ids(
    input_ids: mx.array,                         # (B, T)
    image_grid_thw: Optional[mx.array],          # (n_visual, 3) or None
    spatial_merge_size: int,
    image_token_id: int,                          # placeholder used in input_ids
    video_token_id: int,
    vision_start_token_id: int,
    attention_mask: Optional[mx.array] = None,
) -> tuple[mx.array, mx.array]:
    """3D position IDs for mRoPE with visual-grid handling.

    Adapted near-verbatim from mlx-vlm's
    `Qwen2_5_VLLanguageModel.get_rope_index`. Returns
    `(position_ids: (3, B, T), mrope_position_deltas: (B, 1))`.

    NOTE: `image_grid_thw` is really "visual grid_thw" — the same (t, h, w)
    layout works for both images (t=1) and videos (t>1). For x2t_video, pass
    the video's grid via this slot and pass `video_token_id` as
    `image_token_id` (the placeholder token id in the input sequence).
    """
    batch_size, seq_length = input_ids.shape

    if image_grid_thw is None and video_grid_thw is None:
        # Text-only fast path.
        if attention_mask is not None:
            position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
            position_ids = mx.where(
                attention_mask == 0, mx.ones_like(position_ids), position_ids
            )
            max_position_ids = position_ids.max(axis=-1, keepdims=True)
            position_ids = mx.broadcast_to(position_ids[None, :, :], (3, *position_ids.shape))
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = mx.arange(seq_length).reshape(1, -1)
            position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_length))
            mrope_position_deltas = mx.zeros([batch_size, 1], dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas

    # Image-grid path (adapted from mlx-vlm).
    if attention_mask is None:
        attention_mask = mx.ones_like(input_ids)
    position_ids = mx.ones((3, batch_size, seq_length), dtype=input_ids.dtype)
    image_index = 0
    mrope_position_deltas = []

    for i in range(batch_size):
        ids = mx.where(attention_mask[i] == 1, input_ids[i], mx.zeros_like(input_ids[i]))
        vision_start_indices = mx.sum(
            mx.where(
                ids == vision_start_token_id,
                mx.arange(ids.shape[0]),
                mx.zeros_like(ids),
            )
        )
        vision_tokens = ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum().item()
        input_tokens = ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images = image_nums

        for _ in range(image_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed = input_tokens.index(image_token_id, st)
            else:
                ed = len(input_tokens) + 1
            t, h, w = (
                image_grid_thw[image_index][0],
                image_grid_thw[image_index][1],
                image_grid_thw[image_index][2],
            )
            image_index += 1
            remain_images -= 1
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st
            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if len(llm_pos_ids_list) > 0 else 0
            )
            index = mx.arange(text_len).reshape(1, text_len)
            index = mx.broadcast_to(index, (3, text_len)) + st_idx
            llm_pos_ids_list.append(index)

            # 3D image grid (t, h, w)
            t_index = mx.arange(llm_grid_t).reshape(llm_grid_t, 1)
            t_index = mx.broadcast_to(t_index, (llm_grid_t, llm_grid_h * llm_grid_w)).flatten()
            h_index = mx.arange(llm_grid_h).reshape(1, llm_grid_h, 1)
            h_index = mx.broadcast_to(h_index, (llm_grid_t, llm_grid_h, llm_grid_w)).flatten()
            w_index = mx.arange(llm_grid_w).reshape(1, 1, llm_grid_w)
            w_index = mx.broadcast_to(w_index, (llm_grid_t, llm_grid_h, llm_grid_w)).flatten()
            llm_pos_ids_list.append(
                mx.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if len(llm_pos_ids_list) > 0 else 0
            )
            text_len = len(input_tokens) - st
            t_index = mx.arange(text_len).reshape(1, text_len)
            t_index = mx.broadcast_to(t_index, (3, text_len))
            llm_pos_ids_list.append(t_index + st_idx)

        llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mask = mx.array(attention_mask[i] == 1)
        expanded_mask = mx.expand_dims(mask, axis=0)
        expanded_mask = mx.broadcast_to(expanded_mask, (3, 1, mask.shape[0]))
        expanded_positions = mx.expand_dims(llm_positions, axis=1)
        new_positions = mx.where(
            expanded_mask, expanded_positions, position_ids[:, i:i+1, :]
        )
        updated_position_ids = mx.concatenate(
            [position_ids[:, :i, :], new_positions, position_ids[:, i+1:, :]],
            axis=1,
        )
        position_ids = updated_position_ids
        mrope_position_deltas.append(llm_positions.max() + 1 - len(input_tokens))

    mrope_position_deltas = mx.array(mrope_position_deltas).reshape(-1, 1)
    return position_ids, mrope_position_deltas


# ---------------------------------------------------------------------------
# Inputs-embeds assembly (text + ViT features merged at image-pad positions)
# ---------------------------------------------------------------------------

def _merge_text_embeds_and_image_features(
    text_embeds: mx.array,        # (B, T, D)
    image_features: mx.array,     # (N_post_merger, D)
    input_ids: mx.array,          # (B, T)
    image_token_id: int,
    video_token_id: int,
) -> mx.array:
    """Replace image-pad token embeddings with ViT features.

    Mirrors mlx-vlm's `Model.merge_input_ids_with_image_features`.
    Returns `(B, T, D)` with ViT features slotted at image-pad positions.
    """
    image_positions = input_ids == image_token_id
    if mx.sum(image_positions).item() == 0:
        image_positions = input_ids == video_token_id

    B, T = input_ids.shape
    batch_outputs = []
    feature_start_idx = 0

    for b in range(B):
        mask = image_positions[b]
        n = mx.sum(mask).item()
        if n > 0:
            features = image_features[feature_start_idx : feature_start_idx + n]
            cumsum = mx.cumsum(mask.astype(mx.int32))
            feature_indices = mx.where(mask, cumsum - 1, 0)
            gathered = features[feature_indices]
            mask_expanded = mx.expand_dims(mask, axis=-1)
            out = mx.where(mask_expanded, gathered, text_embeds[b])
            feature_start_idx += n
        else:
            out = text_embeds[b]
        batch_outputs.append(out)

    return mx.stack(batch_outputs, axis=0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class UnderstandingPipeline:
    """x2t_image / x2t_video VQA via Lance MLX.

    Loads:
      - mlx-vlm's `Qwen2_5_VLProcessor` from a HF model repo (small download)
      - mlx-vlm's `VisionModel` from a local `vit.safetensors`
      - Our `LanceModel` from a local `model.safetensors`

    Public method `generate(image, question)` returns the decoded answer string.
    """

    def __init__(
        self,
        lance_model: LanceModel,
        vision_model: VisionModel,
        processor,                     # Qwen2_5_VLProcessor or similar
        text_config: TextConfig,
        vision_config: VisionConfig,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        eos_token_ids: list[int],     # Lance has TWO: <|im_end|> + <|endoftext|>
        endoftext_token_id: int,
    ):
        self.lance_model = lance_model
        self.vision_model = vision_model
        self.processor = processor
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.eos_token_ids = eos_token_ids
        self.endoftext_token_id = endoftext_token_id
        # Single value kept for backwards compat / readability in callsites.
        self.eos_token_id = eos_token_ids[0]

    @classmethod
    def from_pretrained(
        cls,
        lance_weights_dir: Path | str,
        vit_safetensors: Path | str,
        hf_processor_repo: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ) -> "UnderstandingPipeline":
        """Load all three components.

        Args:
            lance_weights_dir: dir produced by `scripts/02_convert.py` for the
                LLM (contains `model.safetensors` + `config.json`).
            vit_safetensors: path to `vit.safetensors` (typically the bundled
                one from `Lance-3B-Video-bf16/`).
            hf_processor_repo: HF repo to fetch tokenizer + image processor
                + chat template from. Defaults to stock Qwen2.5-VL-3B.
        """
        lance_weights_dir = Path(lance_weights_dir)
        vit_safetensors = Path(vit_safetensors)

        # 1. Processor (tokenizer + image preprocessor + chat template).
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(hf_processor_repo)
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        endoftext_id = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        # Lance's generation_config.json declares BOTH 151645 (<|im_end|>) and
        # 151643 (<|endoftext|>) as stop tokens. Honor both.
        eos_token_ids = [im_end_id, endoftext_id]

        # 2. LanceModel from converter output.
        cfg = json.loads((lance_weights_dir / "config.json").read_text())
        text_cfg = TextConfig(
            model_type=cfg["model_type"],
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            intermediate_size=cfg["intermediate_size"],
            num_attention_heads=cfg["num_attention_heads"],
            rms_norm_eps=cfg["rms_norm_eps"],
            vocab_size=cfg["vocab_size"],
            num_key_value_heads=cfg.get("num_key_value_heads"),
            max_position_embeddings=cfg.get("max_position_embeddings", 128000),
            rope_theta=cfg.get("rope_theta", 1e6),
            rope_scaling=cfg.get("rope_scaling"),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        )
        saved_lance = mx.load(str(lance_weights_dir / "model.safetensors"))
        num_latent_positions = saved_lance["latent_pos_embed.pos_embed"].shape[0]
        lance_model = LanceModel(text_cfg, num_latent_positions=num_latent_positions)
        lance_model.load_weights(list(saved_lance.items()))
        mx.eval(lance_model.parameters())

        # 3. VisionModel from local vit.safetensors. mlx-vlm's `sanitize()`
        #    handles the patch_embed.proj.weight 5D transpose.
        # HF and mlx-vlm differ in some config field names; filter + rename.
        import inspect as _inspect
        _vc_fields = set(_inspect.signature(VisionConfig).parameters)
        _hf_vision = dict(cfg["vision_config"])
        # HF uses "in_chans"; mlx-vlm uses "in_channels".
        if "in_chans" in _hf_vision and "in_channels" not in _hf_vision:
            _hf_vision["in_channels"] = _hf_vision.pop("in_chans")
        # Drop fields mlx-vlm doesn't model (e.g. hidden_act — implicit silu).
        _vision_kwargs = {k: v for k, v in _hf_vision.items() if k in _vc_fields}
        _vision_kwargs.setdefault("model_type", "qwen2_5_vl")
        vision_cfg = VisionConfig(**_vision_kwargs)
        vision_model = VisionModel(vision_cfg)
        saved_vit = mx.load(str(vit_safetensors))
        saved_vit = vision_model.sanitize(saved_vit)
        vision_model.load_weights(list(saved_vit.items()))
        mx.eval(vision_model.parameters())

        return cls(
            lance_model=lance_model,
            vision_model=vision_model,
            processor=processor,
            text_config=text_cfg,
            vision_config=vision_cfg,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            eos_token_ids=eos_token_ids,
            endoftext_token_id=endoftext_id,
        )

    # --------- generation -------------

    def generate(
        self,
        image: Image.Image,
        question: str,
        *,
        max_new_tokens: int = 256,
        verbose: bool = False,
        use_cache: bool = True,
        prompt_style: str = "lance",  # "lance" or "qwen_stock"
        instruction: str = "Look at the image carefully and answer the question.",
    ) -> str:
        """Greedy-decode an answer to `question` about `image`.

        Args:
            image: PIL image.
            question: question text.
            max_new_tokens: hard cap on generated tokens.
            verbose: print per-step token info.
            use_cache: enable KV cache (default True). Set False to run the
                slower full-recompute path — useful as a parity baseline.
            prompt_style: "lance" reproduces upstream Lance's prompt format
                (instruction as system prompt, image rendered as
                `<|vision_start|><|video_pad|><|vision_end|>` per their
                `system_prompt_render.py`). "qwen_stock" uses
                AutoProcessor.apply_chat_template — the standard Qwen2.5-VL
                template with `<|image_pad|>` and a generic system prompt.
                Lance was trained against the "lance" format; expect better
                parity against the Phase 0 oracle there.
            instruction: the system-prompt instruction Lance was trained on.
                Default matches `config/examples/x2t_image_example.json`.

        Returns the decoded answer text (without the chat-template suffix).
        """
        # 1-6. Preprocess (shared between cached and non-cached paths).
        prompt_state = self._prepare_prompt(
            image, question, prompt_style=prompt_style,
            instruction=instruction, verbose=verbose,
        )
        input_ids = prompt_state["input_ids"]
        inputs_embeds = prompt_state["inputs_embeds"]
        position_ids = prompt_state["position_ids"]
        position_group = prompt_state["position_group"]

        # 7. Greedy decode loop.
        if use_cache:
            generated_ids = self._decode_with_cache(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                position_group=position_group,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
        else:
            generated_ids = self._decode_no_cache(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                position_ids=position_ids,
                position_group=position_group,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )

        # 8. Decode.
        return self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ---- x2t_video --------------------------------------------------------

    def generate_video(
        self,
        video,                                       # Path/str (.mp4) or list[PIL.Image]
        question: str,
        *,
        num_sample_frames: int = 16,
        target_h: int = 224,
        target_w: int = 224,
        max_new_tokens: int = 256,
        verbose: bool = False,
        use_cache: bool = True,
        prompt_style: str = "lance",
        instruction: str = "Look at the video carefully and answer the question.",
    ) -> str:
        """Greedy-decode an answer to `question` about `video`.

        Args:
            video: Path/str to an MP4 file, or list of PIL frames already
                decoded. If MP4, we evenly sample `num_sample_frames` frames.
            question: question text.
            num_sample_frames: number of frames to sample from the video.
                MUST be even (Qwen2.5-VL ViT temporal_patch_size=2 pairs
                adjacent frames into one temporal patch).
            target_h, target_w: spatial resize target. Defaults to 224×224
                which keeps the ViT's grid small for memory; bump to 336+ if
                fine spatial detail matters for the question.
            max_new_tokens / verbose / use_cache / prompt_style / instruction:
                same semantics as `generate`.
        """
        if num_sample_frames % 2 != 0:
            raise ValueError(
                f"num_sample_frames must be even (got {num_sample_frames}); "
                "Qwen2.5-VL temporal_patch_size=2 pairs adjacent frames."
            )

        # 1. Decode video → (T, H, W, 3) uint8 frames.
        if isinstance(video, (str, Path)):
            import imageio.v3 as iio
            all_frames = [f for f in iio.imiter(str(video))]
            n_total = len(all_frames)
            if n_total == 0:
                raise ValueError(f"video {video} produced 0 frames")
            import numpy as np
            idx = np.linspace(0, n_total - 1, num_sample_frames).astype(int)
            sampled = [all_frames[i] for i in idx]
            if verbose:
                print(f"  sampled {num_sample_frames}/{n_total} frames "
                      f"(indices {idx[:4].tolist()}..{idx[-2:].tolist()})")
        else:
            sampled = list(video)
            if len(sampled) != num_sample_frames:
                raise ValueError(
                    f"video frame list has {len(sampled)} frames; expected "
                    f"{num_sample_frames}. Pass a path or a pre-sampled list."
                )

        # 2. Resize each frame to target_h × target_w.
        pil_frames = []
        for f in sampled:
            if isinstance(f, Image.Image):
                im = f.convert("RGB")
            else:
                import numpy as np
                im = Image.fromarray(np.asarray(f)).convert("RGB")
            im = im.resize((target_w, target_h), Image.LANCZOS)
            pil_frames.append(im)

        # 3. Build prompt with Lance's video placeholder
        #    (<|vision_start|><|video_pad|><|vision_end|>{question}).
        if prompt_style == "lance":
            text = (
                f"<|im_start|>system\n{instruction}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"<|vision_start|><|video_pad|><|vision_end|>{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        elif prompt_style == "qwen_stock":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            raise ValueError(f"unknown prompt_style: {prompt_style!r}")

        # 4. Run processor with videos= (expands video_pad based on
        #    video_grid_thw).
        import numpy as np
        # Qwen video processor expects channels-first (T, C, H, W).
        video_np = np.stack([
            np.asarray(im).transpose(2, 0, 1) for im in pil_frames
        ])                                                          # (T, 3, H, W)
        inputs = self.processor(
            videos=[video_np], text=text, return_tensors="mlx",
        )
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values_videos"]
        video_grid_thw = inputs["video_grid_thw"]

        if verbose:
            print(f"  prompt tokens after expansion: {input_ids.shape[-1]}")
            print(f"  video_grid_thw: {video_grid_thw.tolist()}  "
                  f"(t={video_grid_thw[0,0].item()}, "
                  f"h={video_grid_thw[0,1].item()}, "
                  f"w={video_grid_thw[0,2].item()})")
            print(f"  pixel_values_videos shape: {pixel_values.shape}")

        # 5. ViT forward in video mode (same module — Qwen2.5-VL ViT handles
        #    (t, h, w) grids natively via temporal_patch_size=2).
        vit_dtype = self.vision_model.patch_embed.proj.weight.dtype
        video_features = self.vision_model(
            pixel_values.astype(vit_dtype), video_grid_thw,
        )
        if verbose:
            print(f"  vit features: {video_features.shape}")

        # 6. Merge text embeds + video features.
        text_embeds = self.lance_model.embed_tokens(input_ids)
        inputs_embeds = _merge_text_embeds_and_image_features(
            text_embeds, video_features, input_ids,
            self.video_token_id, self.image_token_id,
        )

        # 7. Position IDs — same routine, pass video's grid_thw via the
        #    generic visual-grid slot, and use video_token_id as the
        #    placeholder marker.
        position_ids, _ = _compute_position_ids(
            input_ids, video_grid_thw,
            spatial_merge_size=self.vision_config.spatial_merge_size,
            image_token_id=self.video_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
        )

        # 8. Position group: all UND for VQA.
        T = input_ids.shape[1]
        position_group = mx.full((T,), int(PositionGroup.TEXT), dtype=mx.int32)

        # 9. Decode.
        if use_cache:
            generated_ids = self._decode_with_cache(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                position_group=position_group,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
        else:
            generated_ids = self._decode_no_cache(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                position_ids=position_ids,
                position_group=position_group,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
        return self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ---- shared prompt preparation ----------------------------------------

    def _prepare_prompt(
        self,
        image,
        question,
        *,
        prompt_style: str = "lance",
        instruction: str = "Look at the image carefully and answer the question.",
        verbose: bool,
    ) -> dict:
        """Steps 1-6 of generate: build prompt, ViT forward, merge, position IDs.

        Returns dict with: input_ids, inputs_embeds, position_ids, position_group.
        Shared between cached/uncached decode paths so both see byte-identical
        inputs to the layer stack (eliminates a class of parity-test confounders).
        """
        # 1. Build the chat-templated prompt with image placeholder.
        #    Both styles write <|image_pad|> in the template so the processor
        #    expands it into N tokens (its expansion logic only fires on the
        #    processor.image_token marker, not video_pad). After tokenization
        #    we optionally substitute video_pad for image_pad in input_ids to
        #    match Lance's training convention.
        if prompt_style == "lance":
            # Reproduce upstream `data/system_prompt_render.py`:
            #   - instruction as system prompt
            #   - image rendered as <|vision_start|><|image_pad|><|vision_end|>
            #     here, but post-tokenization we substitute the image_pad IDs
            #     with video_pad IDs because Lance's training data renders
            #     images as <|video_pad|> by default (system_prompt_render.py
            #     line 175: `parts.append("<|vision_start|><|video_pad|><|vision_end|>")`
            #     when `force_video_pad=False`, which is the default).
            text = (
                f"<|im_start|>system\n{instruction}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            # After tokenization we'll replace image_pad IDs with video_pad IDs.
            substitute_image_pad_for_video_pad = True
            image_placeholder_token_id = self.video_token_id
        elif prompt_style == "qwen_stock":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            substitute_image_pad_for_video_pad = False
            image_placeholder_token_id = self.image_token_id
        else:
            raise ValueError(f"unknown prompt_style: {prompt_style!r}")

        if verbose:
            print(f"  prompt_style: {prompt_style}")
            print(f"  templated prompt:\n{text}")
            print(f"  image placeholder token id: {image_placeholder_token_id}")
            print(f"  substitute image_pad → video_pad: {substitute_image_pad_for_video_pad}")

        # 2. Preprocess. The processor expands <|image_pad|> into N copies
        #    based on image_grid_thw. Always works with <|image_pad|> in the
        #    text; substitution happens after.
        inputs = self.processor(images=image, text=text, return_tensors="mlx")
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        if substitute_image_pad_for_video_pad:
            # Lance training data feeds video_pad tokens at image positions.
            input_ids = mx.where(
                input_ids == self.image_token_id,
                mx.array(self.video_token_id, dtype=input_ids.dtype),
                input_ids,
            )

        if verbose:
            print(f"  prompt tokens after expansion: {input_ids.shape[-1]}")
            print(f"  image_grid_thw: {image_grid_thw.tolist()}")

        # 3. ViT forward.
        vit_dtype = self.vision_model.patch_embed.proj.weight.dtype
        image_features = self.vision_model(
            pixel_values.astype(vit_dtype), image_grid_thw,
        )
        if verbose:
            print(f"  vit features: {image_features.shape}")

        # 4. Merge text embeds + ViT features. Look up the placeholder by
        #    the specific token id that was used in the templated prompt.
        text_embeds = self.lance_model.embed_tokens(input_ids)
        inputs_embeds = _merge_text_embeds_and_image_features(
            text_embeds, image_features, input_ids,
            image_placeholder_token_id,
            # Second fallback id is unused when first matches:
            self.video_token_id if image_placeholder_token_id != self.video_token_id
                else self.image_token_id,
        )

        # 5. Position IDs. _compute_position_ids identifies images by the
        #    image_token_id arg — pass whichever id was used as the
        #    placeholder so the image grid is recognized.
        position_ids, _ = _compute_position_ids(
            input_ids, image_grid_thw,
            spatial_merge_size=self.vision_config.spatial_merge_size,
            image_token_id=image_placeholder_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
        )

        # 6. Position group: all UND for VQA.
        T = input_ids.shape[1]
        position_group = mx.full((T,), int(PositionGroup.TEXT), dtype=mx.int32)

        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "position_group": position_group,
        }

    # ---- KV-cached decode path ---------------------------------------------

    def _decode_with_cache(
        self,
        *,
        inputs_embeds: mx.array,
        position_ids: mx.array,
        position_group: mx.array,
        max_new_tokens: int,
        verbose: bool,
    ) -> list[int]:
        """Prefill the full prompt to populate KV caches, then iterate one
        token at a time using the cache.

        Per-layer cache: one `mlx_lm.models.cache.KVCache` instance per
        LanceMoTLayer. The cache stores already-routed K/V (i.e. the output
        of the per-token UND/GEN projection merge), so resuming on new
        tokens just appends without re-running routing on cached positions.
        """
        n_layers = len(self.lance_model.layers)
        cache = [KVCache() for _ in range(n_layers)]

        # Prefill — run the full prompt through the stack with empty caches.
        h = self.lance_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            position_group=position_group,
            cache=cache,
        )
        logits = self.lance_model.lm_head(h[:, -1:, :])
        next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
        generated_ids: list[int] = [next_token]

        if verbose:
            tok_str = self.processor.tokenizer.decode([next_token])
            print(f"  step 0 (prefill): token {next_token} ({tok_str!r})")

        if next_token in self.eos_token_ids:
            return generated_ids

        # Track the trailing 3D position so we can extend by +1 each step.
        # For text positions after an image, all 3 axes carry the same value,
        # so taking [-1:] from any axis gives the right next-step base.
        last_pos = position_ids[:, :, -1:]  # (3, 1, 1)
        single_pg = mx.array([int(PositionGroup.TEXT)], dtype=mx.int32)

        # Decode steps — one token at a time, cache grows by 1 each step.
        for step in range(1, max_new_tokens):
            next_embed = self.lance_model.embed_tokens(
                mx.array([[next_token]], dtype=mx.int32)
            )
            next_pos = last_pos + 1  # (3, 1, 1)

            h = self.lance_model(
                inputs_embeds=next_embed,
                position_ids=next_pos,
                position_group=single_pg,
                cache=cache,
            )
            logits = self.lance_model.lm_head(h[:, -1:, :])
            next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
            generated_ids.append(next_token)

            if verbose and step < 10:
                tok_str = self.processor.tokenizer.decode([next_token])
                print(f"  step {step}: token {next_token} ({tok_str!r})")

            if next_token in self.eos_token_ids:
                break

            last_pos = next_pos

        return generated_ids

    # ---- No-cache decode path (parity baseline) ----------------------------

    def _decode_no_cache(
        self,
        *,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        position_ids: mx.array,
        position_group: mx.array,
        max_new_tokens: int,
        verbose: bool,
    ) -> list[int]:
        """Full re-forward each step. Slower but no cache state; useful as
        a parity baseline against the cached path."""
        generated_ids: list[int] = []
        for step in range(max_new_tokens):
            h = self.lance_model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                position_group=position_group,
            )
            logits = self.lance_model.lm_head(h[:, -1:, :])
            next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
            generated_ids.append(next_token)

            if verbose and step < 10:
                tok_str = self.processor.tokenizer.decode([next_token])
                print(f"  step {step}: token {next_token} ({tok_str!r})")

            if next_token in self.eos_token_ids:
                break

            next_embed = self.lance_model.embed_tokens(
                mx.array([[next_token]], dtype=input_ids.dtype)
            )
            inputs_embeds = mx.concatenate([inputs_embeds, next_embed], axis=1)
            input_ids = mx.concatenate(
                [input_ids, mx.array([[next_token]], dtype=input_ids.dtype)], axis=1,
            )
            last_pos = position_ids[:, :, -1:]
            position_ids = mx.concatenate([position_ids, last_pos + 1], axis=2)
            position_group = mx.concatenate(
                [position_group, mx.array([int(PositionGroup.TEXT)], dtype=mx.int32)], axis=0,
            )

        return generated_ids
