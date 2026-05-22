"""t2v — text-to-video generation via Lance_3B_Video + Wan2.2 VAE.

Phase 4a MVP. Extends the Phase 3e t2i pipeline pattern with:

  - Lance_3B_Video LLM checkpoint (126976-entry latent_pos_embed table
    covering 31 temporal × 64×64 spatial positions).
  - 3D latent grid (T_lat, h_lat, w_lat) instead of (1, h_lat, w_lat).
  - 3D position-ID construction: t-axis varies per frame.
  - MaPE re-anchor to 2000 (modality 3 = video_gen) instead of 1000.
  - MP4 output via imageio-ffmpeg.

All other Phase 3 learnings carry over:
  - System prompt = Lance t2v instruction (`generate_system_prompt('t2v', 'video')`)
  - Image-as-video convention: `<|video_pad|>` for latent placeholder
  - Timestep embed added ONLY at VAE positions (Phase 3d fix)
  - Custom mask: causal OR bidirectional-within-latent-block (Phase 3e fix)
  - CFG with renormalization (`cfg_renorm_type='global'`)

VAE compression for Wan2.2 (per HANDOFF):
  - Temporal 4× (50 frames → 13 latent frames, via T_lat = (T-1)//4 + 1)
  - Spatial 16× (768x768 → 48x48 latent)
  - 48-channel latent (Lance bundled VAE z_dim=48)

For 768x768 × 50 frames: 13 × 48 × 48 = 29952 latent tokens. Very large.
This MVP uses smaller dims (256x256 × 16 frames → 1280 latent tokens) for
fast iteration before scaling up.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_video.models.wan_2.vae22 import (
    Wan22VAEDecoder,
    denormalize_latents,
)
from mlx_vlm.models.qwen2_5_vl.config import TextConfig

from lance_mlx.model import LanceModel
from lance_mlx.model.flow_head import timestep_schedule
from lance_mlx.model.routing import PositionGroup


# Upstream Lance's t2v system-prompt instruction
# (from data/common.py::generate_system_prompt('t2v', 'video')).
T2V_INSTRUCTION = (
    "Describe the video by detailing the color, quantity, visible text, "
    "shape, size, texture, spatial relationships and motion/camera "
    "movements of the objects and background:"
)

# MaPE anchor for video_gen (modality 3) per upstream `shift_position_ids`.
MAPE_ANCHOR_VIDEO_GEN = 2000

# VAE constants for Lance's bundled Wan2.2 VAE.
VAE_LATENT_CHANNELS = 48
VAE_SPATIAL_DOWNSAMPLE = 16
VAE_TEMPORAL_DOWNSAMPLE = 4    # First chunk = 1 frame, rest = 4 frames each

# Lance_3B_Video latent_pos_embed table dims (per Phase 1a inspection).
MAX_LATENT_SIDE = 64                   # spatial max per axis
MAX_LATENT_FRAMES = 31                 # temporal max
MAX_NUM_LATENT_POSITIONS = MAX_LATENT_FRAMES * MAX_LATENT_SIDE * MAX_LATENT_SIDE   # = 126976


class TextToVideoPipeline:
    """Lance t2v — text prompt → MP4 video via flow-matching."""

    def __init__(
        self,
        lance_model: LanceModel,
        vae_decoder: Wan22VAEDecoder,
        processor,
        text_config: TextConfig,
        image_pad_token_id: int,
        video_pad_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
    ):
        self.lance_model = lance_model
        self.vae_decoder = vae_decoder
        self.processor = processor
        self.text_config = text_config
        self.image_pad_token_id = image_pad_token_id
        self.video_pad_token_id = video_pad_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

    @classmethod
    def from_pretrained(
        cls,
        lance_weights_dir: Path | str,
        vae_safetensors: Path | str,
        hf_processor_repo: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ) -> "TextToVideoPipeline":
        """Loads Lance_3B_Video LLM + Wan2.2 VAE decoder + tokenizer."""
        lance_weights_dir = Path(lance_weights_dir)
        vae_safetensors = Path(vae_safetensors)

        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(hf_processor_repo)
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        # LanceModel (must be Lance_3B_Video for video generation — the
        # latent_pos_embed table needs 126976 entries for temporal coverage).
        # Quantization-aware loader applies nn.quantize if config says so.
        from lance_mlx.model._loader import build_text_config, load_lance_model
        cfg = json.loads((lance_weights_dir / "config.json").read_text())
        text_cfg = build_text_config(cfg)
        # Peek at latent_pos_embed shape for the warning, then load.
        _saved_peek = mx.load(str(lance_weights_dir / "model.safetensors"))
        num_latent_positions = _saved_peek["latent_pos_embed.pos_embed"].shape[0]
        del _saved_peek
        if num_latent_positions != MAX_NUM_LATENT_POSITIONS:
            print(f"WARNING: latent_pos_embed has {num_latent_positions} entries; "
                  f"video pipeline expects {MAX_NUM_LATENT_POSITIONS} (= 31×64×64). "
                  f"Are you using Lance_3B_Video weights?")
        lance_model = load_lance_model(lance_weights_dir)

        vae_decoder = Wan22VAEDecoder(z_dim=VAE_LATENT_CHANNELS, dim=160, dec_dim=256)
        saved_vae = mx.load(str(vae_safetensors))
        dec_state = {
            k: v for k, v in saved_vae.items()
            if k.startswith("decoder.") or k.startswith("conv2.")
        }
        vae_decoder.load_weights(list(dec_state.items()))
        mx.eval(vae_decoder.parameters())

        return cls(
            lance_model=lance_model,
            vae_decoder=vae_decoder,
            processor=processor,
            text_config=text_cfg,
            image_pad_token_id=image_pad_id,
            video_pad_token_id=video_pad_id,
            vision_start_token_id=vision_start_id,
            vision_end_token_id=vision_end_id,
        )

    # ------------------------------------------------------------------ generate

    def generate(
        self,
        prompt: str,
        *,
        num_frames: int = 16,                # 50 per Lance default; smaller for MVP
        height: int = 256,                   # 768 per Lance default; smaller for MVP
        width: int = 256,
        num_steps: int = 30,
        timestep_shift: float = 3.5,
        cfg_scale: float = 4.0,
        cfg_renorm_type: str = "global",
        cfg_renorm_min: float = 0.0,
        cfg_interval: tuple[float, float] | None = None,
        seed: int = 42,
        verbose: bool = False,
        instruction: str = T2V_INSTRUCTION,
        mape_anchor: int | None = None,
        cfg_uncond_mode: str = "empty_prompt",
        spatial_merge_size: int = 1,
        rope_fp32: bool = False,
        prompt_format: str = "ours",
        latent_pos_base: int | None = 0,
    ) -> mx.array:
        """`mape_anchor`: temporal-anchor value for latent t-axis positions.
        **Default changed to None on 2026-05-21** after Phase 5d scale bisect
        (github issue #2) showed that no-shift produces photorealistic
        prompt-aligned output at every practical scale (256² to 768²×13f,
        n_lat ≤ 9216) where the old default (2000) produced painterly
        smearing. The shift was a port-side deviation from upstream
        `shift_position_ids` (whose gate never fires for pure t2v).
        Pass `mape_anchor=2000` to restore legacy behavior. At very high
        n_lat (≥ ~12k, e.g. 768²×17f or larger) outputs may degrade —
        coherence threshold is around n_lat=11,520. The 768²×50f oracle
        scale still has a separate second bug under investigation.

        `cfg_interval`: (lo, hi) tuple — CFG fires only when `lo < t <= hi`,
        else falls to cfg_scale=1.0 (no CFG) for that step. Upstream Lance
        default per `config_factory.py` is `[0.4, 1.0]`. Pass None to apply
        CFG at every step (legacy MLX port behavior — likely a contributor
        to the painterly aesthetic bug per github issue #2 Candidate 1b).

        `cfg_uncond_mode`: 'empty_prompt' (legacy) feeds the full chat-template
        sequence with `prompt=''` through the LLM for the uncond branch.
        'no_text' matches upstream Lance (per `lance_lance.py:627-630` and
        `uncond_forward`): the uncond branch DROPS all text positions and
        feeds only the latent block through the LLM. Upstream's CFG direction
        is "with text vs no text at all" rather than "with prompt vs empty
        prompt"; the latter under-amplifies fine-detail features. **Candidate 3
        in issue #2.**

        `spatial_merge_size`: divisor for h/w axes in `_build_position_ids`'s
        latent grid. Default `1` (legacy). Set to `2` to match upstream Lance's
        `data/common.py::shift_position_ids` and RockTalk's parallel MLX port
        (both divide visual position-ids by spatial_merge_size=2). **P0b
        candidate from issue #2 / Phase 5g research brief.**

        `rope_fp32`: when True, compute cos/sin and the
        `q*cos + rotate_half(q)*sin` rotation in fp32 across all 36 attention
        layers (mlx-vlm's stock path casts cos/sin to bf16 at
        `qwen2_5_vl/language.py:73` before the rotation). Default False
        (legacy bf16 path). **P0a candidate from issue #2 / Phase 5g.**

        `latent_pos_base`: anchor (origin) for the latent block's (t, h, w)
        mrope grid coords. **Default 0 (Phase 5j fix, 2026-05-21):** latent
        grid always starts at origin regardless of prompt length, matching
        Qwen2.5-VL's training convention where visual tokens use 3D-mrope
        grid origin (not concatenated with text positions). The Phase 5i.2
        bisect proved long verbose prompts trigger watercolor while short
        prompts produce sharp output at the same other config — the trigger
        was prompt-length-dependent drift of latent block position-IDs.
        Pass `None` to restore legacy `base=text_len_before_latents`
        behavior (watercolor on long prompts). Phase 5j A/B at 256²×17f
        on the red-panda-surfing oracle prompt: legacy = watercolor,
        base=0 = PHOTOREAL. The fix that closes the painterly aesthetic gap.
        """
        if cfg_interval is None:
            # Legacy behavior: CFG at every step. Effectively cfg_interval=[-inf, +inf].
            cfg_lo, cfg_hi = float("-inf"), float("inf")
        else:
            cfg_lo, cfg_hi = float(cfg_interval[0]), float(cfg_interval[1])

        # P0a (issue #2 / Phase 5g) — fp32 RoPE rotation in all 36 attention layers.
        # Default off (legacy bf16 path); set True to test the research-brief
        # candidate that bf16 rotation perturbs flow-matching velocity precision.
        self.lance_model.set_rope_fp32(bool(rope_fp32))
        """Generate a video as (T_decoded, H, W, 3) uint8-compatible mx.array.

        Caller is responsible for encoding to MP4 (see scripts/10_t2v_demo.py
        for the imageio-ffmpeg path).
        """
        assert height % VAE_SPATIAL_DOWNSAMPLE == 0
        assert width % VAE_SPATIAL_DOWNSAMPLE == 0
        h_lat = height // VAE_SPATIAL_DOWNSAMPLE
        w_lat = width // VAE_SPATIAL_DOWNSAMPLE
        # Wan2.2 VAE temporal compression: T frames → ((T-1)//4 + 1) latent frames.
        # First chunk = 1 frame, remaining chunks of 4. Inverse on decode side.
        t_lat = (num_frames - 1) // VAE_TEMPORAL_DOWNSAMPLE + 1
        n_lat = t_lat * h_lat * w_lat

        assert t_lat <= MAX_LATENT_FRAMES, (
            f"t_lat={t_lat} exceeds MAX_LATENT_FRAMES={MAX_LATENT_FRAMES}"
        )

        if verbose:
            print(f"  video dims: {num_frames}f × {height}×{width}")
            print(f"  latent dims: {t_lat}f × {h_lat}×{w_lat} = {n_lat} tokens")

        # --- Build per-prompt states (cond + uncond) --------------------
        cond_state = self._prepare_state(
            prompt=prompt, instruction=instruction,
            n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat, verbose=verbose,
            mape_anchor=mape_anchor, uncond_no_text=False,
            spatial_merge_size=spatial_merge_size,
            prompt_format=prompt_format,
            latent_pos_base=latent_pos_base,
        )
        if cfg_scale > 1.0:
            uncond_state = self._prepare_state(
                prompt="", instruction=instruction,
                n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat, verbose=False,
                mape_anchor=mape_anchor,
                uncond_no_text=(cfg_uncond_mode == "no_text"),
                spatial_merge_size=spatial_merge_size,
                prompt_format=prompt_format,
                latent_pos_base=latent_pos_base,
            )
            if verbose:
                print(f"  CFG enabled, scale={cfg_scale}, mode={cfg_uncond_mode}, "
                      f"uncond tokens={uncond_state['T']}, cond tokens={cond_state['T']}")
        else:
            uncond_state = None

        # latent_pos_embed indices: flat into (max_frames × max_side × max_side).
        # For grid cell (frame, row, col): idx = frame*64² + row*64 + col.
        lpe_indices = mx.array(
            [
                f * (MAX_LATENT_SIDE ** 2) + r * MAX_LATENT_SIDE + c
                for f in range(t_lat)
                for r in range(h_lat)
                for c in range(w_lat)
            ],
            dtype=mx.int32,
        )

        # --- Init noise -------------------------------------------------
        mx.random.seed(seed)
        latents = mx.random.normal((1, t_lat, h_lat, w_lat, VAE_LATENT_CHANNELS))
        latents_dtype = self.lance_model.embed_tokens.weight.dtype
        latents = latents.astype(latents_dtype)

        # --- Flow loop ---------------------------------------------------
        sched = timestep_schedule(num_steps=num_steps, shift=timestep_shift)
        if verbose:
            print(f"  schedule: {[round(float(sched[i]), 4) for i in range(min(6, num_steps+1))]} ...")

        for step in range(num_steps):
            t = sched[step]
            dt = sched[step] - sched[step + 1]
            # Per upstream Lance: CFG fires only inside cfg_interval; outside, scale collapses to 1.0.
            t_scalar = float(t.item()) if hasattr(t, "item") else float(t)
            cfg_active = (t_scalar > cfg_lo) and (t_scalar <= cfg_hi)
            cfg_scale_step = cfg_scale if cfg_active else 1.0

            v_cond = self._step_velocity(
                state=cond_state, latents=latents, t=t,
                lpe_indices=lpe_indices,
                n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
            )
            if uncond_state is not None and cfg_scale_step > 1.0:
                v_uncond = self._step_velocity(
                    state=uncond_state, latents=latents, t=t,
                    lpe_indices=lpe_indices,
                    n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
                )
                v_cfg = v_uncond + cfg_scale_step * (v_cond - v_uncond)

                if cfg_renorm_type == "global":
                    norm_cond = mx.sqrt(mx.sum(v_cond * v_cond))
                    norm_cfg = mx.sqrt(mx.sum(v_cfg * v_cfg))
                    ratio = norm_cond / (norm_cfg + 1e-8)
                    scale = mx.clip(ratio, cfg_renorm_min, 1.0)
                    velocity = v_cfg * scale
                elif cfg_renorm_type == "channel":
                    norm_cond = mx.sqrt(mx.sum(v_cond * v_cond, axis=-1, keepdims=True))
                    norm_cfg = mx.sqrt(mx.sum(v_cfg * v_cfg, axis=-1, keepdims=True))
                    ratio = norm_cond / (norm_cfg + 1e-8)
                    scale = mx.clip(ratio, cfg_renorm_min, 1.0)
                    velocity = v_cfg * scale
                else:
                    velocity = v_cfg
            else:
                velocity = v_cond

            latents = latents - velocity * dt
            mx.eval(latents)

            if verbose:
                lat_np = latents.astype(mx.float32)
                print(f"  step {step+1}/{num_steps} t={float(t):.4f} dt={float(dt):.4f}  "
                      f"mean={float(mx.mean(lat_np)):.3f}  std={float(mx.std(lat_np)):.3f}")

        # --- VAE decode -------------------------------------------------
        if verbose:
            print(f"  VAE decode ...")
        z = denormalize_latents(latents).astype(self.vae_decoder.conv2.weight.dtype)
        decoded = self.vae_decoder(z)             # (1, T', H', W', 3) in [-1, 1]
        mx.eval(decoded)

        # Convert to uint8 frames (T', H', W', 3). VAE may add extra leading
        # padding frames; we keep them all so the caller can decide.
        import numpy as np
        frames_t = decoded[0]                     # (T', H', W', 3)
        frames_np = np.array(frames_t.astype(mx.float32))
        frames_u8 = ((frames_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return frames_u8

    # ------ per-prompt state assembly ----------------------------------------

    def _prepare_state(
        self,
        *,
        prompt: str,
        instruction: str,
        n_lat: int,
        t_lat: int,
        h_lat: int,
        w_lat: int,
        verbose: bool,
        mape_anchor: int | None = MAPE_ANCHOR_VIDEO_GEN,
        uncond_no_text: bool = False,
        spatial_merge_size: int = 1,
        prompt_format: str = "ours",
        latent_pos_base: int | None = None,
    ) -> dict:
        """Pack the prompt-dependent state needed for one CFG-arm of the flow.

        `uncond_no_text=True` builds a text-stripped sequence containing only
        the latent block (per upstream Lance's `uncond_split_pro_new`,
        `lance_lance.py:755+`, which selects positions where
        `i_sample_modality != 0`, i.e. non-text positions only). Used for the
        CFG-uncond arm. The CFG direction becomes `(v_text - v_no_text)` rather
        than `(v_text - v_empty_prompt)` which under-amplifies fine-detail
        features.
        """
        video_pad_str = "<|video_pad|>" * n_lat
        if uncond_no_text:
            # Minimal sequence — just the latent block, no chat template.
            # Wrap in vision_start/vision_end so the latent block is still
            # delimited (these are also non-text modality positions upstream).
            text = f"<|vision_start|>{video_pad_str}<|vision_end|>"
        elif prompt_format == "rocktalk":
            # Phase 5h: RockTalk's minimal chat template per their HF card:
            # `<|im_start|> [prompt tokens] <|im_end|> <|vision_start|>
            #  [N latent placeholders] <|vision_end|>`
            # No system/user/assistant role tags, no T2V_INSTRUCTION prefix.
            # This is THEIR working pipeline's template — the minimal wrap
            # may be critical for not over-shifting latent position-IDs into
            # out-of-distribution territory relative to training.
            text = (
                f"<|im_start|>{prompt}<|im_end|>"
                f"<|vision_start|>{video_pad_str}<|vision_end|>"
            )
        else:
            # 'ours' (legacy): full chat template with system + user +
            # assistant role tags and the T2V_INSTRUCTION prefix.
            text = (
                f"<|im_start|>system\n{instruction}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"<|vision_start|>{video_pad_str}<|vision_end|>"
            )

        tokenizer = self.processor.tokenizer
        input_ids = mx.array(
            [tokenizer(text, add_special_tokens=False)["input_ids"]],
            dtype=mx.int32,
        )
        T = input_ids.shape[1]
        if verbose:
            print(f"  prompt tokens: {T} ({T - n_lat} text + {n_lat} latent)")

        ids_list = input_ids[0].tolist()
        latent_positions = [
            i for i, v in enumerate(ids_list) if v == self.video_pad_token_id
        ]
        assert len(latent_positions) == n_lat, (
            f"expected {n_lat} latent positions, found {len(latent_positions)}"
        )
        latent_positions_arr = mx.array(latent_positions, dtype=mx.int32)
        first_latent_pos = latent_positions[0]
        text_len_before_latents = first_latent_pos

        position_ids = self._build_position_ids(
            T=T, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
            text_len_before_latents=text_len_before_latents,
            latent_positions=latent_positions,
            mape_anchor=mape_anchor,
            spatial_merge_size=spatial_merge_size,
            latent_pos_base=latent_pos_base,
        )

        position_group = mx.full((T,), int(PositionGroup.TEXT), dtype=mx.int32)
        position_group = self._scatter_set(
            position_group, latent_positions_arr, int(PositionGroup.NOISY_VAE)
        )

        text_embeds = self.lance_model.embed_tokens(input_ids)
        mask = self._build_block_mask(T, latent_positions, dtype=text_embeds.dtype)

        return {
            "T": T,
            "input_ids": input_ids,
            "text_embeds": text_embeds,
            "latent_positions_arr": latent_positions_arr,
            "position_ids": position_ids,
            "position_group": position_group,
            "mask": mask,
        }

    def _step_velocity(
        self,
        *,
        state: dict,
        latents: mx.array,
        t: mx.array,
        lpe_indices: mx.array,
        n_lat: int,
        t_lat: int,
        h_lat: int,
        w_lat: int,
    ) -> mx.array:
        """One forward pass; returns velocity reshaped to (1, t_lat, h_lat, w_lat, C)."""
        latents_flat = latents.reshape(1, n_lat, VAE_LATENT_CHANNELS)
        pe = self.lance_model.latent_pos_embed(lpe_indices)[None, ...]
        t_emb = self.lance_model.time_embedder(t.reshape(1)).reshape(1, 1, -1)
        lat_embed = self.lance_model.vae_in_proj(latents_flat) + pe + t_emb

        inputs_embeds = self._scatter_embeds(
            state["text_embeds"], lat_embed, state["latent_positions_arr"],
        )

        h = self.lance_model(
            inputs_embeds=inputs_embeds,
            position_ids=state["position_ids"],
            position_group=state["position_group"],
            mask=state["mask"],
        )
        h_lat_pos = h[:, state["latent_positions_arr"], :]
        velocity_flat = self.lance_model.llm2vae(h_lat_pos)
        return velocity_flat.reshape(1, t_lat, h_lat, w_lat, VAE_LATENT_CHANNELS)

    # ----- helpers ----------------------------------------------------------

    @staticmethod
    def _scatter_set(arr: mx.array, idx: mx.array, value: int) -> mx.array:
        import numpy as np
        out_np = np.array(arr)
        out_np[np.array(idx)] = value
        return mx.array(out_np)

    @staticmethod
    def _scatter_embeds(
        base: mx.array,
        inserts: mx.array,
        positions: mx.array,
    ) -> mx.array:
        import numpy as np
        target_dtype = base.dtype
        out_np = np.array(base.astype(mx.float32))
        ins_np = np.array(inserts.astype(mx.float32))
        pos_np = np.array(positions)
        out_np[:, pos_np, :] = ins_np
        return mx.array(out_np).astype(target_dtype)

    @staticmethod
    def _build_block_mask(T: int, latent_positions: list[int], dtype) -> mx.array:
        """Causal OR bidirectional-within-latent-block additive mask (T, T).
        Same pattern as t2i — the noisy-VAE positions need full mutual
        attention to denoise coherently."""
        i = mx.arange(T)[:, None]
        j = mx.arange(T)[None, :]
        lat_start = latent_positions[0]
        lat_end = latent_positions[-1] + 1
        in_lat_q = (i >= lat_start) & (i < lat_end)
        in_lat_kv = (j >= lat_start) & (j < lat_end)
        bidirectional = in_lat_q & in_lat_kv
        allowed = (i >= j) | bidirectional
        neg_inf = mx.array(-1e9, dtype=dtype)
        zero = mx.array(0.0, dtype=dtype)
        return mx.where(allowed, zero, neg_inf)

    def _build_position_ids(
        self,
        *,
        T: int,
        t_lat: int,
        h_lat: int,
        w_lat: int,
        text_len_before_latents: int,
        latent_positions: list[int],
        mape_anchor: int | None = MAPE_ANCHOR_VIDEO_GEN,
        spatial_merge_size: int = 1,
        latent_pos_base: int | None = None,
    ) -> mx.array:
        """Build (3, 1, T) position_ids with 3D grid for latent positions.

        Layout: latent token i (in flat row-major (t, h, w) order) gets:
          - t-axis: base + frame_idx     (BEFORE MaPE shift)
          - h-axis: base + (row_idx // spatial_merge_size)
          - w-axis: base + (col_idx // spatial_merge_size)
        Then MaPE re-anchors the t-axis of latent positions:
          - shift = 2000 - first_latent_t_axis_position  (modality 3 = video_gen)
          - applied uniformly to all latent positions

        `spatial_merge_size`: divisor for h/w axes (P0b candidate from issue #2).
        Default `1` = no merging (legacy). Upstream Qwen2.5-VL convention is
        `sms=2` (see `data/common.py::shift_position_ids`). RockTalk's parallel
        MLX port also uses `sms=2`. Setting to 2 halves the spatial position-id
        spread, which matches the trained mrope convention for visual tokens
        and may close residual fine-detail gap on water/textures.

        `latent_pos_base`: anchor (origin) for the latent block's (t, h, w)
        grid coords. **None (default, legacy):** `base = text_len_before_latents`
        — the latent grid starts where text ends, so latent coords drift with
        prompt length. **0 (Phase 5i.2 hypothesis):** `base = 0` — latent grid
        always starts at origin regardless of prompt length, matching
        Qwen2.5-VL's training convention where visual tokens use 3D-mrope
        grid origin (not concatenated with text positions). The Phase 5i.2
        bisect showed long prompts trigger watercolor while short prompts
        produce sharp output at the same other config — strong signal that
        prompt-length-dependent position-ID drift is the bug.
        """
        import numpy as np
        pos = np.zeros((3, 1, T), dtype=np.int32)
        seq = np.arange(T, dtype=np.int32)
        pos[0, 0, :] = seq
        pos[1, 0, :] = seq
        pos[2, 0, :] = seq

        sms = max(1, int(spatial_merge_size))
        base = text_len_before_latents if latent_pos_base is None else int(latent_pos_base)
        for idx, token_pos in enumerate(latent_positions):
            f = idx // (h_lat * w_lat)
            r = (idx % (h_lat * w_lat)) // w_lat
            c = (idx % (h_lat * w_lat)) % w_lat
            pos[0, 0, token_pos] = base + f
            pos[1, 0, token_pos] = base + (r // sms)
            pos[2, 0, token_pos] = base + (c // sms)

        # Tokens after the latent block (vision_end) continue from the max.
        max_grid = max(t_lat - 1, (h_lat - 1) // sms, (w_lat - 1) // sms)
        after_latents_start = latent_positions[-1] + 1
        if after_latents_start < T:
            tail_len = T - after_latents_start
            tail = base + max_grid + 1 + np.arange(tail_len, dtype=np.int32)
            pos[:, 0, after_latents_start:] = tail[None, :]

        # MaPE re-anchor: optionally re-anchor the t-axis of latent positions
        # to `mape_anchor`. Pass None to skip — this matches upstream's
        # `shift_position_ids` behavior for pure t2v (its gate
        # `attn_mode in ['full_noise','full']` never fires for `'noise'`-only
        # samples, so upstream does NOT re-anchor t2v positions). Under
        # investigation as Candidate 0 in github issue #2.
        if mape_anchor is not None:
            first_latent_t = pos[0, 0, latent_positions[0]]
            shift = int(mape_anchor) - int(first_latent_t)
            for token_pos in latent_positions:
                pos[0, 0, token_pos] += shift

        return mx.array(pos)
