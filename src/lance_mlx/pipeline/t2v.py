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
        if num_latent_positions != MAX_NUM_LATENT_POSITIONS:
            print(f"WARNING: latent_pos_embed has {num_latent_positions} entries; "
                  f"video pipeline expects {MAX_NUM_LATENT_POSITIONS} (= 31×64×64). "
                  f"Are you using Lance_3B_Video weights?")
        lance_model = LanceModel(text_cfg, num_latent_positions=num_latent_positions)
        lance_model.load_weights(list(saved_lance.items()))
        mx.eval(lance_model.parameters())

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
        seed: int = 42,
        verbose: bool = False,
        instruction: str = T2V_INSTRUCTION,
    ) -> mx.array:
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
        )
        if cfg_scale > 1.0:
            uncond_state = self._prepare_state(
                prompt="", instruction=instruction,
                n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat, verbose=False,
            )
            if verbose:
                print(f"  CFG enabled, scale={cfg_scale}, "
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

            v_cond = self._step_velocity(
                state=cond_state, latents=latents, t=t,
                lpe_indices=lpe_indices,
                n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
            )
            if uncond_state is not None:
                v_uncond = self._step_velocity(
                    state=uncond_state, latents=latents, t=t,
                    lpe_indices=lpe_indices,
                    n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
                )
                v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

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
    ) -> dict:
        """Pack the prompt-dependent state needed for one CFG-arm of the flow."""
        video_pad_str = "<|video_pad|>" * n_lat
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
    ) -> mx.array:
        """Build (3, 1, T) position_ids with 3D grid for latent positions.

        Layout: latent token i (in flat row-major (t, h, w) order) gets:
          - t-axis: text_len + frame_idx     (BEFORE MaPE shift)
          - h-axis: text_len + row_idx
          - w-axis: text_len + col_idx
        Then MaPE re-anchors the t-axis of latent positions:
          - shift = 2000 - first_latent_t_axis_position  (modality 3 = video_gen)
          - applied uniformly to all latent positions
        """
        import numpy as np
        pos = np.zeros((3, 1, T), dtype=np.int32)
        seq = np.arange(T, dtype=np.int32)
        pos[0, 0, :] = seq
        pos[1, 0, :] = seq
        pos[2, 0, :] = seq

        base = text_len_before_latents
        for idx, token_pos in enumerate(latent_positions):
            f = idx // (h_lat * w_lat)
            r = (idx % (h_lat * w_lat)) // w_lat
            c = (idx % (h_lat * w_lat)) % w_lat
            pos[0, 0, token_pos] = base + f
            pos[1, 0, token_pos] = base + r
            pos[2, 0, token_pos] = base + c

        # Tokens after the latent block (vision_end) continue from the max.
        max_grid = max(t_lat, h_lat, w_lat) - 1
        after_latents_start = latent_positions[-1] + 1
        if after_latents_start < T:
            tail_len = T - after_latents_start
            tail = base + max_grid + 1 + np.arange(tail_len, dtype=np.int32)
            pos[:, 0, after_latents_start:] = tail[None, :]

        # MaPE re-anchor: t-axis of latent positions → all anchored to 2000
        # (video_gen, modality 3 per upstream shift_position_ids).
        first_latent_t = pos[0, 0, latent_positions[0]]
        shift = MAPE_ANCHOR_VIDEO_GEN - int(first_latent_t)
        for token_pos in latent_positions:
            pos[0, 0, token_pos] += shift

        return mx.array(pos)
