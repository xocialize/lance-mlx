"""video_edit — instruction-based video editing via Lance + Wan2.2 VAE.

Phase 4d MVP. Fusion of image_edit's two-latent-block layout with t2v's
3D (T_lat, H_lat, W_lat) latent grid. Same flow loop, same MaPE anchor,
same bidir-within-block attention pattern — just N_frames > 1.

Algorithmic overview:

  z_clean = vae_encoder(input_video)             # (1, T_lat, H_lat, W_lat, 48)
  z_t = randn_like(z_clean)                      # noisy target init

  for t in schedule(1.0 → 0.0, 30 steps):
      embed[clean_positions] = vae_in_proj(z_clean) + latent_pos_embed + time_embedder(0)
      embed[noisy_positions] = vae_in_proj(z_t)     + latent_pos_embed + time_embedder(t)
      h = lance_model(embed, position_ids, position_group, mask)
      velocity = llm2vae(h[noisy_positions])
      z_t = z_t - velocity * dt
  video = vae_decoder(denormalize(z_t))

Token sequence layout (same as image_edit but n_lat = T_lat × H_lat × W_lat):

  <|im_start|>system\n{edit_instruction}<|im_end|>\n
  <|im_start|>user\n<|vision_start|>{video_pad × n_lat (clean ref)}<|vision_end|>{user_text}<|im_end|>\n
  <|im_start|>assistant\n<|vision_start|>{video_pad × n_lat (noisy target)}<|vision_end|>

Routing + position handling: identical to image_edit. Both latent blocks use
PositionGroup.CLEAN_VAE / NOISY_VAE (both → GEN expert). MaPE anchors both
blocks' t-axis to 1000.

Default weights: Lance_3B_Video (per upstream — video tasks use the video
specialist, which has the larger 126976-entry latent_pos_embed table).

MVP simplifications (same as image_edit):
  - Skip Qwen2.5-VL ViT semantic stream of input video.
  - Text-only CFG (matches upstream cfg_type=0).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_video.models.wan_2.vae22 import (
    Wan22VAEDecoder,
    Wan22VAEEncoder,
    denormalize_latents,
)
from mlx_vlm.models.qwen2_5_vl.config import TextConfig
from PIL import Image

from lance_mlx.model import LanceModel
from lance_mlx.model.flow_head import timestep_schedule
from lance_mlx.model.routing import PositionGroup


# Upstream Lance's video_edit system-prompt (same template generator as
# image_edit but with vision_type="video").
EDIT_INSTRUCTION = (
    "Describe the key features of the input video (color, shape, size, "
    "texture, objects, background), then explain how the user's text "
    "instruction should alter or modify the video. Generate a new video "
    "that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate."
)

MAPE_ANCHOR_VIDEO_GEN = 1000  # same anchor as image_edit per upstream

VAE_LATENT_CHANNELS = 48
VAE_SPATIAL_DOWNSAMPLE = 16
VAE_TEMPORAL_DOWNSAMPLE = 4
MAX_LATENT_SIDE = 64           # Lance_3B_Video latent_pos_embed grid


class VideoEditPipeline:
    """Lance video_edit — input video + text instruction → edited video frames."""

    def __init__(
        self,
        lance_model: LanceModel,
        vae_encoder: Wan22VAEEncoder,
        vae_decoder: Wan22VAEDecoder,
        processor,
        text_config: TextConfig,
        image_pad_token_id: int,
        video_pad_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
    ):
        self.lance_model = lance_model
        self.vae_encoder = vae_encoder
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
    ) -> "VideoEditPipeline":
        lance_weights_dir = Path(lance_weights_dir)
        vae_safetensors = Path(vae_safetensors)

        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(hf_processor_repo)
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

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

        saved_vae = mx.load(str(vae_safetensors))
        vae_encoder = Wan22VAEEncoder(z_dim=VAE_LATENT_CHANNELS, dim=160)
        enc_state = {
            k: v for k, v in saved_vae.items()
            if k.startswith("encoder.") or k.startswith("conv1.")
        }
        vae_encoder.load_weights(list(enc_state.items()))
        mx.eval(vae_encoder.parameters())

        vae_decoder = Wan22VAEDecoder(z_dim=VAE_LATENT_CHANNELS, dim=160, dec_dim=256)
        dec_state = {
            k: v for k, v in saved_vae.items()
            if k.startswith("decoder.") or k.startswith("conv2.")
        }
        vae_decoder.load_weights(list(dec_state.items()))
        mx.eval(vae_decoder.parameters())

        return cls(
            lance_model=lance_model,
            vae_encoder=vae_encoder,
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
        input_video,                                # Path/str (.mp4) or np.ndarray (T,H,W,3)
        instruction: str,
        *,
        height: int = 256,
        width: int = 256,
        num_frames: int = 17,                       # must satisfy (num_frames-1) % 4 == 0
        num_steps: int = 30,
        timestep_shift: float = 3.5,
        cfg_scale: float = 4.0,
        cfg_renorm_type: str = "global",
        cfg_renorm_min: float = 0.0,
        seed: int = 42,
        verbose: bool = False,
        system_prompt: str = EDIT_INSTRUCTION,
    ) -> np.ndarray:
        """Edit a video.

        Returns a (T_out, H, W, 3) uint8 numpy array of decoded frames.
        T_out = (T_lat - 1) × 4 + 1 where T_lat = (num_frames - 1) // 4 + 1.
        """
        assert height % VAE_SPATIAL_DOWNSAMPLE == 0
        assert width % VAE_SPATIAL_DOWNSAMPLE == 0
        h_lat = height // VAE_SPATIAL_DOWNSAMPLE
        w_lat = width // VAE_SPATIAL_DOWNSAMPLE
        t_lat = (num_frames - 1) // VAE_TEMPORAL_DOWNSAMPLE + 1
        n_lat = t_lat * h_lat * w_lat

        # --- 1. Load + VAE-encode source video --------------------------
        src = self._load_video_tensor(input_video, num_frames=num_frames,
                                       height=height, width=width, verbose=verbose)
        if verbose:
            print(f"  source video: {tuple(src.shape)}  range=[{float(src.min()):.2f}, {float(src.max()):.2f}]")
            print(f"  latent dims: t_lat={t_lat} h_lat={h_lat} w_lat={w_lat} n_lat={n_lat}")
        z_clean = self.vae_encoder(src)                             # (1, t_lat, h_lat, w_lat, 48)
        mx.eval(z_clean)
        z_clean = z_clean.astype(self.lance_model.embed_tokens.weight.dtype)
        if verbose:
            print(f"  z_clean: {tuple(z_clean.shape)}  "
                  f"mean={float(mx.mean(z_clean)):.3f} std={float(mx.std(z_clean)):.3f}")

        # --- 2. Per-prompt state ----------------------------------------
        cond_state = self._prepare_state(
            instruction=instruction, system_prompt=system_prompt,
            n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat, verbose=verbose,
        )
        if cfg_scale > 1.0:
            uncond_state = self._prepare_state(
                instruction="", system_prompt=system_prompt,
                n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat, verbose=False,
            )
        else:
            uncond_state = None

        lpe_indices = mx.array(
            [
                f * (MAX_LATENT_SIDE ** 2) + r * MAX_LATENT_SIDE + c
                for f in range(t_lat)
                for r in range(h_lat)
                for c in range(w_lat)
            ],
            dtype=mx.int32,
        )

        # --- 3. Init noise ----------------------------------------------
        mx.random.seed(seed)
        z_t = mx.random.normal((1, t_lat, h_lat, w_lat, VAE_LATENT_CHANNELS))
        z_t = z_t.astype(z_clean.dtype)

        # --- 4. Flow loop -----------------------------------------------
        sched = timestep_schedule(num_steps=num_steps, shift=timestep_shift)
        for step in range(num_steps):
            t = sched[step]
            dt = sched[step] - sched[step + 1]

            v_cond = self._step_velocity(
                state=cond_state, z_t=z_t, z_clean=z_clean, t=t,
                lpe_indices=lpe_indices, n_lat=n_lat,
                t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
            )
            if uncond_state is not None:
                v_uncond = self._step_velocity(
                    state=uncond_state, z_t=z_t, z_clean=z_clean, t=t,
                    lpe_indices=lpe_indices, n_lat=n_lat,
                    t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
                )
                v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
                if cfg_renorm_type == "global":
                    nc = mx.sqrt(mx.sum(v_cond * v_cond))
                    nf = mx.sqrt(mx.sum(v_cfg * v_cfg))
                    velocity = v_cfg * mx.clip(nc / (nf + 1e-8), cfg_renorm_min, 1.0)
                elif cfg_renorm_type == "channel":
                    nc = mx.sqrt(mx.sum(v_cond * v_cond, axis=-1, keepdims=True))
                    nf = mx.sqrt(mx.sum(v_cfg * v_cfg, axis=-1, keepdims=True))
                    velocity = v_cfg * mx.clip(nc / (nf + 1e-8), cfg_renorm_min, 1.0)
                else:
                    velocity = v_cfg
            else:
                velocity = v_cond

            z_t = z_t - velocity * dt
            mx.eval(z_t)
            if verbose:
                z_np = z_t.astype(mx.float32)
                print(f"  step {step+1}/{num_steps} t={float(t):.4f} dt={float(dt):.4f}  "
                      f"mean={float(mx.mean(z_np)):.3f}  std={float(mx.std(z_np)):.3f}")

        # --- 5. VAE decode + return as uint8 frames ---------------------
        z = denormalize_latents(z_t).astype(self.vae_decoder.conv2.weight.dtype)
        decoded = self.vae_decoder(z)
        mx.eval(decoded)
        frames = decoded[0]                                           # (T_out, H, W, 3)
        frames_np = np.array(frames.astype(mx.float32))
        frames_u8 = ((frames_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return frames_u8

    # ----- per-prompt state assembly (mirrors image_edit) ------------------

    def _prepare_state(
        self,
        *,
        instruction: str,
        system_prompt: str,
        n_lat: int,
        t_lat: int,
        h_lat: int,
        w_lat: int,
        verbose: bool,
    ) -> dict:
        video_pad_str = "<|video_pad|>" * n_lat
        text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{video_pad_str}<|vision_end|>{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n<|vision_start|>{video_pad_str}<|vision_end|>"
        )
        tokenizer = self.processor.tokenizer
        input_ids = mx.array(
            [tokenizer(text, add_special_tokens=False)["input_ids"]],
            dtype=mx.int32,
        )
        T = input_ids.shape[1]

        ids_list = input_ids[0].tolist()
        video_pad_positions = [
            i for i, v in enumerate(ids_list) if v == self.video_pad_token_id
        ]
        assert len(video_pad_positions) == 2 * n_lat, (
            f"expected {2 * n_lat} video_pad tokens, found {len(video_pad_positions)}"
        )
        clean_positions = video_pad_positions[:n_lat]
        noisy_positions = video_pad_positions[n_lat:]
        if verbose:
            print(f"  prompt tokens: {T}  "
                  f"clean=[{clean_positions[0]}..{clean_positions[-1]}]  "
                  f"noisy=[{noisy_positions[0]}..{noisy_positions[-1]}]")

        clean_positions_arr = mx.array(clean_positions, dtype=mx.int32)
        noisy_positions_arr = mx.array(noisy_positions, dtype=mx.int32)

        position_ids = self._build_position_ids(
            T=T, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
            clean_positions=clean_positions,
            noisy_positions=noisy_positions,
        )

        position_group = mx.full((T,), int(PositionGroup.TEXT), dtype=mx.int32)
        position_group = self._scatter_set(
            position_group, clean_positions_arr, int(PositionGroup.CLEAN_VAE)
        )
        position_group = self._scatter_set(
            position_group, noisy_positions_arr, int(PositionGroup.NOISY_VAE)
        )

        text_embeds = self.lance_model.embed_tokens(input_ids)
        mask = self._build_block_mask(
            T, clean_positions, noisy_positions, dtype=text_embeds.dtype
        )

        return {
            "T": T,
            "input_ids": input_ids,
            "text_embeds": text_embeds,
            "clean_positions_arr": clean_positions_arr,
            "noisy_positions_arr": noisy_positions_arr,
            "position_ids": position_ids,
            "position_group": position_group,
            "mask": mask,
        }

    @staticmethod
    def _build_block_mask(T, clean_positions, noisy_positions, dtype):
        i = mx.arange(T)[:, None]
        j = mx.arange(T)[None, :]
        cs, ce = clean_positions[0], clean_positions[-1] + 1
        ns, ne = noisy_positions[0], noisy_positions[-1] + 1
        bidir_clean = ((i >= cs) & (i < ce)) & ((j >= cs) & (j < ce))
        bidir_noisy = ((i >= ns) & (i < ne)) & ((j >= ns) & (j < ne))
        allowed = (i >= j) | bidir_clean | bidir_noisy
        neg_inf = mx.array(-1e9, dtype=dtype)
        zero = mx.array(0.0, dtype=dtype)
        return mx.where(allowed, zero, neg_inf)

    def _step_velocity(
        self, *, state, z_t, z_clean, t,
        lpe_indices, n_lat, t_lat, h_lat, w_lat,
    ):
        z_clean_flat = z_clean.reshape(1, n_lat, VAE_LATENT_CHANNELS)
        z_t_flat = z_t.reshape(1, n_lat, VAE_LATENT_CHANNELS)
        pe = self.lance_model.latent_pos_embed(lpe_indices)[None, ...]

        t_zero = mx.zeros((1,), dtype=t.dtype)
        t_emb_clean = self.lance_model.time_embedder(t_zero).reshape(1, 1, -1)
        t_emb_noisy = self.lance_model.time_embedder(t.reshape(1)).reshape(1, 1, -1)

        clean_embed = self.lance_model.vae_in_proj(z_clean_flat) + pe + t_emb_clean
        noisy_embed = self.lance_model.vae_in_proj(z_t_flat) + pe + t_emb_noisy

        inputs_embeds = self._scatter_two_blocks(
            state["text_embeds"],
            clean_embed, state["clean_positions_arr"],
            noisy_embed, state["noisy_positions_arr"],
        )

        h = self.lance_model(
            inputs_embeds=inputs_embeds,
            position_ids=state["position_ids"],
            position_group=state["position_group"],
            mask=state["mask"],
        )
        h_noisy = h[:, state["noisy_positions_arr"], :]
        velocity_flat = self.lance_model.llm2vae(h_noisy)
        return velocity_flat.reshape(1, t_lat, h_lat, w_lat, VAE_LATENT_CHANNELS)

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _load_video_tensor(
        video,
        *,
        num_frames: int,
        height: int,
        width: int,
        verbose: bool,
    ) -> mx.array:
        """Load + uniformly-sample → (1, T=num_frames, H, W, 3) in [-1, 1]."""
        if isinstance(video, (str, Path)):
            import imageio.v3 as iio
            all_frames = [f for f in iio.imiter(str(video))]
            n_total = len(all_frames)
            idx = np.linspace(0, n_total - 1, num_frames).astype(int)
            sampled = [all_frames[i] for i in idx]
            if verbose:
                print(f"  sampled {num_frames}/{n_total} frames from video "
                      f"(indices {idx[:3].tolist()}..{idx[-2:].tolist()})")
        else:
            sampled = list(video)
            if len(sampled) != num_frames:
                raise ValueError(
                    f"video has {len(sampled)} frames, expected {num_frames}"
                )
        resized = []
        for f in sampled:
            im = Image.fromarray(np.asarray(f)).convert("RGB").resize((width, height), Image.LANCZOS)
            resized.append(np.asarray(im, dtype=np.float32))
        arr = np.stack(resized) / 127.5 - 1.0                            # (T, H, W, 3) in [-1,1]
        arr = arr[None, ...]                                              # (1, T, H, W, 3)
        return mx.array(arr)

    @staticmethod
    def _scatter_set(arr, idx, value):
        out_np = np.array(arr)
        out_np[np.array(idx)] = value
        return mx.array(out_np)

    @staticmethod
    def _scatter_two_blocks(base, clean_block, clean_pos, noisy_block, noisy_pos):
        target_dtype = base.dtype
        out_np = np.array(base.astype(mx.float32))
        c_np = np.array(clean_block.astype(mx.float32))
        n_np = np.array(noisy_block.astype(mx.float32))
        out_np[:, np.array(clean_pos), :] = c_np
        out_np[:, np.array(noisy_pos), :] = n_np
        return mx.array(out_np).astype(target_dtype)

    def _build_position_ids(
        self, *, T, t_lat, h_lat, w_lat, clean_positions, noisy_positions,
    ):
        """Both blocks share the same 3D (t,h,w) grid spatially. MaPE anchors both."""
        pos = np.zeros((3, 1, T), dtype=np.int32)
        seq = np.arange(T, dtype=np.int32)
        pos[0, 0, :] = seq
        pos[1, 0, :] = seq
        pos[2, 0, :] = seq

        clean_base = clean_positions[0]
        max_grid = max(t_lat, h_lat, w_lat) - 1

        for block_positions in (clean_positions, noisy_positions):
            for idx, token_pos in enumerate(block_positions):
                f = idx // (h_lat * w_lat)
                rest = idx % (h_lat * w_lat)
                r = rest // w_lat
                c = rest % w_lat
                pos[0, 0, token_pos] = clean_base + f
                pos[1, 0, token_pos] = clean_base + r
                pos[2, 0, token_pos] = clean_base + c

        # Tail tokens after the noisy block.
        after_clean = clean_positions[-1] + 1
        before_noisy = noisy_positions[0]
        if before_noisy > after_clean:
            gap_len = before_noisy - after_clean
            tail = clean_base + max_grid + 1 + np.arange(gap_len, dtype=np.int32)
            pos[:, 0, after_clean:before_noisy] = tail[None, :]
        after_noisy = noisy_positions[-1] + 1
        if after_noisy < T:
            gap_len = before_noisy - after_clean
            tail_start = clean_base + max_grid + 1 + gap_len
            tail_len = T - after_noisy
            tail = tail_start + np.arange(tail_len, dtype=np.int32)
            pos[:, 0, after_noisy:] = tail[None, :]

        # MaPE re-anchor t-axis of both latent blocks to 1000.
        first_latent_t = pos[0, 0, clean_positions[0]]
        shift = MAPE_ANCHOR_VIDEO_GEN - int(first_latent_t)
        for token_pos in clean_positions + noisy_positions:
            pos[0, 0, token_pos] += shift

        return mx.array(pos)
