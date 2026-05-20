# phase1a_lance_3b_video — key topology summary

- Source: `hf://bytedance-research/Lance/Lance_3B_Video/model.safetensors`
- Total tensors: **1411**
- Total parameters: **7.11 B**
- Dtypes: F32=1411

## Component breakdown

### `vit` — 390 tensors, ~0.67 B params

- `vit_model.blocks.0.attn.proj.bias` [1280]
- `vit_model.blocks.0.attn.proj.weight` [1280, 1280]
- `vit_model.blocks.0.attn.qkv.bias` [3840]
- `vit_model.blocks.0.attn.qkv.weight` [3840, 1280]
- `vit_model.blocks.0.mlp.down_proj.bias` [1280]
- ... and 385 more

### `attn_und` — 252 tensors, ~0.34 B params

- `language_model.model.layers.0.self_attn.k_proj.bias` [256]
- `language_model.model.layers.0.self_attn.k_proj.weight` [256, 2048]
- `language_model.model.layers.0.self_attn.o_proj.weight` [2048, 2048]
- `language_model.model.layers.0.self_attn.q_proj.bias` [2048]
- `language_model.model.layers.0.self_attn.q_proj.weight` [2048, 2048]
- ... and 247 more

### `attn_moe_gen` — 252 tensors, ~0.34 B params

- `language_model.model.layers.0.self_attn.k_proj_moe_gen.bias` [256]
- `language_model.model.layers.0.self_attn.k_proj_moe_gen.weight` [256, 2048]
- `language_model.model.layers.0.self_attn.o_proj_moe_gen.weight` [2048, 2048]
- `language_model.model.layers.0.self_attn.q_proj_moe_gen.bias` [2048]
- `language_model.model.layers.0.self_attn.q_proj_moe_gen.weight` [2048, 2048]
- ... and 247 more

### `mlp_und` — 108 tensors, ~2.43 B params

- `language_model.model.layers.0.mlp.down_proj.weight` [2048, 11008]
- `language_model.model.layers.0.mlp.gate_proj.weight` [11008, 2048]
- `language_model.model.layers.0.mlp.up_proj.weight` [11008, 2048]
- `language_model.model.layers.1.mlp.down_proj.weight` [2048, 11008]
- `language_model.model.layers.1.mlp.gate_proj.weight` [11008, 2048]
- ... and 103 more

### `mlp_moe_gen` — 108 tensors, ~2.43 B params

- `language_model.model.layers.0.mlp_moe_gen.down_proj.weight` [2048, 11008]
- `language_model.model.layers.0.mlp_moe_gen.gate_proj.weight` [11008, 2048]
- `language_model.model.layers.0.mlp_moe_gen.up_proj.weight` [11008, 2048]
- `language_model.model.layers.1.mlp_moe_gen.down_proj.weight` [2048, 11008]
- `language_model.model.layers.1.mlp_moe_gen.gate_proj.weight` [11008, 2048]
- ... and 103 more

### `layernorm_und` — 72 tensors, ~0.00 B params

- `language_model.model.layers.0.input_layernorm.weight` [2048]
- `language_model.model.layers.0.post_attention_layernorm.weight` [2048]
- `language_model.model.layers.1.input_layernorm.weight` [2048]
- `language_model.model.layers.1.post_attention_layernorm.weight` [2048]
- `language_model.model.layers.10.input_layernorm.weight` [2048]
- ... and 67 more

### `layernorm_moe_gen` — 72 tensors, ~0.00 B params

- `language_model.model.layers.0.input_layernorm_moe_gen.weight` [2048]
- `language_model.model.layers.0.post_attention_layernorm_moe_gen.weight` [2048]
- `language_model.model.layers.1.input_layernorm_moe_gen.weight` [2048]
- `language_model.model.layers.1.post_attention_layernorm_moe_gen.weight` [2048]
- `language_model.model.layers.10.input_layernorm_moe_gen.weight` [2048]
- ... and 67 more

### `qk_norm_und` — 72 tensors, ~0.00 B params

- `language_model.model.layers.0.self_attn.k_norm.weight` [128]
- `language_model.model.layers.0.self_attn.q_norm.weight` [128]
- `language_model.model.layers.1.self_attn.k_norm.weight` [128]
- `language_model.model.layers.1.self_attn.q_norm.weight` [128]
- `language_model.model.layers.10.self_attn.k_norm.weight` [128]
- ... and 67 more

### `qk_norm_moe_gen` — 72 tensors, ~0.00 B params

- `language_model.model.layers.0.self_attn.k_norm_moe_gen.weight` [128]
- `language_model.model.layers.0.self_attn.q_norm_moe_gen.weight` [128]
- `language_model.model.layers.1.self_attn.k_norm_moe_gen.weight` [128]
- `language_model.model.layers.1.self_attn.q_norm_moe_gen.weight` [128]
- `language_model.model.layers.10.self_attn.k_norm_moe_gen.weight` [128]
- ... and 67 more

### `timestep_embedder` — 4 tensors, ~0.00 B params

- `time_embedder.mlp.0.bias` [2048]
- `time_embedder.mlp.0.weight` [2048, 256]
- `time_embedder.mlp.2.bias` [2048]
- `time_embedder.mlp.2.weight` [2048, 2048]

### `flow_head` — 2 tensors, ~0.00 B params

- `llm2vae.bias` [48]
- `llm2vae.weight` [48, 2048]

### `vae_in_proj` — 2 tensors, ~0.00 B params

- `vae2llm.bias` [2048]
- `vae2llm.weight` [2048, 48]

### `lm_head` — 1 tensors, ~0.31 B params

- `language_model.lm_head.weight` [151936, 2048]

### `embeddings` — 1 tensors, ~0.31 B params

- `language_model.model.embed_tokens.weight` [151936, 2048]

### `final_norm` — 1 tensors, ~0.00 B params

- `language_model.model.norm.weight` [2048]

### `final_norm_moe_gen` — 1 tensors, ~0.00 B params

- `language_model.model.norm_moe_gen.weight` [2048]

### `latent_pos_embed` — 1 tensors, ~0.26 B params

- `latent_pos_embed.pos_embed` [126976, 2048]

