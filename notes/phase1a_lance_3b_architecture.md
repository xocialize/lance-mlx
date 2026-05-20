# Lance architecture — resolved from key inspection

## Q1: MaPE Δ_m learned or hard-coded?

- **q1_verdict**: HARD_CODED
- **evidence**: No mape.* keys in checkpoint
- **next_step**: Read llm_config.json for the Δ_m constants

## Q2: LM head untied?

- **q2_verdict**: UNTIED
- **evidence**: separate lm_head [151936, 2048] vs embed [151936, 2048]
- **next_step**: Load lm_head as independent parameter in LanceModel

## Q3: Flow head structure?

- **q3_verdict**: SINGLE_LINEAR_48_WITH_BIAS
- **shapes**: {'llm2vae.bias': [48], 'llm2vae.weight': [48, 2048]}
- **evidence**: llm2vae has both weight and bias — scaffold should use bias=True
- **next_step**: Update flow_head.py: nn.Linear(hidden, 48, bias=True)

## Q4: Attention QKV shared or split?

- **q4_verdict**: DUPLICATED_MOE_GEN_VERIFIED
- **und_count**: 252
- **gen_count**: 252
- **evidence**: 252 UND + 252 GEN attn keys = 36 layers × 7 tensors (4 weights + 3 biases) per side
- **next_step**: Confirm shape symmetry between matching (und, _moe_gen) pairs in convert.py

## Q5: Number of MaPE position groups?

- **q5_verdict**: CONSULT_CONFIG
- **next_step**: Read llm_config.json for 'mape_num_groups' or 'num_modality_groups'


## Next steps

1. Update `lance_mlx/model/lance_llm.py` LanceMoTLayer with the resolved attention topology (Q4)
2. Update `lance_mlx/model/flow_head.py` FlowHead with the resolved structure (Q3)
3. Update `lance_mlx/model/mape.py` MaPEOffsets `learned` flag based on Q1
4. Adjust `lance_mlx.model.routing.PositionGroup` enum if Q5 reveals a different group count
5. Read `llm_config.json` to confirm hyperparameters
