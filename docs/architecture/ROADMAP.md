# Forge LLM (MUD) — Architecture Roadmap

## Current Status (V2.0 — SlimeRegister Paradigm)

- **Ternary2Bit Inference:** Operational with BitNet b1.58-2B-4T (30 layers, 2560 hidden, 6912 FFN). Real autoregressive generation with KV cache.
- **RMSNorm Quantization Fix:** Peak-based i8 quantization (`f32 / peak * 127`) prevents activation collapse that caused all-zero registers.
- **JEPA var_ema:** Persistent variance EMA stored in `SlimeWorkspace` (not `&mut 1.0` temp), enabling proper orbital convergence across layers.
- **Autoregressive Loop:** 32-token generation with EOS (token 0) stop, feeds predicted token embedding back as input.
- **Zero-Allocation Forward:** All buffers pre-allocated in `SlimeWorkspace`, no `vec![]` in hot loop (P-01).
- **Hardware Acceleration:** AVX2 ternary GEMV kernels (`ternary_gemv_i8act`), SiLU vectorial, Adam step.
- **QAT Pipeline:** STE autograd, sigma-reparameterization, knowledge distillation, corpus adaptation ready.

## Phase 14 — Inference Quality & Correctness (ACTIVE)

### 1. Model Fidelity
- [x] RMSNorm → i8 quantization (peak-based scaling, was zeroing activations)
- [x] JEPA var_ema persistence (was resetting every layer)
- [x] Autoregressive generation loop (was single-token only)
- [x] Workspace re-dimensioned to actual model shape (ffn_mid=6912, not 4096)
- [x] Token embedding to SlimeRegister scale fix (`emb_val / PRQ_INPUT_SCALE`)
- [ ] Multi-expert routing (num_experts=1 today, should support >1)
- [ ] Sub-norm integration (`attn_sub_norm.weight`, `ffn_sub_norm.weight`)
- [ ] fp32 reference parity test (compare against dequantized reference)

### 2. Generation Quality
- [ ] Watermark-aware corpus sanitization (prevent model from learning dataset marks)
- [ ] Temperature/Top-K/Top-P sampling (today: argmax-only)
- [ ] Repetition penalty
- [ ] Long-context KV cache beyond max_pos

## Phase 15 — Training & Adaptation

- [x] Sigma-reparam QAT (spectral norm normalization, enabled by default)
- [x] STE autograd ops (Quantize, RMSNorm, MHA, KLDiv)
- [x] Adam optimizer with warmup + cosine decay
- [x] Bilingual corpus (42k lines ES/EN)
- [x] Knowledge distillation (KL-Div + MiniLM attention)
- [x] ECC parity generation on convert
- [ ] Calibration dataset for PRQ scale initialization
- [ ] Multi-GPU Vulkan backend for training

## Phase 16 — Production Hardening

- [ ] CI pipeline with `cargo clippy -D warnings` and full test suite
- [ ] Benchmark suite (TPS, memory, accuracy vs reference)
- [ ] Model zoo / hub integration
- [ ] Docker deployment
- [ ] REST API server mode
