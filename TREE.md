# MUD Project Tree

```
├── AGENTS.md                  # AI agent project context (P-# rules, fixes, architecture)
├── Cargo.toml                 # Rust workspace root
├── mud.sh                     # Orchestrator CLI
├── README.md
│
├── src/                       # ── Core Engine ──
│   ├── main.rs                # TUI dashboard + autoregressive inference
│   ├── lib.rs                 # Library root
│   ├── hardware.rs            # CPU feature detection
│   │
│   ├── asm/                   # AVX2 assembly kernels
│   │   ├── ternary_gemv.s     # Main FP32 GEMV
│   │   ├── rmsnorm.s          # RMS normalization
│   │   ├── silu.s             # SiLU activation
│   │   ├── adam_step.s        # Adam optimizer step
│   │   └── ...
│   │
│   ├── mud/                   # ── MUD Engine Modules ──
│   │   ├── slime.rs           # SlimeRegister (i16 + u16), SlimeWorkspace, init_from_embed
│   │   ├── slime_forward.rs   # evaluate_slime_block, mhc_residual, apply_output_norm
│   │   ├── slime_backward.rs  # Backward pass (STE gradients)
│   │   ├── slime_jepa.rs      # jepa_stabilizer, check_tensor_health
│   │   ├── corpus_trainer.rs  # MudCorpusTrainer (QAT training loop)
│   │   ├── speculative.rs     # DSpark drafter
│   │   ├── self_play.rs       # Synthetic self-play
│   │   ├── muon.rs            # Muon optimizer (Newton-Schulz)
│   │   ├── galore.rs          # GaLore optimizer
│   │   ├── qat_dispatcher.rs  # Vulkan QAT dispatcher
│   │   ├── ecc.rs             # Error-correcting codes
│   │   ├── workspace.rs       # Workspace meta-layer
│   │   ├── rlvr.rs            # RLVR metrics
│   │   └── ...
│   │
│   ├── model/                 # Tokenizer + model loading
│   ├── gguf/                  # GGUF converter
│   └── vulkan/                # Vulkan backend (HMP offloading)
│
├── tools/                     # ── CLI Utilities ──
│   ├── step_inference.rs      # Non-interactive inference (headless)
│   ├── run_trainer.rs         # Training launcher
│   ├── universal_converter/   # safetensors → .mud conversion
│   ├── diagnose_model.rs      # Model diagnostic
│   ├── mud_calibrator.rs      # Calibration
│   └── ... (40+ tools)
│
├── docs/                      # ── Documentation ──
│   ├── README.md              # Documentation index
│   ├── architecture/          # Engine specs, manifestos, plans
│   ├── audits/                # V1-V33 audit reports
│   ├── research/              # Papers, theoretical analysis
│   ├── sessions/              # Daily session reports
│   │   └── MUD_SESSION_REPORT_2026-07-01.md  # Latest: JEPA Gate Rewire
│   ├── manuals/               # User guides, protocols
│   └── dumps/                 # Disassembly, debug logs
│
├── models/
│   ├── smollm2/               # SmolLM2-135M (active training model)
│   │   ├── smollm2.mud        # 282 MB — converted ternary model
│   │   └── model.safetensors  # Original HuggingFace weights
│   └── phi-4-mini/            # Phi-4-mini (not yet converted)
│
├── forge_autograd/            # Standalone autograd lib
├── playground/                # C++ calculus playground
├── training/corpus/           # Training corpus
├── weights/checkpoints/       # Saved checkpoints
│   └── model_latest_checkpoint.mud
│
├── assets/shaders/            # Vulkan compute shaders (.comp + .spv)
└── .cargo/config.toml         # RUSTFLAGS with target CPU features
```

## Key Files Changed (2026-07-01 Session)

| File | Change |
|------|--------|
| `src/mud/slime.rs` | Added `jepa_z` field to workspace; added `init_from_embed()` |
| `src/mud/slime_jepa.rs` | `jepa_stabilizer` reads/writes `z` from `z_buf`, stores `v_jepa` in `jepa_energy` + tape |
| `src/mud/slime_forward.rs` | `mhc_residual` applies `sigmoid(v_jepa)` gate; passes `z_buf` to stabilizer |
| `src/mud/corpus_trainer.rs` | Embedding init uses `init_from_embed`; P-13 fallbacks → `.expect()` |
| `src/main.rs` | Embedding init uses `init_from_embed`; enhanced telemetry |
| `src/mud/speculative.rs` | Embedding init uses `init_from_embed` |
| `src/mud/self_play.rs` | Embedding init uses `init_from_embed` |
| `AGENTS.md` | Updated for JEPA Gate Rewire, `jepa_z`, `SlimeRegister` format |
| `docs/README.md` | Latest session reference updated |
| `docs/sessions/MUD_SESSION_REPORT_2026-07-01.md` | Sections 5-8 added |
