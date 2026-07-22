# MUD Project Tree

> P-18 root: only SSOT + entry docs. Everything else under `docs/`.  
> Updated: 2026-07-17 (F1/F2 trainable mHC+STP, unified trainer UI, project-adapted `mud.sh train`).

```
├── GEMINI.md                  # SSOT: policies (P-#), status, ledger L-## + F1/F2/UI
├── AGENTS.md                  # Agent context (derived; must not contradict GEMINI)
├── VISION_ROADMAP.md          # Product vision + Q3–Q4 phases
├── PLAN_MAESTRO.md            # Deep architecture narrative + MoE
├── README.md                  # Public project intro
├── TREE.md                    # This file
├── LICENSE
├── Cargo.toml / build.rs / mud.sh   # mud.sh: `train` = project-adapted corpus + STP on + 64 steps/chunk
│
├── src/
│   ├── asm/                   # AVX2 kernels (11 live)
│   ├── mud/
│   │   ├── trainer_ui.rs      # [NEW] unified box/notes console formatting (P-06 clean)
│   │   ├── stp_loss.rs        # [NEW] F2: STP geodesic trajectory aux loss (AVX2, TLS scratch)
│   │   ├── arena_judge.rs     # [NEW] F3: RLVR judge (Verifiable/Rust/Text/Professor, no-API, local cosine)
│   │   ├── slime_forward.rs    # mHC residual + tape capture for α/β grads
│   │   ├── slime_backward.rs   # SlimeLayerGradients.mhc_{alpha,beta}_grad + backward
│   │   ├── corpus_trainer.rs   # train_on_sequence; STP hook; mHC SGD writeback (CPU+ash)
│   │   └── …                   # slime, MoE, CSA, packing, JEPA…
│   ├── vulkan/                # ash backend + gemv_policy
│   ├── main.rs                # Inference CLI
│   └── …
│
├── tools/                     # Bins: run_trainer, audit, converter, benches…
├── assets/shaders/            # GLSL + spirv
├── training/corpus/           # .txt/.md + project_corpus.txt (assembled by mud.sh train)
├── models/                    # .mud checkpoints
├── forge_autograd/            # Isolated autograd crate (avx_math.rs reused by STP)
│
└── docs/
    ├── README.md              # Docs index
    ├── STATUS_REPORT.md       # Logros vs deuda (was root)
    ├── architecture/
    │   ├── MUD_TRAINER_TERNARY_JEPA_MHC.md   # §9 F1/F2 verified · §10 unified UI
    │   ├── MUD_COMPUTE_STACK.md
    │   └── …
    ├── research/
    │   ├── MUD_PLAN_MHC_STP_TRAINABLE.md     # F1/F2 plan (Phase 1+2 DONE; Phase 3 n=2 deferred)
    │   ├── MUD_GAP_ANALYSIS_POST_L15.md
    │   ├── MUD_IMPROVEMENTS_POST_AE.md       # F+ backlog
    │   └── …
    ├── audits/  ├── sessions/  ├── manuals/  └── dumps/
```

## Recent work (2026-07-17)

| Item | File | Status |
|------|------|--------|
| F1 trainable mHC α/β | `src/mud/stp_loss.rs` cross-ref · `slime_backward.rs` · `corpus_trainer.rs` | DONE + verified |
| F2 STP trajectory loss | `src/mud/stp_loss.rs` | DONE + verified (`MUD_TRAIN_STP`) |
| Unified trainer console | `src/mud/trainer_ui.rs` | DONE (one box, `note()` tags, no emoji) |
| F3 RLVR debate (juez + reward/penalty + aprendizaje) | `src/mud/arena_judge.rs` · `debate_trainer.rs` · `arena_games.rs` | DONE (no-API TextJudge/ProfessorJudge; `run_game` infinito hasta basta; `MUD_DEBATE_LEARN` default OFF) |
| F3+ Seed-driven Training Circuit | `corpus_trainer::run_training_circuit` · `mud.sh circuit` | DONE (baterías por semilla vía LCG; align/debate/games/professor; time-box por fase; telemetría + logs/circuit.log; guarda al quit; honores = `circuit_eval_integrity` estructural + `circuit_benchmark_games` win-rate vs baseline, rollback `.bak_circuit`) |
| Project-adapted `mud.sh train` | `mud.sh` (`build_project_corpus`, `compute_project_chunks`) | DONE (STP on, 64 steps/chunk, corpus = docs+src+ES/EN) |

## Root policy (P-18)

**Allowed at repo root:** `GEMINI.md`, `AGENTS.md`, `VISION_ROADMAP.md`, `PLAN_MAESTRO.md`, `README.md`, `TREE.md`, `LICENSE`.

**Moved 2026-07-16:** status report, ASM/Vulkan plans, audits, housekeeping → `docs/`.

