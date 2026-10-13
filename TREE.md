# MUD Project Tree

> P-18 root: only SSOT + entry docs. Everything else under `docs/`.  
> Updated: 2026-10-11 (auditoría completa + plan maestro `R-##`; ver `docs/planning/`) — previamente 2026-07-26 (consolidación: 59 ficheros borrados, ROADMAP.md consolidado)

```
├── GEMINI.md                  # SSOT: policies (P-#), status (§0.3 auditoría 2026-10-11), ledger L-## + backlog R-##
├── AGENTS.md                  # Agent context (derived; must not contradict GEMINI)
├── ROADMAP.md                 # Product vision + phases + architectural narrative (consolidado 2026-07-26)
├── README.md                  # Public project intro
├── TREE.md                    # This file
├── LICENSE
├── Cargo.toml / build.rs / mud.sh   # mud.sh: train/align/circuit/audit-full/health/… (v2.1)
│
├── src/
│   ├── asm/                   # AVX2 kernels (16 live, +SlimexCellV5 unpacker)
│   ├── mud/
│   │   ├── trainer_ui.rs      # unified box/notes console formatting (P-06 clean)
│   │   ├── stp_loss.rs        # F2: STP geodesic trajectory aux loss (AVX2, TLS scratch)
│   │   ├── arena_judge.rs     # F3: RLVR judge (Verifiable/Rust/Text/Professor, no-API, local cosine)
│   │   ├── fenwick.rs         # F4/F1/F5b: FenwickF32/U32/I32/LogF32 + AVX2/BMI1/BMI2 batch ops
│   │   ├── loss_cert.rs       # Stream K + F5b: CertMode {Endpoints, WindowMeans, Percentile}
│   │   ├── muon.rs            # L-02 Newton-Schulz (in-place + scratch reuse, P-01)
│   │   ├── galore.rs          # GaLore projection (caller-owned p_matrix, P-01)
│   │   ├── grad_checkpoint.rs # L-15+H: ResidualBank recompute (doc SEG/SEGS clarifier)
│   │   ├── corpus.rs          # SPRINT-4 + SEC-H7: collect_files_inner + COLLECT_FILES_MAX_DEPTH=32
│   │   ├── kv_compressed.rs   # Stream I: Delta+LZ4 + MUD_KV_COMPRESS_CELLV5 reserved
│   │   ├── debate_trainer.rs  # F1 Fenwick nucleus sampler (per-doppel pre-alloc)
│   │   ├── corpus_trainer/    # 7-module split: mod/metrics/emb/integrity/alignment/debate/circuit/session/train_step
│   │   ├── circuit_rpg.rs     # RPG lineage + titles + battle_hp (2026-07-21)
│   │   ├── worm.rs            # MUD-WORM (WormPropagator + WormholeRouter/Network, 30 tests)
│   │   ├── boot_indexer.rs    # SX-05 AOT boot indexer
│   │   ├── memory_continuity.rs # SX-04 256-slot memory continuity (L1 O(1))
│   │   ├── file_fusion.rs     # SX-06 MudFileMerger
│   │   ├── smart_tokenizer.rs # ST-01 zero-alloc streaming BPE
│   │   ├── cmud.rs            # L-14 C-MUD thinker (ThinkingState + trainable sidecar)
│   │   ├── slime_x.rs         # SX-07 SlimeX Dynamic Stack + ShadowExpertBus
│   │   └── …                   # slime, MoE, CSA, packing, JEPA… (407 lib tests)
│   ├── vulkan/                # ash 0.38 backend + gemv_policy + D5 elementwise/combined dispatchers
│   ├── main.rs                # Inference CLI (Fenwick nucleus sampling + chat history)
│   └── …
│
├── tools/                     # Bins: run_trainer, audit, converter, benches (asm/fenwick/pipeline/mhc_stress)
├── assets/shaders/            # GLSL + spirv (ternary_gemv, elementwise, ternary_backward, …)
├── training/corpus/           # .txt/.md + project_corpus.txt (assembled by mud.sh train)
├── models/                    # .mud checkpoints (smollm2.mud, ternary_bonsai_1.7b.mud, …)
├── forge_autograd/            # Isolated autograd crate (avx_math.rs reused by STP/Muon)
│
└── docs/
    ├── README.md              # Docs index
    ├── STATUS_REPORT.md       # Logros vs deuda (was root)
    ├── IMPLEMENTATION_PLAN*.md  # Cross-session implementation plans
    ├── architecture/           # Compute stack, register bit-packing, ASM docs, SlimeX phase1 log
    │   ├── MUD_TRAINER_TERNARY_JEPA_MHC.md   # §9 F1/F2 verified · §10 unified UI
    │   ├── MUD_COMPUTE_STACK.md
    │   └── …
    ├── research/
    │   ├── MUD_PLAN_MHC_STP_TRAINABLE.md     # F1/F2 plan (Phase 1+2 DONE; Phase 3 n=2 deferred)
    │   ├── MUD_FENWICK_TREE_OPTIMIZATION_PLAN.md   # F4/F1/F5b (DONE); F2/F3 deferred
    │   ├── MUD_VULKAN_DELEGATION_ANALYSIS_2026-07-26.md
    │   ├── MUD_GPU_SIGNAL_KEEPALIVE_PLAN_2026-07-26.md   # BATCH E deferred (HW sign-off + dL/dX shader)
    │   ├── CELLV5_CMUD_OPTIMIZATION_PLAN.md   # BATCH D (5.1/5.2 deferred)
    │   ├── MUD_IMPROVEMENTS_POST_AE.md       # F+ backlog (F-L DONE)
    │   └── …
    ├── planning/               # **MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md (backlog activo R-##)**, PHASE_D_TRAINING_SCALE_PLAN, CELLV5_CMUD_OPTIMIZATION_PLAN
    ├── seguridad/              # Sprints 6-7 docs (THREAT_MODEL, HARDENING, INCIDENT_RESPONSE)
    ├── audits/                 # **MUD_AUDIT_2026-10-11_LOW_COMPUTE.md (vigente)** + FULL_2026-08-03 + V01–V38 + BUG_LOGITS_CORRUPTION
    │   └── MUD_AUDIT_LATEST.md   # puntero a la auditoría vigente + plan maestro (2026-10-11)
    ├── sessions/               # MUD_SESSION_REPORT_YYYY-MM-DD
    │   └── MUD_SESSION_REPORT_2026-10-13_OLA2_OLA3.md   # Ola 2 completa + R-40/R-48
    ├── manuals/                # LAUNCH_COUNTDOWN, MUD_TRAINING_PROTOCOLS, HOUSEKEEPING_PLAN
    └── dumps/                  # tmp dumps
```

## Recent work (2026-07-26 — Batch A/B/C/D-partial)

| Item | File | Status |
|------|------|--------|
| **F1** Fenwick nucleus sampling en `debate_trainer` | `debate_trainer.rs` (+4 Doppelganger pre-alloc buffers) | ✅ DONE |
| **F5b** `CertMode::Percentile` + `percentile_loss` | `loss_cert.rs` (FenwickU32 histogram) + 6 tests | ✅ DONE |
| **Muon/GaLore/Chunked** scratch reuse + in-place NS | `muon.rs`, `galore.rs` (elim `grad.to_vec()` + lifted scratch) | ✅ DONE |
| **SEC-H1** caps `MudFile::load` | `mod.rs` (`MAX_META_ENTRIES`, `MAX_TENSOR_ENTRIES`, `MAX_STRING_LEN`, `MAX_SHAPE_DIMS`) + 3 tests | ✅ DONE |
| **SEC-H8** `MUD_CIRCUIT_MAX_CYCLES` cap | `corpus_trainer/circuit.rs` | ✅ DONE |
| **SEC-H7** `COLLECT_FILES_MAX_DEPTH=32` | `corpus.rs` (depth-bounded recursion) + 1 test | ✅ DONE |
| **PERF-7** tied-output zero copy | `inference.rs` (`std::mem::take` después del forward; ~295 MB saved smollm2) | ✅ DONE |
| **run_trainer** canonical checkpoint dir | `run_trainer.rs` + `MUD_TRAIN_CKPT_DIR` / `MUD_TRAIN_CORPUS_DIR` env | ✅ DONE |
| **MUD_KV_COMPRESS_CELLV5** env reserve | `mud.sh` + `kv_compressed.rs` (sync warning until Fase-3) | ✅ DONE (plumbing) |
| **5.1** Cellular AVX2 kernels | `src/asm/cmud_cell_v5_avx2.s` (NO escrito) | 🟡 DEFERRED (riesgo diseño/validación) |
| **5.2** Cellular KV pre-compression | `CompressedKvCache` CellV5 path (NO escrito) | 🟡 DEFERRED (invariante lossless) |
| **Vulkan D5** `step_elementwise_sync` integration | `slime_jepa.rs`, `stp_loss.rs` (NO conectado) | 🟡 DEFERRED (HW sign-off + dL/dX shader) |

**Estado verificado 2026-10-11:** `cargo test --lib` (debug) 461 ✅ · `cargo test --lib --release` **460 + 1 FALLO (UB fenwick)** ❌ · clippy `--all-targets` 0 warnings ✅
Backlog activo: `docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md` · Auditoría: `docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`
Documentación de sesión (última): `docs/sessions/MUD_SESSION_REPORT_2026-10-13_OLA2_OLA3.md` (histórico: `..._2026-07-26_PENDIENTES.md`)

## Root policy (P-18)

**Allowed at repo root:** `GEMINI.md`, `AGENTS.md`, `ROADMAP.md`, `README.md`, `TREE.md`, `LICENSE`.

**Moved 2026-07-16:** status report, ASM/Vulkan plans, audits, housekeeping → `docs/`.

