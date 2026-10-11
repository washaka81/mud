# MUD Directory Structure

> **Última actualización:** 2026-07-26 (post-consolidación de documentación)
> **Nota:** Este archivo refleja la estructura actual del repositorio. Archivos obsoletos han sido eliminados.

## Raíz del repositorio

```text
├── GEMINI.md                  # SSOT: políticas (P-#), estado, ledger L-##
├── AGENTS.md                  # Contexto para agentes (derivado de GEMINI)
├── ROADMAP.md                 # Visión de producto + fases + narrativa arquitectónica (consolidado 2026-07-26)
├── README.md                  # Intro pública del proyecto
├── TREE.md                    # Esta estructura (resumen)
├── CHANGELOG.md               # Historial de cambios
├── LICENSE
├── Cargo.toml / Cargo.lock / build.rs
├── mud.sh                     # Orquestador CLI (train/align/circuit/audit-full/health/…)
├── .gitignore / .cargo/
├── assets/shaders/            # Vulkan compute shaders (.comp)
├── benches/                   # Benchmarks
├── forge_autograd/            # Subcrate de autograd (Rust)
├── src/                       # ── Core Engine ──
│   ├── main.rs                # TUI inference
│   ├── lib.rs                 # Library root
│   ├── hardware.rs            # CPU feature detection
│   ├── asm/                   # AVX2 assembly kernels
│   ├── bin/                   # Binary tools
│   ├── gguf/                  # GGUF parsing
│   ├── model/                 # Tokenizer + model
│   ├── mud/                   # MUD core (forward, backward, trainer, etc.)
│   ├── vulkan/                # ash Vulkan backend
│   └── vulkan/                # ash Vulkan backend
├── tools/                     # ── Herramientas ──
│   ├── universal_converter/   # Conversor GGUF→.mud
│   ├── run_trainer.rs         # Trainer entry point
│   ├── forge_llm.rs           # Forge model from scratch
│   ├── mud_forge.rs           # Forge model from scratch
│   ├── gemv_auto_bench.rs     # GEMV break-even bench
│   ├── mud_full_audit.rs      # Full audit
│   ├── diagnose_model.rs      # Model diagnostics
│   ├── scale_audit.rs         # Scale audit (P1.1)
│   └── ... (60+ tools)
├── tests/                     # Integration tests
├── docs/                      # ── Documentación ── (ver abajo)
├── training/                  # Corpus + vocab
└── models/                    # Modelos .mud (no en repo)
```

## `docs/` — Estructura actualizada

```text
docs/
├── README.md                  # Índice de documentación
├── STATUS_REPORT.md           # Logros vs deuda (resumen)
├── ROADMAP.md                 # (redirect → ../ROADMAP.md)
├── MUD_HISTORIAL_CONVERSACIONES_ANTIGRAVITY.md  # Histórico de conversaciones
├── IMPLEMENTATION_PLAN_V2.md  # Plan de implementación post-SlimeX/CMUD
├── architecture/              # Specs y stack técnico
│   ├── MUD_COMPUTE_STACK.md   # Stack de cómputo LIVE (SSOT técnico)
│   ├── MUD_REGISTER_BIT_PACKING.md  # Bit packing de registros
│   ├── MUD_TRAINER_TERNARY_JEPA_MHC.md  # Trainer JEPA/mHC (§9 F1/F2)
│   ├── SLIMEX_PHASE1_ASM_POINTERS_LOG.md  # SlimeX Phase 1 log
│   ├── ASM_KERNEL_DOCUMENTATION.md  # Documentación de kernels ASM
│   └── ROADMAP.md             # (redirect → ../../ROADMAP.md)
├── audits/                    # Auditorías históricas (V02–V38)
│   ├── MUD_AUDIT_LATEST.md    # V38 — Post-Sprint 5 (audit canónico)
│   ├── MUD_AUDIT_REPORT_V02_TERNARY_SHOCK.md
│   ├── MUD_AUDIT_REPORT_V03_PTQ_FAILURE.md
│   ├── MUD_AUDIT_REPORT_V04_AUTOTRAINER.md
│   ├── MUD_AUDIT_REPORT_V06_QAT_SOLUTION.md
│   ├── MUD_AUDIT_REPORT_V07_DEEP_MATH.md
│   ├── MUD_AUDIT_REPORT_V09_MATH_STABILIZATION.md
│   ├── MUD_AUDIT_REPORT_V10.md
│   ├── MUD_AUDIT_REPORT_V10_COGNITIVE_RECOVERY.md
│   ├── MUD_AUDIT_REPORT_V11_HOLOGRAPHIC_ALIGNMENT.md
│   ├── MUD_AUDIT_REPORT_V12_BITNET_RESTORATION.md
│   ├── MUD_AUDIT_REPORT_V13.md
│   ├── MUD_AUDIT_REPORT_V13_TOKENIZER_APHASIA.md
│   ├── MUD_AUDIT_REPORT_V14_LDT_GRPO.md
│   ├── MUD_AUDIT_REPORT_V15_INT8_AVX2.md
│   ├── MUD_AUDIT_REPORT_V16_DYNAMIC_INFERENCE.md
│   ├── MUD_AUDIT_REPORT_V17_BFLOAT16.md
│   ├── MUD_AUDIT_REPORT_V17_JEPA_ANTICIPATION.md
│   ├── MUD_AUDIT_REPORT_V18_BITNET_CONVERSION_CRISIS.md
│   ├── MUD_AUDIT_REPORT_V18_GRPO_LDT.md
│   ├── MUD_AUDIT_REPORT_V19_TEST_TIME_COMPUTE_AND_RLVR.md
│   ├── MUD_AUDIT_REPORT_V20.md
│   ├── MUD_AUDIT_REPORT_V21_QAT_RECOVERY.md
│   ├── MUD_AUDIT_REPORT_V22_WORKSPACE.md
│   ├── MUD_AUDIT_REPORT_V23_AGENTS.md
│   ├── MUD_AUDIT_REPORT_V24_MEMORY.md
│   ├── MUD_AUDIT_REPORT_V25_COMPREHENSIVE.md
│   ├── MUD_AUDIT_REPORT_V27_ALL_ZEROS_RESOLVED.md
│   ├── MUD_AUDIT_REPORT_V28_JEPA_SPIRAL.md
│   ├── MUD_AUDIT_REPORT_V29_TERNARY_FORMAT_MISMATCH.md
│   ├── MUD_AUDIT_REPORT_V30_JEPA_SCALE_MISMATCH.md
│   ├── MUD_AUDIT_REPORT_V31_INFERENCE_PERFORMANCE.md
│   ├── MUD_AUDIT_REPORT_V32_SYNTHETIC_SELF_PLAY.md
│   ├── MUD_AUDIT_REPORT_V33_HMP_AND_HOMEOSTASIS.md
│   ├── MUD_AUDIT_REPORT_V34_DUAL_STATE_SLIMEREGISTER.md
│   ├── MUD_AUDIT_REPORT_V35_ELUT_4BIT_TOOLING_PURGE.md
│   ├── MUD_AUDIT_REPORT_V36_EZOP.md
│   ├── MUD_AUDIT_REPORT_V38_POST_SPRINT_5.md
│   ├── MUD_AUDIT_TRAINING_ASSIMILATION.md
│   ├── MUD_CONVERSION_AUDIT.md
│   ├── MUD_QUALITY_AUDIT_BATTERY.md
│   ├── MUD_SYSTEM_PERFORMANCE_AND_COGNITIVE_GAINS_2026-07-23.md
│   ├── MUD_MASTER_CELL_V5_AND_WORM_GAIN_CALCULATIONS_2026-07-23.md
│   ├── OPTIMAL_U32_BIT_ALLOCATION_RATIO_PROOF.md
│   ├── BUG_LOGITS_CORRUPTION.md
│   ├── CHAT_REFACTOR_REPORT.md
│   ├── LEELA_READING_ARCHITECTURE.md
│   ├── ANTIGRAVITY_CHAT_SUMMARY_AND_PROJECT_DIRECTION.md
│   ├── PROJECT_DIRECTION_ANTIGRAVITY_LEELA.md
│   ├── ASM_AUDIT_REPORT.md
│   ├── AUDIT_REPORT.md
│   └── UNIVERSAL_CONVERTER_AUDIT_FINDINGS.md
├── dumps/                    # Dumps de debug (txt/log)
│   ├── mud_disassembly_2026-07-23.txt
│   ├── mud_disassembly_latest.txt
│   ├── mud_disassembly.txt
│   ├── dump_2026-07-23.txt
│   ├── qat_restore.log
│   ├── restore_iq.log
│   └── restore_iq_full.log
├── dumps_archive/            # Dumps archivados (históricos)
│   ├── ai_disassembly.txt
│   ├── audit_build_log.txt
│   ├── bitnet_mud_dump.txt
│   ├── conversion_manifest.txt
│   ├── fix_tokenizer.patch
│   ├── full_source_tensors.txt
│   └── tokens_dump.txt
├── manuals/                  # Protocolos y manuales
│   ├── HOUSEKEEPING_PLAN.md
│   ├── LAUNCH_COUNTDOWN.md
│   ├── MUD_CALIBRATION_PROTOCOL.md
│   ├── MUD_DIRECTORY_STRUCTURE.md  # Este archivo
│   ├── MUD_GUIDELINES.md
│   ├── MUD_ROADMAP_MASTER.md       # (redirect → ../../ROADMAP.md)
│   ├── MUD_TRAINING_PROTOCOLS.md
│   └── MUD_USER_MANUAL.md
├── planning/                 # Planes de implementación
│   ├── PHASE_D_TRAINING_SCALE_PLAN.md
│   └── CELLV5_CMUD_OPTIMIZATION_PLAN.md
├── research/                 # Papers, gap analysis, mejoras F+
│   ├── MUD_IMPROVEMENTS_POST_AE.md  # Siguiente backlog F–L
│   ├── MUD_GAP_ANALYSIS_POST_L15.md  # Residuales P0–P3 + env knobs
│   ├── MUD_VULKAN_DELEGATION_ANALYSIS_2026-07-26.md
│   ├── MUD_GPU_SIGNAL_KEEPALIVE_PLAN_2026-07-26.md
│   ├── MUD_FENWICK_TREE_OPTIMIZATION_PLAN.md
│   ├── MUD_FENWICK_ASM_POINTER_ANALYSIS.md
│   ├── MUD_PLAN_CIRCUIT_ALGORITHMS.md
│   ├── MUD_ULTRA_COMPRESSED_REGISTER_AND_MEMORY_CONTINUITY_MASTER.md
│   ├── MUD_MEMORY_CONTINUITY_TABLE_AND_FILE_FUSION.md
│   ├── MUD_ENGINE_BOOT_INDEXER_SPECIFICATION.md
│   ├── MUD_WORM_IMPLEMENTATION.md
│   ├── MUD_LATENT_NEURAL_MEMORY_WORM_ARCHITECTURE.md
│   ├── DEEPSEEK_DSPARK_RESEARCH_2026-07-20.md
│   ├── CMUD_THINKER_INTEGRATION_MASTER_EQUATION.md
│   ├── MUD_PLAN_MHC_STP_TRAINABLE.md
│   ├── MUD_PLAN_RLVR_DEBATE_SEED.md
│   ├── MUD_PLAN_PRIORIDADES_ROADMAP_2026-07-20.md  # (redirect → ../../ROADMAP.md)
│   ├── RESEARCH_PAPERS.md    # Índice de papers (53 papers, 5 dominios)
│   ├── SAMPLE_EFFICIENCY_TRAINING.md
│   ├── TRAIN_TELEMETRY_FORENSIC_2026-07-20.md
│   ├── ARXIV_TERNARY_MAIN_PAGE_REFERENCE.md
│   ├── ASM_POINTER_REGISTER_OPTIMIZATION.md
│   ├── BONSAI_MODELS_AND_MUD.md
│   ├── CFT_LOGGAS_VERTEX_OPERATOR_NOTE.md
│   ├── CMUD_LOGGAS_FEASIBILITY.md
│   ├── COMPLEX_THINKING_MANIFOLD.md
│   ├── DEEP_QAT_AND_SELF_PLAY_MANIFESTO.md
│   ├── DEEPSEEK_V4_TERNARY_INTEGRATION.md
│   ├── GITHUB_RUST_PAPERS.md
│   ├── low_cost_efficient_training.md
│   ├── MUD_TECH_RESEARCH_2026.md
│   ├── MUD_WHITE_PAPER.md
│   ├── PACKED_REGISTER_V3_OPTIMIZATION.md
│   ├── RESEARCH_AGENT_DISTILLATION.md
│   ├── RESEARCH_RECURSIVE_MODELS.md
│   ├── RESEARCH_SLEEP_FOLDING.md
│   └── RUST_LLM_TERNARY_ECOSYSTEM.md
├── seguridad/                # Documentos de seguridad (Sprints 6-7)
│   ├── THREAT_MODEL.md
│   ├── HARDENING.md
│   └── INCIDENT_RESPONSE.md
└── sessions/                 # Reportes de sesión (históricos)
    ├── MUD_SESSION_REPORT_2026-07-26_PENDIENTES.md  # Batch A/B/C/D-partial
    ├── MUD_SESSION_REPORT_2026-07-25.md
    ├── MUD_SESSION_REPORT_2026-07-24.md
    ├── MUD_SESSION_REPORT_2026-07-24_SPRINT_5.md
    ├── MUD_SESSION_REPORT_2026-07-24_REMEDIATION.md
    ├── MUD_SESSION_REPORT_2026-07-24_TEST_FIXES.md
    ├── MUD_SESSION_REPORT_2026-07-24_TRAINER_REVIEW.md
    ├── MUD_SESSION_REPORT_2026-07-23.md
    ├── MUD_SESSION_REPORT_2026-07-23_PART2_SLIMEX_CMUD.md
    ├── MUD_SESSION_REPORT_2026-07-22.md
    ├── MUD_SESSION_REPORT_2026-07-21.md
    ├── MUD_SESSION_REPORT_2026-07-20.md
    ├── MUD_SESSION_REPORT_2026-07-18.md
    ├── MUD_SESSION_REPORT_2026-07-16_STREAMS_AE.md
    ├── MUD_SESSION_REPORT_2026-07-16_PHASE_B.md
    ├── MUD_SESSION_REPORT_2026-07-16_PHASE_BPLUS.md
    ├── MUD_SESSION_REPORT_2026-07-16.md
    ├── MUD_SESSION_REPORT_2026-07-16_L05_L06.md
    ├── MUD_SESSION_REPORT_2026-07-16_L09_EZOP.md
    ├── MUD_SESSION_REPORT_2026-07-16_L10_PACKING.md
    ├── MUD_SESSION_REPORT_2026-07-16_L11_MOE.md
    ├── MUD_SESSION_REPORT_2026-07-16_L12_CI.md
    ├── MUD_SESSION_REPORT_2026-07-16_L13_HCA.md
    ├── MUD_SESSION_REPORT_2026-07-16_L14_CMUD.md
    ├── MUD_SESSION_REPORT_2026-07-16_L15_AUDIT_RESEARCH.md
    ├── MUD_SESSION_REPORT_2026-07-04.md
    ├── MUD_SESSION_REPORT_2026-07-01.md
    ├── MUD_SESSION_REPORT_2026-06-30.md
    ├── MUD_SESSION_REPORT_2026-06-29.md
    ├── MUD_SESSION_REPORT_2026-06-28.md
    ├── MUD_SESSION_REPORT_2026-06-27.md
    ├── MUD_SESSION_REPORT_2026-06-24.md
    ├── MUD_SESSION_REPORT_2026-06-22.md
    ├── MUD_SESSION_REPORT_2026-06-20_ALL_ZEROS_BUG.md
    ├── MUD_SESSION_REPORT_2026-06-19_2.md
    ├── MUD_SESSION_REPORT_2026-06-19.md
    ├── MUD_SESSION_REPORT_2026-06-18_PHASE_SINGULARITY.md
    ├── MUD_SESSION_REPORT_2026-06-18_PHASE5-6.md
    ├── MUD_SESSION_REPORT_2026-06-18_PHASE12.md
    ├── MUD_SESSION_REPORT_2026-06-18_HOUSEKEEPING.md
    ├── MUD_SESSION_REPORT_2026-06-18.md
    ├── MUD_SESSION_REPORT_2026-06-17.md
    ├── MUD_SESSION_REPORT_2026-06-15.md
    ├── MUD_SESSION_REPORT_2026-06-14.md
    ├── MUD_SESSION_REPORT_2026-06-13.md
    ├── MUD_SESSION_REPORT_2026-06-11.md
    ├── MUD_SESSION_REPORT_2026-06-10_PHASE2.md
    ├── MUD_SESSION_REPORT_2026-06-10_FINAL.md
    ├── MUD_SESSION_REPORT_2026-06-10.md
    ├── MUD_SESSION_REPORT_2026-06-10_COGNITIVE_RESTORATION.md
    ├── MUD_SESSION_REPORT_2026-06-09_FIXES_AND_VULKAN_DEDUP.md
    ├── MUD_SESSION_REPORT_2026-06-08_QAT_STABILITY.md
    ├── MUD_SESSION_REPORT_2026-06-08_MEMORY_AUDIT.md
    ├── MUD_SESSION_REPORT_2026-06-08_LQAT_INTEGRATION.md
    ├── MUD_SESSION_REPORT_2026-06-08_ASM_CACHE_AUDIT.md
    ├── MUD_SESSION_REPORT_2026-06-08.md
    ├── MUD_SESSION_REPORT_2026-06-05.md
    ├── MUD_SESSION_REPORT_2026-06-04.md
    ├── MUD_SESSION_REPORT_2026-06-03.md
    ├── MUD_SESSION_REPORT_2026-05-25.md
    ├── MUD_AUDIT_ACHIEVEMENTS_2026-07-20.md
    ├── MUD_OPTIMIZATION_LOG.md
    ├── SESSION_SUMMARY.md
    ├── SPRINT_2_PLAN.md
    ├── SPRINT_3_PLAN.md
    ├── SPRINT_4_PLAN.md
    └── TECH_LOG.md
```

## Archivos obsoletos eliminados (2026-07-26)

| Categoría | Archivos eliminados |
|-----------|-------------------|
| Audits | V01 (histórico), MUD_MATHEMATICAL_AUDIT, MUD_STATISTICAL_AUDIT, V26_BACKUP, V25_WARP_PERF |
| Architecture | 37 archivos obsoletos (SLIME_REGISTER_V2, MUD_ARCHITECTURE, MUD_MASTER_MANIFESTO, ONEBIT_RESEARCH, ternary_diffusion_math, MUD_TERNARY_ISA, MUD_HARDWARE_ISA, MUD_POINTER_STRATEGY, MUD_ISA_DISPATCH, MUD_DATA_ARCHITECTURE, MUD_DELEGATION_ROUTER, MUD_COGNITIVE_ARCH, MUD_MEMORY_CACHE, MUD_MOE_EXPERTS, MUD_ORCHESTRATION, MUD_HOLOGRAPHIC_PHASE_ALIGNMENT, MUD_ULTIMATE_PRECISION, MUD_TRAINER_SLIMEX_DYNAMIC, MUD_RRM_MICROKERNELS, ORNITH_HYBRID_ARCH, SYSTEM_COMPLETE, HYBRID_ELUT_JEPA_PLAN, JEPA_TTC_32BIT_PLAN, MUD_1_TO_1_CONVERSION_PLAN, MUD_COMPREHENSIVE_RESEARCH, MUD_EMBED_TERNARIZATION, BITNET_HAMMING_CODES, BITNET_SPEC, ENGINE_MANIFESTO, ASM_VULKAN_BOTTLENECK_2026-07, ASM_CORRECTION_PLAN, VULKAN_AVX2_THREADS_OPTIMIZATION, MUD_OVERVIEW, ASM_OPTIMIZATION_PLAN, MUD_KERNEL_PLAN, MUD_SLIMEREGISTER_PLAN, SLIMEX_LLM_SPECIFICATION) |
| Manuals | MUD_V1_MASTER_REPORT, MUD_VS_OXILLAMA, UPDATE_2026-07-22, MUD_UNIVERSAL_PROTOCOL_V2, MUD_CRITICOS_MAXIMOS, MUD_OPTIMIZATION_ANALYSIS |
| Research | PAPERS_RESEARCH, MUD_FIX_PLAN_2026-07-20, PLAN_PROXIMO_PASO, jepa_anticipation_paradigm, JEPA_DOPPLER_FILTER, HMP_MATHEMATICAL_PROOF, HMP_VULKAN_ASYNC_OFFLOAD, diffusion_gemma_research, ternary_coding_paradigms, TRAINING_SPEED_ADAPTIVE, MUD_QAT_RESTRUCTURING_V11, MUD_PLAN_CORRECCIONES_OPTIMIZACIONES_V38, JEPA_FP32_PRECISION_CHAIN, JEPA_LEXICAL_RESONANCE, MAI_THINKING_LOW_END_INTEGRATION, CMUD_VULKAN_VS_AVX2, COMPLEX_REASONING_* (3 archivos) |
| Planning | PLAN_DIA_2_2026-07-26 |
| Root | mud_disassembly.txt (stray P-18 violation) |
| **Total** | **59 archivos eliminados** |

## Jerarquía de documentos

| Documento | Rol | Autoridad |
|-----------|-----|-----------|
| `GEMINI.md` | Políticas, estado, ledger | **SSOT** técnico |
| `AGENTS.md` | Contexto para agentes | Derivado de GEMINI |
| `ROADMAP.md` | Visión + fases + narrativa | **SSOT** dirección |
| `docs/architecture/MUD_COMPUTE_STACK.md` | Stack de cómputo | Referencia técnica |
| `docs/architecture/MUD_REGISTER_BIT_PACKING.md` | Bit packing | Referencia técnica |
