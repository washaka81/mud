# MUD Directory Structure\n\n```text
.
├── .gitignore
├── .mud_env
├── Cargo.lock
├── Cargo.toml
├── GEMINI.md
├── LICENSE
├── README.md
├── assets
│   └── shaders
│       ├── ghost_align.comp
│       ├── heartbeat.comp
│       ├── shadow_optimizer.comp
│       ├── silu_gate.comp
│       ├── simple_mul.comp
│       ├── ternary_backward.comp
│       ├── ternary_backward_opt.comp
│       ├── ternary_gemv_unified.comp
│       └── vulkan_keepalive.comp
├── benches
│   └── generate_diffusion.rs
├── build.rs
├── docs
│   ├── README.md
│   ├── ROADMAP.md
│   ├── architecture
│   │   ├── BITNET_HAMMING_CODES.md
│   │   ├── BITNET_SPEC.md
│   │   ├── ENGINE_MANIFESTO.md
│   │   ├── MUD_ARCHITECTURE.md
│   │   ├── MUD_COGNITIVE_ARCH.md
│   │   ├── MUD_COMPREHENSIVE_RESEARCH.md
│   │   ├── MUD_DATA_ARCHITECTURE.md
│   │   ├── MUD_DELEGATION_ROUTER.md
│   │   ├── MUD_EMBED_TERNARIZATION.md
│   │   ├── MUD_MASTER_MANIFESTO.md
│   │   ├── MUD_MOE_EXPERTS.md
│   │   ├── MUD_ORCHESTRATION.md
│   │   ├── MUD_OVERVIEW.md
│   │   ├── MUD_SYSTEM_UPGRADE_V1.5.md
│   │   ├── MUD_ULTIMATE_PRECISION.md
│   │   ├── ONEBIT_RESEARCH.md
│   │   └── ternary_diffusion_math.md
│   ├── audits
│   │   ├── BUG_LOGITS_CORRUPTION.md
│   │   ├── MUD_AUDIT.md
│   │   ├── MUD_AUDIT_LATEST.md
│   │   ├── MUD_AUDIT_REPORT_V01.md
│   │   ├── MUD_AUDIT_REPORT_V02_TERNARY_SHOCK.md
│   │   ├── MUD_AUDIT_REPORT_V03_PTQ_FAILURE.md
│   │   ├── MUD_AUDIT_REPORT_V04_AUTOTRAINER.md
│   │   ├── MUD_AUDIT_REPORT_V06_QAT_SOLUTION.md
│   │   ├── MUD_AUDIT_REPORT_V07_DEEP_MATH.md
│   │   ├── MUD_AUDIT_REPORT_V09_MATH_STABILIZATION.md
│   │   ├── MUD_AUDIT_REPORT_V10.md
│   │   ├── MUD_AUDIT_REPORT_V10_COGNITIVE_RECOVERY.md
│   │   ├── MUD_AUDIT_REPORT_V11_HOLOGRAPHIC_ALIGNMENT.md
│   │   ├── MUD_AUDIT_REPORT_V12_BITNET_RESTORATION.md
│   │   ├── MUD_AUDIT_REPORT_V13.md
│   │   ├── MUD_AUDIT_REPORT_V13_TOKENIZER_APHASIA.md
│   │   ├── MUD_AUDIT_REPORT_V14_LDT_GRPO.md
│   │   ├── MUD_AUDIT_REPORT_V15_INT8_AVX2.md
│   │   ├── MUD_AUDIT_RESOLUTION.md
│   │   ├── MUD_CONVERSION_AUDIT.md
│   │   ├── MUD_FINAL_AUDIT_REPORT.txt
│   │   ├── MUD_MATHEMATICAL_AUDIT.md
│   │   ├── MUD_STATISTICAL_AUDIT.md
│   │   └── inference_results.md
│   ├── dumps
│   │   ├── ai_disassembly.txt
│   │   ├── audit_build_log.txt
│   │   ├── chat_out.txt
│   │   ├── conversion_manifest.txt
│   │   ├── dumper_output.txt
│   │   ├── fix_tokenizer.patch
│   │   ├── full_source_tensors.txt
│   │   ├── mud_disassembly.txt
│   │   ├── scratch.py
│   │   ├── tokens_dump.txt
│   │   └── traces.jsonl
│   ├── hardware
│   │   ├── MUD_HARDWARE_ISA.md
│   │   ├── MUD_ISA_DISPATCH.md
│   │   ├── MUD_KERNEL_PLAN.md
│   │   ├── MUD_MEMORY_CACHE.md
│   │   ├── MUD_POINTER_STRATEGY.md
│   │   ├── MUD_RRM_MICROKERNELS.md
│   │   └── MUD_TERNARY_ISA.md
│   ├── manuals
│   │   ├── MUD_CALIBRATION_PROTOCOL.md
│   │   ├── MUD_CRITICOS_MAXIMOS.md
│   │   ├── MUD_DIRECTORY_STRUCTURE.md
│   │   ├── MUD_GUIDELINES.md
│   │   ├── MUD_OPTIMIZATION_ANALYSIS.md
│   │   ├── MUD_ROADMAP.md
│   │   ├── MUD_TRAINING_PROTOCOLS.md
│   │   ├── MUD_UNIVERSAL_PROTOCOL_V2.md
│   │   ├── MUD_USER_MANUAL.md
│   │   ├── MUD_V1_MASTER_REPORT.md
│   │   └── MUD_VS_OXILLAMA.md
│   ├── mud_disassembly.txt
│   ├── research
│   │   ├── 2604.27396.pdf
│   │   ├── ARXIV_TERNARY_MAIN_PAGE_REFERENCE.md
│   │   ├── GITHUB_RUST_PAPERS.md
│   │   ├── MUD_TECH_RESEARCH_2026.md
│   │   ├── MUD_WHITE_PAPER.md
│   │   ├── PAPERS_RESEARCH.md
│   │   ├── RESEARCH_AGENT_DISTILLATION.md
│   │   ├── RESEARCH_PAPERS.md
│   │   ├── RESEARCH_RECURSIVE_MODELS.md
│   │   ├── RESEARCH_SLEEP_FOLDING.md
│   │   ├── RUST_LLM_TERNARY_ECOSYSTEM.md
│   │   ├── diffusion_gemma_research.md
│   │   ├── jepa_anticipation_paradigm.md
│   │   └── low_cost_efficient_training.md
│   └── sessions
│       ├── MUD_OPTIMIZATION_LOG.md
│       ├── MUD_SESSION_REPORT_2026-05-25.md
│       ├── MUD_SESSION_REPORT_2026-06-03.md
│       ├── MUD_SESSION_REPORT_2026-06-04.md
│       ├── MUD_SESSION_REPORT_2026-06-05.md
│       ├── MUD_SESSION_REPORT_2026-06-08.md
│       ├── MUD_SESSION_REPORT_2026-06-08_ASM_CACHE_AUDIT.md
│       ├── MUD_SESSION_REPORT_2026-06-08_LQAT_INTEGRATION.md
│       ├── MUD_SESSION_REPORT_2026-06-08_MEMORY_AUDIT.md
│       ├── MUD_SESSION_REPORT_2026-06-08_QAT_STABILITY.md
│       ├── MUD_SESSION_REPORT_2026-06-09_FIXES_AND_VULKAN_DEDUP.md
│       ├── MUD_SESSION_REPORT_2026-06-10.md
│       ├── MUD_SESSION_REPORT_2026-06-10_FINAL.md
│       ├── MUD_SESSION_REPORT_2026-06-10_PHASE2.md
│       ├── MUD_SESSION_REPORT_2026-06-11.md
│       ├── MUD_SESSION_REPORT_2026-06-13.md
│       ├── MUD_SESSION_REPORT_2026-06-14.md
│       ├── MUD_SESSION_REPORT_2026-06-15.md
│       ├── MUD_SESSION_REPORT_2026-06-17.md
│       ├── MUD_SESSION_REPORT_2026-06-18.md
│       ├── MUD_SESSION_REPORT_2026_06_10.md
│       ├── SESSION_SUMMARY.md
│       └── TECH_LOG.md
├── forge_autograd
│   ├── Cargo.lock
│   ├── Cargo.toml
│   └── src
│       ├── avx_math.rs
│       └── lib.rs
├── mud.sh
├── senior_skill
│   ├── SKILL.md
│   └── references
│       ├── example_reference.md
│       └── mud-spec.md
├── src
│   ├── asm
│   │   ├── adam_step.s
│   │   ├── mamba.s
│   │   ├── math.s
│   │   ├── mod.rs
│   │   ├── q4_0_gemv.s
│   │   ├── rmsnorm.s
│   │   ├── rope.s
│   │   ├── sgemm.s
│   │   ├── silu.s
│   │   ├── ternary_gemm_batch4.s
│   │   ├── ternary_gemv.s
│   │   ├── ternary_gemv_4rows.s
│   │   ├── ternary_lut.s
│   │   ├── ternary_pext.s
│   │   └── tests.rs
│   ├── bin
│   │   ├── bench_sgemm.rs
│   │   └── test_inf.rs
│   ├── gguf
│   │   ├── dump_metadata.rs
│   │   ├── dump_tensors.rs
│   │   ├── mod.rs
│   │   ├── tests.rs
│   │   └── tokenizer_test.rs
│   ├── hardware.rs
│   ├── lib.rs
│   ├── main.rs
│   ├── model
│   │   ├── mod.rs
│   │   ├── tokenizer.rs
│   │   ├── tokenizer.rs.orig
│   │   ├── tokenizer.rs.rej
│   │   └── tokenizer_test.rs
│   ├── mud
│   │   ├── corpus_trainer.rs
│   │   ├── dspy.rs
│   │   ├── ecc.rs
│   │   ├── forward.rs
│   │   ├── inference.rs
│   │   ├── ldt_micro.rs
│   │   ├── mod.rs
│   │   ├── routing.rs
│   │   ├── sampling.rs
│   │   ├── skills
│   │   │   ├── autoformatter.rs
│   │   │   ├── code_formatter.rs
│   │   │   ├── coding.rs
│   │   │   ├── data_analysis.rs
│   │   │   ├── language.rs
│   │   │   ├── learning.rs
│   │   │   ├── logic_marks.rs
│   │   │   ├── logic_math.rs
│   │   │   ├── memory.rs
│   │   │   ├── mod.rs
│   │   │   ├── personality.rs
│   │   │   ├── plotting.rs
│   │   │   ├── retrieval.rs
│   │   │   ├── text_styling.rs
│   │   │   ├── translator.rs
│   │   │   └── web_search.rs
│   │   ├── speculative.rs
│   │   ├── tests.rs
│   │   └── workspace.rs
│   └── vulkan
│       ├── mod.rs
│       └── vulkan_backend.rs
├── tests
│   ├── data
│   │   ├── data.csv
│   │   └── test_doc.txt
│   ├── test_keys.py
│   └── test_tps.exp
├── tools
│   ├── __pycache__
│   │   ├── hardware_profiler.cpython-314.pyc
│   │   └── rescue_model.cpython-314.pyc
│   ├── attention_audit.rs
│   ├── awake_aligner.rs
│   ├── boundary_validator.rs
│   ├── chat_once.rs
│   ├── check_scale.rs
│   ├── cognitive_integrity.rs
│   ├── conversion_verifier.rs
│   ├── create_blank_mud.rs
│   ├── debug_signal.py
│   ├── deep_math_audit.rs
│   ├── diagnose_chat.rs
│   ├── diagnose_layers.rs
│   ├── diagnose_model.rs
│   ├── diffusion_demo.rs
│   ├── download_corpus.py
│   ├── download_model.py
│   ├── dump_gguf_meta.py
│   ├── embed_audit.rs
│   ├── embed_ternarize.rs
│   ├── eval_harness.rs
│   ├── expert_anatomy.rs
│   ├── fix_metadata.rs
│   ├── galore_dora_benchmark.rs
│   ├── gguf_to_mud.rs
│   ├── gguf_to_safetensors.rs
│   ├── hw_detect.rs
│   ├── inference_bench.rs
│   ├── int4_quantizer.rs
│   ├── interactive_validator.rs
│   ├── iq_box.rs
│   ├── iteration_validator.rs
│   ├── jamba_benchmark.rs
│   ├── jepa_wave_benchmark.rs
│   ├── kernel_bench.rs
│   ├── language_audit.rs
│   ├── ldt_audit.rs
│   ├── legacy
│   │   ├── autopsy.rs
│   │   ├── autopsy_gguf.rs
│   │   ├── check_bf16.rs
│   │   ├── dead_code_audit.rs
│   │   ├── engine_diagnostics.rs
│   │   ├── fix_labels.py
│   │   ├── legacy_inference.rs
│   │   ├── legacy_transformer.rs
│   │   ├── read_meta.rs
│   │   ├── repro_crash.rs
│   │   ├── scratch_vocab.rs
│   │   ├── seed_knowledge.rs
│   │   ├── trace_bak.rs
│   │   ├── trace_bug6.rs
│   │   ├── training_health.rs
│   │   └── verify_parity.rs
│   ├── matrix_benchmark.rs
│   ├── memory_benchmark.rs
│   ├── merge_safetensors.py
│   ├── model_banner.rs
│   ├── model_dumper.rs
│   ├── moe_audit.rs
│   ├── mud_calibrator.rs
│   ├── mud_corpus_trainer.rs
│   ├── mud_diagnostics.rs
│   ├── mud_forge.rs
│   ├── mud_fusion.rs
│   ├── mud_offsets.rs
│   ├── mud_selector.rs
│   ├── phase14_audit.rs
│   ├── precision_benchmark.rs
│   ├── propagation_probe.rs
│   ├── ptr_audit.rs
│   ├── python_wave_probe.py
│   ├── qat_benchmark.rs
│   ├── recalibration_projector.rs
│   ├── refactor_forward.py
│   ├── skills
│   │   ├── quantization-stability-auditor.skill
│   │   ├── recursive-reasoning-architect.skill
│   │   ├── super-math-engineer.skill
│   │   ├── super-senior-programmer.skill
│   │   ├── ternary-simd-hardware-expert.skill
│   │   └── vulkan-gpu-kernel-engineer.skill
│   ├── step_inference.rs
│   ├── step_profiler.rs
│   ├── tensor_health.rs
│   ├── tensor_microscope.rs
│   ├── ternary_audit.rs
│   ├── test_embed.rs
│   ├── test_tokenizer.rs
│   ├── test_tokenizer_bpe.rs
│   ├── tokenizer_audit.rs
│   ├── training_estimator.rs
│   ├── universal_converter
│   │   ├── calibration.rs
│   │   ├── main.rs
│   │   ├── parser.rs
│   │   └── quantizer.rs
│   ├── vocab_check.rs
│   ├── vulkan_simulator.rs
│   ├── wave_alignment_audit.rs
│   ├── wave_superposition_demo.rs
│   └── weight_audit.rs
└── training
    ├── README.md
    ├── corpus
    │   └── unified_corpus.txt
    └── vocab_es_en.txt

```\n
