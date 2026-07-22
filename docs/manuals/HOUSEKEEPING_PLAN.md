# HOUSEKEEPING PLAN — Forge LLM
## Eliminar Python, limpiar raíz, actualizar TREE.md

---

## FASE 0: RESPALDO (ANTES DE EMPEZAR)

```bash
git add -A && git stash
# Si algo sale mal: git stash pop
```

---

## PASO 1: ELIMINAR ARCHIVOS PYTHON (P-07)

### 1.1 Scripts Python en raíz (6 archivos)

| Archivo | Dependencias | Propósito | Acción |
|---------|-------------|-----------|--------|
| `fix_corpus.py` | `re` | Modifica `corpus_trainer.rs` vía regex | **Eliminar** |
| `fix_corpus2.py` | `re` | Modifica `corpus_trainer.rs` vía regex | **Eliminar** |
| `fix_corpus3.py` | `re` | Modifica `corpus_trainer.rs` vía regex | **Eliminar** |
| `restore_corpus.py` | `json` | Lee `.jsonl` transcript, modifica Rust | **Eliminar** |
| `update_telemetry.py` | `os` | Modifica `train_telemetry.rs` vía regex | **Eliminar** |
| `tools/patch_global_stream.py` | `re` | Modifica `corpus_trainer.rs` vía regex | **Eliminar** |

```bash
rm fix_corpus.py fix_corpus2.py fix_corpus3.py restore_corpus.py update_telemetry.py
rm tools/patch_global_stream.py
```

### 1.2 Entorno virtual Python

```bash
rm -rf .venv/
```

### 1.3 Playground legacy (contiene referencia Python)

`playground/calculus_llm/slime.sh:38` invoca `python3` a un archivo que no existe.

```bash
# Opción A: eliminar todo playground (C++ legacy, no Rust)
rm -rf playground/

# Opción B: solo eliminar la referencia (mantener)
rm playground/calculus_llm/slime.sh
```

---

## PASO 2: ELIMINAR SCRATCH FILES Y BINARIOS

### 2.1 Scratch Rust source (6 archivos)

| Archivo | Líneas | Acción |
|---------|--------|--------|
| `scratch.rs` | 81 | **Eliminar** |
| `scratch2.rs` | 751 | **Eliminar** |
| `scratch3.rs` | 291 | **Eliminar** |
| `scratch4.rs` | 619 | **Eliminar** |
| `scratch_telemetry.rs` | 13,630 | **Eliminar** |
| `test_affinity.rs` | 508 | **Eliminar** |

```bash
rm scratch.rs scratch2.rs scratch3.rs scratch4.rs scratch_telemetry.rs test_affinity.rs
```

### 2.2 Scratch binaries (4 ELF, 3.9MB cada uno)

```bash
rm scratch scratch2 scratch3 scratch4
```

---

## PASO 3: ELIMINAR ARTEFACTOS DE MERGE

```bash
rm src/main.rs.orig src/main.rs.rej
rm src/model/tokenizer.rs.orig src/model/tokenizer.rs.rej
```

---

## PASO 4: ELIMINAR BUILD ARTEFACTOS SUELTOS

```bash
rm ternary_gemv.o libasm_test.a
```

---

## PASO 5: LIMPIAR LOGS Y DEBUG FILES

### 5.1 Logs en raíz (no los de logs/)

| Archivo | Acción |
|---------|--------|
| `crash.log` | **Eliminar** |
| `iter_val.log` | **Eliminar** |
| `output.log` | **Eliminar** |
| `test_output.log` | **Eliminar** |
| `validator.log` | **Eliminar** |
| `mud_metrics.log` | **Eliminar** (regenerado cada run) |
| `mud_train_metrics.log` | **Eliminar** (regenerado cada run) |

```bash
rm crash.log iter_val.log output.log test_output.log validator.log
rm mud_metrics.log mud_train_metrics.log
```

### 5.2 Debug dumps

| Archivo | Acción |
|---------|--------|
| `dump.txt` | **Eliminar** |
| `bt.txt` | **Eliminar** |
| `out.txt` | **Eliminar** |
| `mud_disassembly.txt` | **Eliminar** |
| `tui_raw.log` | **Eliminar** |
| `tui_clean.txt` | **Eliminar** |

```bash
rm dump.txt bt.txt out.txt mud_disassembly.txt tui_raw.log tui_clean.txt
```

---

## PASO 6: LIMPIAR ARCHIVOS DE CONFIG/HUÉRFANOS

```bash
rm .mud_env
```

---

## PASO 7: ACTUALIZAR `.gitignore`

Añadir entradas para los patrones que deben ignorarse:

```
# --- SCRATCH FILES ---
scratch*.rs
scratch*
test_affinity.rs

# --- MERGE ARTIFACTS ---
*.orig
*.rej

# --- BUILD ARTIFACTS ---
*.o
*.a

# --- RUNTIME LOGS (root) ---
/mud_metrics.log
/mud_train_metrics.log
/crash.log
/iter_val.log
/output.log
/test_output.log
/validator.log

# --- DEBUG DUMPS ---
/dump.txt
/bt.txt
/out.txt
/mud_disassembly.txt
/tui_*.log
/tui_*.txt

# --- PYTHON ---
*.py
.venv/
```

---

## PASO 8: ACTUALIZAR DOCS QUE REFERENCIAN PYTHON

### 8.1 `docs/manuals/MUD_DIRECTORY_STRUCTURE.md`

Eliminar o marcar como obsoletas las líneas que listan:
- `test_keys.py` (line 226)
- `hardware_profiler.cpython-314.pyc` (line 230-231)
- `rescue_model.cpython-314.pyc`
- `python_wave_probe.py` (line 302)

### 8.2 `docs/audits/MUD_AUDIT_REPORT_V01.md`

Eliminar referencias a:
- `python3 training/v37_master_trainer.py` (line 55)
- `python3 tools/cognitive_dashboard.py` (line 56)

### 8.3 `docs/audits/MUD_AUDIT_REPORT_V10.md`

Eliminar referencia a `fix_labels.py` (line 11).

---

## PASO 9: ACTUALIZAR `AGENTS.md`

### 9.1 Eliminar referencias Python

Buscar:
- Línea `Last Research Session: 2026-07-01` — referencias a Python en la descripción
- P-07 dice "No Python in production" — está correcto
- Eliminar mención a `generate_asm_batch4.py` si existe

### 9.2 Actualizar conteo de tests

`cargo test` (era 89, puede haber cambiado)

```bash
cargo test 2>&1 | tail -5
```

---

## PASO 10: VERIFICAR CARGO.TOML

### 10.1 Eliminar dependencias Python-related

No hay dependencias Python en Cargo.toml (solo Rust crates). Verificar que no haya `pyo3` o similar:

```bash
grep -i "python\|pyo3\|pypi" Cargo.toml
# Esperado: sin resultado
```

---

## PASO 11: VERIFICACIÓN POST-CLEANUP

```bash
# Estado del workspace
git status

# Build check
cargo check --release 2>&1

# Clippy (P-06)
cargo clippy --all-targets 2>&1

# Tests
cargo test 2>&1 | tail -10
```

---

## PASO 12: ACTUALIZAR `TREE.md`

Reemplazar con la estructura real post-cleanup:

```markdown
# MUD Project Tree

```
├── AGENTS.md                  # AI agent project context
├── Cargo.toml                 # Rust workspace root
├── build.rs                   # ASM compilation
├── mud.sh                     # Orchestrator CLI
├── LICENSE
├── README.md
├── PLAN_MAESTRO.md            # Master plan
├── VISION_ROADMAP.md          # Architecture vision
├── ASM_CORRECTION_PLAN.md     # ASM optimization plan
├── VULKAN_AVX2_THREADS_OPTIMIZATION.md  # GPU/CPU pipeline
├── STATUS_REPORT.md           # Progress vs debt
│
├── src/                       # ── Core Engine ──
│   ├── main.rs                # TUI inference
│   ├── lib.rs                 # Library root
│   ├── hardware.rs            # CPU feature detection
│   │
│   ├── asm/                   # AVX2 assembly kernels (14 files)
│   │   ├── ternary_gemv.s     # Main FP32 GEMV (8 accum)
│   │   ├── ternary_gemv_4rows.s # 4-row batched GEMV
│   │   ├── ternary_gemm_batch4.s # Batch-4 GEMM
│   │   ├── ternary_backward.s # GCC backward pass
│   │   ├── adam_step.s        # Adam optimizer
│   │   ├── silu.s             # SiLU activation
│   │   ├── rmsnorm.s          # RMS norm (simple)
│   │   ├── slime_rmsnorm.s    # Slime register RMSNorm
│   │   ├── math.s             # Dot product, sum squares, etc.
│   │   ├── sgemm.s            # SGEMM
│   │   ├── mamba.s            # Mamba SSM scan
│   │   ├── rope.s             # RoPE positional encoding
│   │   ├── q4_0_gemv.s        # Q4_0 quantized GEMV
│   │   ├── lm_head.s          # LM head argmax
│   │   ├── mod.rs             # FFI declarations
│   │   └── tests.rs           # ASM test suite
│   │
│   ├── model/                 # Tokenizer + model loading
│   │   ├── tokenizer.rs       # BPE tokenizer
│   │   └── tokenizer_test.rs
│   │
│   ├── gguf/                  # GGUF parser
│   │
│   ├── mud/                   # ── MUD Engine ──
│   │   ├── mod.rs             # MudFile, MudTensor (733 loc)
│   │   ├── constants.rs       # Shared constants
│   │   ├── slime.rs           # SlimeRegister, init_from_embed
│   │   ├── workspace.rs       # AlignedBuffer, SlimeWorkspace
│   │   ├── slime_forward.rs   # evaluate_slime_block (865 loc)
│   │   ├── slime_backward.rs  # Backward pass STE (739 loc)
│   │   ├── slime_jepa.rs      # JEPA stabilizer (365 loc)
│   │   ├── corpus_trainer.rs  # QAT training loop (2949 loc)
│   │   ├── pcore_pool.rs      # 8-thread pool (121 loc)
│   │   ├── speculative.rs     # DSpark drafter
│   │   ├── self_play.rs       # Synthetic self-play
│   │   ├── muon.rs            # Muon optimizer
│   │   ├── galore.rs          # GaLore optimizer
│   │   ├── ash_qat_dispatcher.rs  # Vulkan QAT dispatch
│   │   ├── ecc.rs             # Error-correcting codes
│   │   ├── ldt_micro.rs       # Lattice dynamics
│   │   ├── routing.rs         # MoE routing (future)
│   │   ├── holographic_loss.rs # Holographic contrastive loss
│   │   ├── memory_bank.rs     # Memory bank (experimental)
│   │   ├── subagents.rs       # Subagent system (experimental)
│   │   ├── arena_games.rs     # Arena training (experimental)
│   │   ├── sandbox.rs         # Code sandbox (experimental)
│   │   ├── dspy.rs            # DSPy integration (experimental)
│   │   ├── rlvr.rs            # RLVR metrics
│   │   ├── debate_trainer.rs  # Debate training
│   │   └── tests.rs           # Integration tests (834 loc)
│   │
│   └── vulkan/                # Vulkan backend
│       ├── mod.rs
│       └── ash_backend.rs     # Ash compute backend
│
├── forge_autograd/            # Standalone autograd crate
│   └── src/
│       ├── lib.rs             # Tape, Node, backward (2467 loc)
│       └── avx_math.rs        # AVX2 math intrinsics
│
├── tools/                     # ── CLI Utilities ──
│   ├── warp_aligner.rs        # TUI trainer
│   ├── run_trainer.rs         # Training launcher
│   ├── step_inference.rs      # Headless inference
│   ├── universal_converter/   # safetensors → .mud
│   ├── diagnose_model.rs      # Model diagnostic
│   ├── converter_auditor.rs   # Post-conversion audit
│   ├── variance_inspector.rs  # Variance telemetry
│   ├── doppler_radar.rs       # Tensor inspection
│   ├── list_tensors.rs        # Tensor listing
│   ├── training_healthcheck.rs # Optimizer selection
│   ├── mud_calibrator.rs      # Calibration
│   ├── quadrature_bench.rs    # QAT benchmark
│   ├── boundary_validator.rs  # Ternary boundary audit
│   ├── ezop_bench.rs          # EZOP raw pointer bench
│   ├── zerocopy_bench.rs      # Zero-copy Vulkan bench
│   └── train_telemetry.rs     # TUI telemetry graph
│
├── assets/shaders/            # Vulkan compute shaders
│   ├── ternary_gemv_unified.comp
│   ├── shadow_optimizer.comp
│   ├── ternary_backward.comp
│   ├── ternary_backward_opt.comp
│   ├── mha.comp               # Multi-head attention
│   ├── rms_norm.comp          # RMS normalization
│   ├── silu_gate.comp         # SiLU activation
│   ├── dspark_drafter.comp    # Speculative drafter
│   ├── ghost_align.comp       # Ghost alignment
│   ├── tensor_thermodynamics.comp  # Thermodynamic monitoring
│   └── heartbeat.comp         # GPU keep-alive
│
├── tests/                     # Integration tests
│
├── training/                  # Training corpus
│   └── corpus/
│       └── unified_corpus.txt
│
├── models/                    # Model weights (.mud)
│
├── weights/checkpoints/       # Saved checkpoints
│
├── docs/                      # Documentation
│   ├── README.md
│   ├── architecture/
│   ├── audits/
│   ├── research/
│   ├── sessions/
│   ├── manuals/
│   └── dumps/
│
├── .cargo/config.toml         # RUSTFLAGS
└── .gitignore
```
```

---

## TABLA RESUMEN: ARCHIVOS A ELIMINAR

| # | Archivo | Tipo | Paso |
|---|---------|------|------|
| 1 | `fix_corpus.py` | Python | 1.1 |
| 2 | `fix_corpus2.py` | Python | 1.1 |
| 3 | `fix_corpus3.py` | Python | 1.1 |
| 4 | `restore_corpus.py` | Python | 1.1 |
| 5 | `update_telemetry.py` | Python | 1.1 |
| 6 | `tools/patch_global_stream.py` | Python | 1.1 |
| 7 | `.venv/` | Python env | 1.2 |
| 8 | `playground/` (o `slime.sh`) | Legacy C++ | 1.3 |
| 9 | `scratch.rs` | Scratch Rust | 2.1 |
| 10 | `scratch2.rs` | Scratch Rust | 2.1 |
| 11 | `scratch3.rs` | Scratch Rust | 2.1 |
| 12 | `scratch4.rs` | Scratch Rust | 2.1 |
| 13 | `scratch_telemetry.rs` | Scratch Rust | 2.1 |
| 14 | `test_affinity.rs` | Scratch Rust | 2.1 |
| 15 | `scratch` | Binary ELF | 2.2 |
| 16 | `scratch2` | Binary ELF | 2.2 |
| 17 | `scratch3` | Binary ELF | 2.2 |
| 18 | `scratch4` | Binary ELF | 2.2 |
| 19 | `src/main.rs.orig` | Merge artifact | 3 |
| 20 | `src/main.rs.rej` | Merge artifact | 3 |
| 21 | `src/model/tokenizer.rs.orig` | Merge artifact | 3 |
| 22 | `src/model/tokenizer.rs.rej` | Merge artifact | 3 |
| 23 | `ternary_gemv.o` | Build artifact | 4 |
| 24 | `libasm_test.a` | Build artifact | 4 |
| 25 | `crash.log` | Log | 5.1 |
| 26 | `iter_val.log` | Log | 5.1 |
| 27 | `output.log` | Log | 5.1 |
| 28 | `test_output.log` | Log | 5.1 |
| 29 | `validator.log` | Log | 5.1 |
| 30 | `mud_metrics.log` | Log | 5.1 |
| 31 | `mud_train_metrics.log` | Log | 5.1 |
| 32 | `dump.txt` | Debug dump | 5.2 |
| 33 | `bt.txt` | Debug dump | 5.2 |
| 34 | `out.txt` | Debug dump | 5.2 |
| 35 | `mud_disassembly.txt` | Debug dump | 5.2 |
| 36 | `tui_raw.log` | Debug dump | 5.2 |
| 37 | `tui_clean.txt` | Debug dump | 5.2 |
| 38 | `.mud_env` | Config | 6 |

**Total: 38 archivos/directorios eliminados.**

---

## VERIFICACIÓN POST-CLEANUP

```bash
# 1. No debe haber .py en el repo
find . -name '*.py' 2>/dev/null | grep -v '.git/' | grep -v 'target/'

# 2. No debe haber scratch files
find . -name 'scratch*' 2>/dev/null | grep -v '.git/'

# 3. No debe haber merge artifacts
find . -name '*.orig' -o -name '*.rej' 2>/dev/null | grep -v '.git/'

# 4. No debe haber .o/.a sueltos
find . -name '*.o' -o -name '*.a' 2>/dev/null | grep -v '.git/' | grep -v 'target/'

# 5. Build funciona
cargo check --release 2>&1

# 6. Clippy (P-06)
cargo clippy --all-targets 2>&1

# 7. Tests
cargo test 2>&1 | tail -5
```

---

## RAÍZ ESPERADA POST-CLEANUP

```
/forge_llm/
├── AGENTS.md
├── ASM_CORRECTION_PLAN.md
├── ASM_OPTIMIZATION_PLAN.md
├── AUDIT_REPORT.md
├── ASM_AUDIT_REPORT.md
├── Cargo.toml
├── LICENSE
├── PLAN_MAESTRO.md
├── README.md
├── STATUS_REPORT.md
├── TREE.md
├── VISION_ROADMAP.md
├── VULKAN_AVX2_THREADS_OPTIMIZATION.md
├── build.rs
├── mud.sh
├── .cargo/
├── .git/
├── .gitignore
├── assets/
├── docs/
├── forge_autograd/
├── models/
├── src/
├── tests/
├── tools/
├── training/
└── weights/
```

Solo 7 archivos en raíz + 7 reportes/documentos + 8 directorios. Nada más.
