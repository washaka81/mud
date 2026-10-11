# MUD Security — Threat Model (Sprint 6)

> **Sprint:** 6 (Security Documentation)
> **Fecha:** 2026-07-24
> **Estado:** ✅ Completado
> **Base:** `SECURITY.md` (Sprint 1) — ampliado con análisis detallado

---

## 1. Alcance

Forge LLM (MUD) es un motor local, offline de inferencia y entrenamiento de LLMs.
**No realiza llamadas de red durante la inferencia o entrenamiento por defecto.**

Este documento define el modelo de amenazas, superficie de ataque, y mitigaciones
para todos los componentes del sistema.

---

## 2. Actores de amenaza

| Actor | Motivación | Capacidad |
|---|---|---|
| **Usuario local** | Curiosidad, debugging | Acceso completo al proceso |
| **Modelo malicioso** | `.mud` crafted para explotar parser | Controla datos de entrada |
| **Driver Vulkan** | Bug en driver de GPU | Acceso a memoria GPU/vRAM |
| **Feature flags** | Desarrollador activa `unsafe-rce` | RCE vía `rustc` / `bash` |
| **Corpus de entrenamiento** | Texto malicioso en datos | Prompt injection, jailbreaking |

---

## 3. Superficie de ataque

### 3.1 `.mud` file loader

**Riesgo:** OOB reads, integer overflow, corrupt memory

**Componentes:**
- `MudFile::load` — parser binario de formato `.mud`
- `MudTensor::data_base` — offset relativo a región de datos
- `MudFile::save` — escritura de tensores

**Mitigaciones:**
- ✅ Validación de tensor counts, shapes, y offsets en `MudFile::load`
- ✅ `data_base + offset` corregido (FIX D, 2026-07-18)
- ✅ `diagnose_model` para health checks
- ✅ `MudFileMerger` valida magic bytes antes de merge

**Tests de seguridad:**
- `test_mud_file_load_rejects_invalid_magic`
- `test_tensor_offset_out_of_bounds`
- `test_data_base_offset_not_absolute`

### 3.2 ASM kernels (P-00)

**Riesgo:** Raw pointer misuse → OOB reads/writes, SIGSEGV

**Componentes:**
- `src/asm/*.s` — Assembly AVX2 kernels
- `src/asm/mod.rs` — Wrappers Rust con `unsafe`
- `src/mud/ezop.rs` — Raw pointer hot loops

**Mitigaciones:**
- ✅ Cada `unsafe` tiene `// SAFETY:` comment
- ✅ P-00 raw pointer mastery enforced en hot path
- ✅ `black_box` en benchmarks evita optimización agresiva
- ✅ Tests de correctness comparan ASM vs scalar

**Tests de seguridad:**
- `test_asm_unpack_cells_v5_matches_scalar`
- `test_ternary_gemv_correctness`
- `test_adam_step_correctness`

### 3.3 Vulkan backend

**Riesgo:** Driver exploit, SIGSEGV, vRAM corruption

**Componentes:**
- `src/vulkan/ash_backend.rs` — ash 0.38 bindings
- `probe_gpu()` — GPU detection y blacklist
- `ash_qat_dispatcher.rs` — QAT dispatch

**Mitigaciones:**
- ✅ `probe_gpu()` blacklistea Iris Xe in-binary
- ✅ `MUD_USE_VULKAN=0` fuerza CPU
- ✅ `MUD_GPU_GEMV=0` deshabilita GPU GEMV
- ✅ Iris Xe SIGSEGV documentado (2026-07-18)

**Tests de seguridad:**
- `test_should_try_off_via_effective`
- `test_stream_f_qkv_one_cb`

### 3.4 Feature flags

**Riesgo:** RCE via `unsafe-rce` / `unsafe-cli`

**Componentes:**
- `Cargo.toml` — feature flags
- `RlvrCritic::evaluate_rust_code` — compila & ejecuta Rust
- `TerminalSandbox::execute_command` — ejecuta `bash -c`

**Mitigaciones:**
- ✅ **Disabled by default** (`default = []`)
- ✅ Documentado en `SECURITY.md`
- ✅ Separación clara: production vs research builds

### 3.5 Path traversal (AgentWorkspace)

**Riesgo:** Acceso a archivos fuera del workspace

**Componentes:**
- `src/mud/workspace_agent.rs` — `AgentWorkspace`
- `src/mud/skills/` — Skills que acceden a filesystem

**Mitigaciones:**
- ✅ `fs::canonicalize()` resuelve symlinks y `..`
- ✅ `starts_with(&root_canon)` asegura paths dentro del root
- ✅ No hay `unsafe` en path handling

**Tests de seguridad:**
- `test_absolute_path_rejected`
- `test_path_traversal_blocked`

### 3.6 ASM AVX2 unpacker (SlimeX CellV5)

**Riesgo:** Buffer overflow en descompresión de celdas

**Componentes:**
- `src/asm/slimex_cell_v5_avx2.s` — Assembly unpacker
- `src/mud/slime_x.rs` — ShadowExpertBus
- `src/mud/slime.rs` — `slimex_cells` buffer

**Mitigaciones:**
- ✅ `AlignedArray<u32>` 64-byte aligned
- ✅ Zero-size cuando `MUD_SLIMEX ≠ 1`
- ✅ Raw-ptr hot path con bounds checking en setup
- ✅ Tests roundtrip + equi_byte_partition

---

## 4. Análisis de riesgos (DREAD)

| Componente | Damage | Reproducibilidad | Exploitabilidad | Afectados | Descubrimiento | Score |
|---|---|---|---|---|---|---|
| `.mud` loader | 8 | 10 | 6 | 10 | 8 | **8.2** |
| ASM kernels | 10 | 8 | 5 | 8 | 7 | **7.6** |
| Vulkan backend | 7 | 6 | 4 | 6 | 5 | **5.8** |
| Feature flags | 10 | 10 | 8 | 10 | 9 | **9.4** |
| Path traversal | 6 | 7 | 5 | 8 | 6 | **6.4** |
| SlimeX unpacker | 8 | 7 | 4 | 6 | 5 | **6.0** |

**Prioridad de mitigación:** Feature flags > `.mud` loader > ASM kernels > Path traversal > SlimeX > Vulkan

---

## 5. Hardening recommendations

Ver `docs/seguridad/HARDENING.md` para detalles de implementación.

---

## 6. Referencias

- `SECURITY.md` — Security policy (Sprint 1)
- `docs/sessions/MUD_SESSION_REPORT_2026-07-18.md` — Iris Xe SIGSEGV
- `docs/sessions/MUD_SESSION_REPORT_2026-07-20.md` — QAT fixes
- `docs/sessions/MUD_SESSION_REPORT_2026-07-24_SPRINT_5.md` — Sprint 5 security
