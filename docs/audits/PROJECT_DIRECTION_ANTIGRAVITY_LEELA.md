# 📋 INFORME: Dirección del Proyecto MUD — Antigravity, Leela y Roadmap

**Fecha:** 2026-07-24  
**Fuente:** `GEMINI.md`, `AGENTS.md`, `VISION_ROADMAP.md`, `docs/research/MUD_IMPROVEMENTS_POST_AE.md`, `docs/MUD_HISTORIAL_CONVERSACIONES_ANTIGRAVITY.md`

---

## 1. ¿Qué es "Antigravity"?

**"Antigravity"** es el nombre del **agente de IA** (Google DeepMind Advanced Agentic Coding) que ha estado trabajando en el proyecto Forge LLM (MUD). No es un concepto técnico dentro del proyecto.

- **Documento maestro:** `docs/MUD_HISTORIAL_CONVERSACIONES_ANTIGRAVITY.md` (398 líneas, 91 sesiones registradas)
- **Períodos de desarrollo:**
  - **Phase 0** (Jun 03–10): Auditoría inicial, kernels AVX2, multi-agent setup
  - **Phase 1** (Jun 11–17): Bare-metal tuning, Vulkan Ash 0.38, BitNet conversion
  - **Phase 2** (Jun 18–30): Housekeeping P-08, SlimeRegister f32 refactor
  - **Phase 3** (Jul 01–20): C-MUD, SlimeX CellV5, Memory Continuity Table, Telemetry TUI
  - **Phase 4** (Jul 21–24): Seed-driven training circuit, Q2_0 dequant fix, trainer split, Sprints 4-7

---

## 2. ¿Qué es "Leela"?

**"Leela" no aparece en ningún archivo del proyecto.** No hay menciones de Leela Chess, Leela Zero, o cualquier concepto relacionado. La búsqueda en todo el codebase (`.rs`, `.md`, `.toml`, `.json`, `.sh`) devuelve cero resultados para "leela".

---

## 3. Estado actual del proyecto (2026-07-24)

### ✅ Completado (Ledger A–E + Depth A–E + F1/F2/TLM + Sprints 4-7)

| Área | Estado |
|------|--------|
| Forward + backward STE QAT | ✅ ELUT 4-bit + PRQ |
| SlimeRegister f32 matmul_accum | ✅ (P-02) |
| GEMV AVX2×8 + auto GPU | ✅ (gemv_policy) |
| Optimizers LIVE | ✅ Muon / GaLore / Chunked / Adam |
| JEPA OU + mHC | ✅ |
| Full-seq train + L-10 packing | ✅ |
| MoE ExpertBus | ✅ |
| CSA top-k + HCA 32k | ✅ |
| C-MUD complex manifold | ✅ (L-14) |
| Fenwick trees | ✅ (802 LOC, 25 tests, 19.5× speedup) |
| Telemetry TUI | ✅ |
| Hot loops pointer/LUT | ✅ (P-00/P-01) |
| Security docs (Sprints 6-7) | ✅ |
| **Tests** | **344 passing, 0 failing** |
| **Clippy** | **0 errors, 0 warnings** |

### 🔧 Trabajo reciente (2026-07-24)

| Trabajo | Detalle |
|---------|---------|
| **Sprint 5** | 5.1 zero-alloc `evaluate_slime_block_moe`; 5.2 `Mutex`→`AtomicU8`; 5.3 LUT backward; 5.4 branchless pack |
| **Fenwick trees** | `FenwickF32`/`FenwickU32`/`FenwickI32`/`FenwickLogF32` + AVX2/BMI1/BMI2 batch ops |
| **Race condition fixes** | 2 tests fallidos → 344/344 passing |
| **Dead code cleanup** | 5 `#[allow(dead_code)]` → 1 (ash_backend) |
| **Chat refactor** | Debug condicional, conversation history, system prompt, temperature, Fenwick sampling, colores |

---

## 4. Dirección futura (F+ backlog + Phase D)

### 4.1 Streams cerrados (F/K/G/H/I/J/L)

Según `docs/research/MUD_IMPROVEMENTS_POST_AE.md`, todos los streams de mejora post-A–E están **COMPLETADOS**:

| Stream | Tema | Estado |
|--------|------|--------|
| **F** | QKV multi-matrix one CB | ✅ 2026-07-23 |
| **K** | Loss certification CI | ✅ 2026-07-23 |
| **G** | Multi-expert STE joint | ✅ 2026-07-23 (SlimeXSlot + ShadowExpertBus) |
| **H** | Long full-seq / grad ckpt | ✅ 2026-07-23 |
| **I** | KV bf16/quant | ✅ 2026-07-23 (KvDtype::BF16) |
| **J** | CSA v2 LSH | ✅ 2026-07-23 (SimHash prefilter) |
| **L** | Converter P-13 | ✅ 2026-07-23 |

### 4.2 Próximas decisiones (AGENTS.md §"Próximas decisiones")

> **F4 Fenwick trees** (`docs/research/MUD_FENWICK_TREE_OPTIMIZATION_PLAN.md`, 575 LOC + 18 tests) **o** Sprints 6-7 (docs/seguridad). Ver `docs/sessions/MUD_SESSION_REPORT_2026-07-24_SPRINT_5.md`.

**Estado:** Fenwick trees ya implementados (`src/mud/fenwick.rs`, 802 LOC). Sprints 6-7 (security docs) ya completados.

### 4.3 Backlog de investigación restante

| Item | Estado |
|------|--------|
| Sparse MoE multi-expert backward | ✅ COMPLETED (2026-07-23) |
| Learned W_compress CSA | ✅ COMPLETED (2026-07-23) |
| Native Google BF16 | ✅ COMPLETED (2026-07-23) |
| C-MUD × log-gas (JHEP CFT) | ⏸️ DEFERRED (feasibility study only) |

### 4.4 Phase D — Escala y madureza (resto Q4 2026)

| Objetivo | Hoy | Objetivo dic 2026 |
|----------|-----|-------------------|
| Training | ~ord. hours/epoch | **<2h/epoch** |
| Inference | ~5 tok/s (2B clase) | **~20 tok/s** |
| Optimizer | Muon/GaLore/Adam | + multi-expert STE |
| Overlap CPU/GPU | L-05 + GEMV auto | QKV one-CB (F) |
| Context | 32k HCA + CSA | bf16 KV / longer |
| Tests | 344 | 200+ + loss cert |

### 4.5 RPG Battle Circuit (futuro)

> **RPG Battle Circuit (Fase futura):** Evolución por supervivencia, "barras de vida", debate de clones y recompensas de 5-epoch para el jugador ganador.

---

## 5. Variables de entorno clave (product knobs)

| Área | Variables |
|------|-----------|
| Optimizers | `MUD_USE_VULKAN` (Muon NS GPU) |
| MoE | `MUD_TRAIN_EXPERT`, `MUD_MOE_CLONE`, `MUD_MOE_TOP_K`, `MUD_MOE_TRAIN=1\|hash` |
| GEMV | `MUD_GPU_GEMV=auto\|0\|1`, `MUD_GPU_GEMV_MIN` |
| Full-seq | `MUD_TRAIN_FULL_SEQ`, `MUD_TRAIN_SEQ_LEN` (try 128) |
| Grad ckpt | `MUD_GRAD_CKPT`, `MUD_GRAD_CKPT_SEG`, `MUD_GRAD_CKPT_RESIDUAL=1` |
| CSA | `MUD_CSA`, `MUD_CSA_TOP_K`, `MUD_CSA_LSH=1`, `MUD_CSA_LSH_BITS` |
| KV | `MUD_KV_DTYPE=f32\|f16\|bf16` |
| Loss cert | `MUD_LOSS_CERT_FAST`, `MUD_CI_LOSS_CERT=1` |
| Context | `MUD_MAX_POS`, `MUD_HCA_WINDOW`, `MUD_HCA_RATIO` |
| Chat | `MUD_DEBUG`, `MUD_SYSTEM_PROMPT`, `MUD_INFER_TEMPERATURE`, `MUD_INFER_TOP_P`, `MUD_CMUD_THINK` |

---

## 6. Conclusión

El proyecto Forge LLM (MUD) está en una fase de **madureza y escala** (Phase D). Todos los bloques fundacionales (A–E, depth A–E, F1/F2/TLM, Sprints 4-7) están completados y verificados (344 tests, 0 clippy warnings).

La dirección futura se centra en:
1. **Performance:** <2h/epoch training, ~20 tok/s inference
2. **Escalabilidad:** bf16 KV, longer context, multi-expert STE
3. **Producto:** RPG Battle Circuit (evolución por supervivencia)

**"Antigravity"** es el agente de IA que ha desarrollado el proyecto. **"Leela"** no existe en el proyecto.
