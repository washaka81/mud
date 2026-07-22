# FORGE LLM — VISIÓN DE ARQUITECTURA Y HOJA DE RUTA

> **Jerarquía de documentos (2026-07-16)**  
> - Políticas, estado técnico y ledger **L-##**: [`GEMINI.md`](./GEMINI.md) (**SSOT**)  
> - Contexto para agentes: [`AGENTS.md`](./AGENTS.md)  
> - Stack de cómputo: [`docs/architecture/MUD_COMPUTE_STACK.md`](./docs/architecture/MUD_COMPUTE_STACK.md)  
> - Narrativa profunda + Mini MoE: [`PLAN_MAESTRO.md`](./PLAN_MAESTRO.md)  
> - Gap residual + env knobs: [`docs/research/MUD_GAP_ANALYSIS_POST_L15.md`](./docs/research/MUD_GAP_ANALYSIS_POST_L15.md)  
> - **Mejoras F+ (siguiente):** [`docs/research/MUD_IMPROVEMENTS_POST_AE.md`](./docs/research/MUD_IMPROVEMENTS_POST_AE.md)  
> - Session A–E: [`docs/sessions/MUD_SESSION_REPORT_2026-07-16_STREAMS_AE.md`](./docs/sessions/MUD_SESSION_REPORT_2026-07-16_STREAMS_AE.md)  
> - **Despegue:** [`docs/manuals/LAUNCH_COUNTDOWN.md`](./docs/manuals/LAUNCH_COUNTDOWN.md)  
> - Este archivo: **visión de producto + fases Q3–Q4 2026**

---

## 1. IDENTIDAD DEL PROYECTO

Forge LLM (MUD — Modular Understanding Dynamics) es un motor de inferencia y entrenamiento para LLMs ternarios (1.58-bit) orientado a **CPU Intel de consumo**, con aceleración opcional en iGPU (Iris Xe vía Vulkan). Target de diseño de referencia: **Intel i7-1260P** (P-cores + HT, Iris Xe).

**No es** un wrapper de llama.cpp ni un port de PyTorch. Es Rust + ASM AVX2 + Vulkan Compute, con control del pipeline desde el kernel hasta el pool de hilos.

### Filosofía fundacional

- **Sin dependencias pesadas** — runtime en un binario compacto  
- **Sin Python en producción** — P-07  
- **Sin NVIDIA obligatorio** — hardware Intel estándar  
- **Control total** — ELUT, JEPA/mHC, optimizers, dispatch CPU/GPU  

### Norte de producto

Un LLM ~2B (~400MB en 1.58-bit) **entrenable y usable** en un portátil ~800€, offline y privado.

**No es el objetivo** ganar MMLU frente a frontier APIs.  
**Sí es el objetivo** capacidad media útil × coste casi cero × accesibilidad total.

---

## 2. ESTADO ACTUAL (2026-07-16, post A–E) — alineado con GEMINI §0

### Lo que funciona (LIVE)

| Área | Realidad |
|------|----------|
| Forward + backward STE QAT | ELUT 4-bit + PRQ |
| Accum / registers | **FP32** `matmul_accum` |
| GEMV | AVX2×8 + **auto GPU** (`gemv_policy`) |
| SiLU / attn dots / LM logits | ASM cableado |
| Optimizers | Muon / GaLore / Chunked / **Adam moments** |
| Estabilidad | JEPA OU + mHC |
| Train | Sampled Softmax + **full-seq windows** (L-10 + stream D) |
| MoE | ExpertBus + **`.mud` load / train-expert** (B) |
| Context | HCA 32k + **CSA top-k** (E) |
| ash Vulkan | NS, GEMV, RMSNorm/MHA helpers, QAT double-buffer |
| Tests | ~186 lib; `./mud.sh ci|audit-full` |
| Telemetry TUI | **DONE** — key-parse `[TELEM]`/`[DW]` + Weight Δ panel (2026-07-20) |
| Hot loops | **DONE** — pointer/LUT (P-00/P-01) in pack/dequant/clamp |

### Ledger y streams cerrados

| Bloque | Estado |
|--------|--------|
| **Fase A** L-01…L-08 | **DONE** |
| **Fase B/B+** GEMV tile + GPU path | **DONE** |
| **Fase C** L-09…L-11 | **DONE** |
| **Fase E-early** L-12…L-15 | **DONE** (C-MUD research + ckpt) |
| **Depth A–E** Adam→CSA | **DONE** 2026-07-16 |
| **F1** Trainable mHC α/β | **DONE** 2026-07-17 (`mhc_scale_sgd_step`, CPU+ash) |
| **F2** STP trajectory loss | **DONE** 2026-07-17 (`stp_loss`, `MUD_TRAIN_STP`) |
| **UI** Unified trainer console | **DONE** 2026-07-17 (`trainer_ui.rs` + `mud.sh train`) |

Stack detail: [`docs/architecture/MUD_COMPUTE_STACK.md`](./docs/architecture/MUD_COMPUTE_STACK.md).

---

## 3. VISIÓN ARQUITECTÓNICA (resumen)

```
Inference / Training (STE) / Drafter
        ↓
  SlimeWorkspace (P-00/P-01)
        ↓
  AVX2×8  |  ash (auto GEMV / NS / QAT)  |  PCorePool
        ↓
  Weights: ELUT 4-bit + PRQ
  Optimizers LIVE: Muon · GaLore · Chunked · Adam/Sparse
  Context: dense ring + HCA + CSA top-k
  Estabilidad: JEPA + mHC
```

Principios: P-00…P-27 — índice completo en `GEMINI.md` §9.

---

## 4. HOJA DE RUTA Q3–Q4 2026

### Fase A–E ledger — **cerrada**

Criterio de salida cumplido: strategies live, sin dead workspace, banners = verdad, packing, MoE bus, CI, HCA 32k, grad ckpt, C-MUD kernel.

### Depth streams A–E — **cerrada** (2026-07-16)

Adam · MoE load · GEMV auto · full-seq · CSA indexer.

### Siguiente (F+ research backlog)

> **F1/F2/UI CLOSED (2026-07-17):** mHC α/β trainable + STP aux loss + unified
> trainer console + project-adapted `mud.sh train`. Ver
> `docs/architecture/MUD_TRAINER_TERNARY_JEPA_MHC.md` §9–§10 y
> `docs/research/MUD_PLAN_MHC_STP_TRAINABLE.md`. Phase 3 (mHC `n=2`) deferred.
>
> **TLM CLOSED (2026-07-20):** live training telemetry TUI — `[TELEM]`→log + key parser + Weight Δ panel; pointer-optimized hot loops (P-00/P-01). Open: trainer banner LR hardcoded (display-only).

Ver **[`MUD_IMPROVEMENTS_POST_AE.md`](./docs/research/MUD_IMPROVEMENTS_POST_AE.md)**:

| ID | Tema | Prioridad sugerida |
|----|------|--------------------|
| **F** | QKV multi-matrix one CB | Alta / scoped |
| **K** | Loss certification CI | Alta / tooling |
| **G** | Multi-expert STE joint (SlimeX Dynamic Stack / ShadowExpertBus) | Producto MoE |
| **H** | Joint BPTT / seq largas | Train depth |
| **I** | KV bf16/quant | Scale RAM |
| **J** | CSA v2 LSH / W_compress | Research 1M |

> **Nota Arquitectónica (G):** La implementación de **SlimeX** permitirá acoplar/desacoplar expertos dinámicamente en tiempo de ejecución. Esto abre la puerta a un escalamiento híbrido futuro donde la carga se distribuya en caliente entre GPU (Vulkan) y CPU (AVX2/PCorePool) bajo demanda.

### Fase D scale / madurez (resto Q4)

4-pillar efficiency, bf16 shadows, pipeline overlap — objetivo &lt;2h/epoch cuando F/H/I maduren.

---

## 5. DECISIONES CLAVE (recordatorio)

| Tema | Decisión |
|------|----------|
| CPU vs GPU GEMV | **auto policy** profiled; force 0/1 via env |
| Optimizer | Por forma de matriz — **LIVE** en step |
| JEPA/mHC | OU + gate; radio ~√hidden |
| CSA | Top-k HCA en inferencia; full HCA si hay tape de train |
| ASM | Solo hot paths medidos |

---

## 6. MÉTRICAS OBJETIVO

| Métrica | Hoy (jul 2026) | Objetivo dic 2026 |
|---------|----------------|-------------------|
| Training | full-seq windows; ~ord. hours/epoch | &lt;2h/epoch |
| Inference | ~5 tok/s clase 2B (ord.) | ~20 tok/s |
| Optimizer live | Muon/GaLore/Chunked/Adam | + multi-expert STE |
| Overlap CPU/GPU | L-05 + GEMV auto | QKV one-CB (F) |
| Context | 32k HCA + CSA top-k | bf16 KV / longer |
| Tests | ~186 | 200+ + loss cert |

---

## 7. HANDOFF — SIGUIENTE SESIÓN

```
T-0 DESPEGUE GO (2026-07-16) — ver LAUNCH_COUNTDOWN.md

> **Addendum 2026-07-20:** base reconverted sane; telemetry TUI fixed (`[TELEM]`→log + ΔW panel, TLM DONE); hot loops pointer-optimized (P-00/P-01). Open debt: trainer banner LR hardcoded `3e-4`.

ÓRBITA (post-launch):
  F  QKV multi-matrix one CB
  K  loss_certification_bench → CI gate
  (G/H según prioridad producto)

LEER: GEMINI.md · LAUNCH_COUNTDOWN.md · MUD_IMPROVEMENTS_POST_AE.md
```

---

## 8. RESUMEN EJECUTIVO

```
HOY:     Motor ternario estable; GEMV AVX2×8 + auto GPU;
         optimizers LIVE (incl. Adam); full-seq train; MoE load;
         CSA top-k; HCA 32k; CI + full audit.

CERRADO: Ledger A–E + depth A–E.

LUEGO:   F/K tooling-perf → G/H MoE/train depth → I/J scale.
         + RPG Battle Circuit (Fase futura): Evolución por supervivencia, "barras de vida", debate de clones y recompensas de 5-epocas para el jugador ganador.
```

El proyecto tiene base real y el backlog de “deuda fundacional” está cerrado. La visión se ejecuta ahora como **mejoras medidas (F+)**, no como reabrir L-01.
