# MUD Project Documentation Index

Documentación del motor Forge LLM (MUD). **Índice:** 2026-07-16 (post streams A–E).

## Jerarquía canónica (raíz del repo)

| Documento | Rol | Autoridad |
|-----------|-----|-----------|
| [`../GEMINI.md`](../GEMINI.md) | Políticas P-#, estado, ledger **L-##** | **SSOT** técnico |
| [`../AGENTS.md`](../AGENTS.md) | Contexto para agentes (derivado) | No contradice GEMINI |
| [`../VISION_ROADMAP.md`](../VISION_ROADMAP.md) | Visión de producto + fases Q3–Q4 | **SSOT** dirección |
| [`../PLAN_MAESTRO.md`](../PLAN_MAESTRO.md) | Narrativa MoE + métricas | Expansión de visión |
| [`STATUS_REPORT.md`](STATUS_REPORT.md) | Logros vs deuda (resumen) | Alineado a GEMINI §0/§6 |
| [`../README.md`](../README.md) | Intro del proyecto | Público |

Si hay conflicto: **GEMINI** gana en estado/políticas; **VISION_ROADMAP** en dirección de producto.

## Estructura de `docs/`

| Carpeta | Contenido |
|---------|-----------|
| [`architecture/`](architecture/) | Specs y stack — **empieza por** [`MUD_COMPUTE_STACK.md`](architecture/MUD_COMPUTE_STACK.md) |
| [`sessions/`](sessions/) | Reportes de sesión — **A–E:** [`MUD_SESSION_REPORT_2026-07-16_STREAMS_AE.md`](sessions/MUD_SESSION_REPORT_2026-07-16_STREAMS_AE.md) |
| [`audits/`](audits/) | Auditorías históricas (`MUD_AUDIT_LATEST.md` puede retrasarse vs GEMINI) |
| [`research/`](research/) | Papers, gap analysis, **mejoras F+** |
| [`manuals/`](manuals/) | Protocolos + **[`LAUNCH_COUNTDOWN.md`](manuals/LAUNCH_COUNTDOWN.md)** (T-minus despegue) |
| [`dumps/`](dumps/) | Dumps temporales de debug |

### Research vivo (post-ledger)

| Doc | Rol |
|-----|-----|
| [`research/MUD_GAP_ANALYSIS_POST_L15.md`](research/MUD_GAP_ANALYSIS_POST_L15.md) | Residuales P0–P3 + env knobs A–E |
| [`research/MUD_IMPROVEMENTS_POST_AE.md`](research/MUD_IMPROVEMENTS_POST_AE.md) | **Siguiente backlog F–L** (QKV, MoE STE, BPTT, KV quant, CSA v2, tooling) |
| [`research/DEEPSEEK_V4_TERNARY_INTEGRATION.md`](research/DEEPSEEK_V4_TERNARY_INTEGRATION.md) | Teoría mHC / CSA / Muon / DSpark |

## Por dónde empezar

### Operador / humano
1. [`VISION_ROADMAP.md`](../VISION_ROADMAP.md) — fase actual y norte  
2. [`GEMINI.md`](../GEMINI.md) §0 — verdad de runtime  
3. [`architecture/MUD_COMPUTE_STACK.md`](architecture/MUD_COMPUTE_STACK.md) — ELUT / FP32 / AVX2×8 / ash auto  
4. [`manuals/MUD_USER_MANUAL.md`](manuals/MUD_USER_MANUAL.md) — convertir / entrenar (si aplica)

### Agente / siguiente sesión de código
1. [`GEMINI.md`](../GEMINI.md) §0 + §6.4 handoff  
2. [`research/MUD_IMPROVEMENTS_POST_AE.md`](research/MUD_IMPROVEMENTS_POST_AE.md) — **empezar por F o K**  
3. [`AGENTS.md`](../AGENTS.md) — runtime truth + comandos  
4. [`sessions/MUD_SESSION_REPORT_2026-07-16_STREAMS_AE.md`](sessions/MUD_SESSION_REPORT_2026-07-16_STREAMS_AE.md) — qué se cerró en A–E  

**No** reabrir L-01 ni “Phase A debt” como si estuviera abierta.

## Estado LIVE (recordatorio 2026-07-16)

| Pieza | Live |
|-------|------|
| Forward GEMV | AVX2×8 + **auto ash** (`MUD_GPU_GEMV`) |
| Accum | FP32 |
| QAT step | Muon / GaLore / Chunked / **Adam** + STE pack |
| Train | **Full-seq** windows (default) |
| MoE | Load + train-expert |
| Context | HCA 32k + **CSA top-k** |
| LM head | `lm_head_logits_avx2` |
| Tests | ~186 lib; `./mud.sh ci\|audit-full` |

## Roadmap por fases (resumen)

| Fase | Estado |
|------|--------|
| A — L-01…L-08 | **DONE** |
| B — Perf / GEMV GPU | **DONE** (+ stream C auto) |
| C — L-09…L-11 Modular | **DONE** (+ stream B MoE load) |
| D/E ledger L-12…L-15 | **DONE** |
| Depth A–E | **DONE** |
| **F+ improvements** | **OPEN** → `MUD_IMPROVEMENTS_POST_AE.md` |

Detalle: `VISION_ROADMAP.md` + `GEMINI.md` §6.
