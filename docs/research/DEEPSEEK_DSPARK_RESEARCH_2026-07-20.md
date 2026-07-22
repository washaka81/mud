# Research: DeepSeek DSPARK — Speculative Decoding Semi-Autoregresivo

**Fecha:** 2026-07-20 · **Fuente:** web research (arxiv 2607.05147, repos DeepSpec/HuggingFace, V4 paper 2606.19348, RFC DeepSpec #52).
**Licencia:** MIT (DeepSpec + checkpoints) → **adaptable** a MUD sin restricción.

---

## 1. Qué es DSPARK

`DSpark` = *Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
Generation* (DeepSeek-AI, 2026). **No es un modelo nuevo** — es un **framework de
inferencia** que acelera el decoding especulativo: un *draft* rápido propone
varios tokens, el *target* (modelo gran) los verifica en un solo pase.

Se open-sourcó junto a **DeepSpec** (MIT): codebase completa para entrenar y
evaluar drafters (data-prep → train → eval). Incluye 3 drafters:
**DSpark** (semi-autoregresivo), **DFlash** (paralelo puro), **Eagle3** (autoregresivo TTT).

---

## 2. Arquitectura (las 2 innovaciones)

### 2.1 Generación semi-autoregresiva
- **Backbone paralelo** propone un bloque de `γ` tokens (~5–7) en **un** forward.
- **Cabeza secuencial ligera (Markov head)** añade dependencia intra-bloque:
  cada token draft se sesga por el token aceptado previo → mitiga el
  **"suffix decay"** (los tokens finales de un bloque paralelo se rechazan mucho).
- Resultado: retiene el throughput del drafter paralelo pero recupera calidad
  cercana al autoregresivo.

### 2.2 Verificación por confianza + scheduler hardware-aware
- **Confidence head** (entrenado end-to-end, calibrado con *Sequential Temperature
  Scaling*) predice la **probabilidad de supervivencia del prefijo**.
- **Hardware-aware prefix scheduler** ajusta **cuántos tokens verificar** por request
  según: (a) supervivencia estimada, (b) **carga actual del motor**.
  - GPU libre → verifica bloque largo (menor latencia por usuario).
  - Pico de carga → verifica solo el prefijo de alta confianza, descarta el
    sufijo de baja confianza *antes* de consumir cómputo.
- Esto evita el **cliff de throughput** bajo SLAs estrictos (donde verificar
  bloques largos mata la concurrencia).

---

## 3. Resultados (despliegue real, no solo benchmark)

| Target | Speedup per-user | Condición |
|--------|-------------------|------------|
| DeepSeek-V4-Flash | **+60%–85%** | mismo throughput agregado |
| DeepSeek-V4-Pro | **+57%–78%** | mismo throughput agregado |

Despliegue en el sistema de serving V4 (vs baseline MTP-1). Bajo SLAs
estrictos (120 TPS Flash / 50 TPS Pro) el baseline se degrada severo;
DSpark mantiene throughput robusto. Corrie el **Pareto frontier** del serving.

---

## 4. Training del drafter (barato y escalable)

- **Target congelado**; el draft comparte `embedding` + `LM head` (congelados),
  se entrena solo: backbone drafter + bloque secuencial + confidence head.
- **Anchor-bounded sequence packing:** se muestrean `N` *anchors* de la secuencia
  target, se empaquetan esos bloques de predicción aisłados en batches densos
  vía **token-level attention indices** (no máscaras 2D) → mantiene masking
  causal exacto entre secuencias/anchors independientes, **sin padding**.
- **Hidden-state communication:** en vez de shipear logits de todo el vocabulario
  `O(V)` entre workers, se cachean las activaciones del target y se comunica
  **solo el hidden-state previo a la LM-head** → complejidad de comm `O(d)`
  (d = hidden). Reduce drásticamente el ancho de banda en entrenamiento
  distribuido del drafter.

---

## 5. Ecosistema / checkpoints (MIT)

- DeepSpec: `github.com/deepseek-ai/DeepSpec` (data/train/eval).
- Checkpoints DSpark para **Qwen3-{4B,8B,14B}**, **Gemma4-12B**, y
  **DeepSeek-V4-Flash/Pro** (en HuggingFace).
- Configs de entrenamiento por target familia.

---

## 6. Relevancia DIRECTA a MUD (mapeo a piezas existentes)

| Técnica DSPARK | Pieza MUD equivalente | Nota |
|----------------------|--------------------------|------|
| Speculative drafter | **`src/mud/...drafter`** (ya existe, zero-alloc) | MUD YA tiene drafter; DSPARK mejora su arquitectura |
| Anchor-bounded packing (token-level idx) | **L-10 `sequence_pack`** (full-chunk pairs, no pad) | MUD ya empaceta sin pad; adoptar *attention indices* para causal multi-secuencia |
| Hardware-aware scheduler (varía γ por carga) | **`gemv_policy`** (CPU/GPU auto) | MUD ya tiene dispatch por carga; subir al nivel *speculativo* |
| Hidden-state comm `O(d)` | **PCorePool** (8 P-cores, sin Rayon) | evitar shipear logits `O(V)` entre hilos/workers |
| Spherical norm (alineación draft-target) | **JEPA / mHC** (normalize manifold) | RFC DeepSpec #52 (V22): `F.normalize` previene "rejection cascade" en MoE router misalignment — reusa la normalización que YA tiene MUD |
| Confidence head + STS calibración | (nuevo, pequeño) | cabeza ligera entrenable sobre el draft de MUD |

**Conclusión:** DSPARK no es un giro; es un conjunto de técnicas que **potencian
el drafter y el packing que MUD YA tiene**, más una cabeza de confianza que MUD
no tiene. Todo bajo MIT.

---

## 7. Propuesta de adopción en MUD (track de research "DSP")

> Mantener como **backlog de investigación** (no bloquea launch). Cada ítem con
> validación CORTA (segundos), coherente con `MUD_PLAN_PRIORIDADES_ROADMAP_2026-07-20.md`.

| ID | Ítem | Acción concreta | Validación CORTA |
|----|------|--------------------|----------------------|
| **DSP-1** | Drafter semi-autoregresivo | Añadir **Markov/sequential head** al drafter existente: backbone paralelo (γ tokens) + cabeza secuencial que bias por token previo. Mitiga suffix decay. | `cargo test` drafter: bloque γ=5 aceptado > autoregresivo puro en mismo target sintético (segundos). |
| **DSP-2** | Confidence head + scheduler | Cabeza de confianza (pequeña) sobre el draft; scheduler que **varía γ** según carga del `PCorePool` (reusa `gemv_policy`). | Test: bajo "carga alta" simulada, γ_effective baja y throughput se mantiene (bench <1 min). |
| **DSP-3** | Anchor-bounded packing | Extender **L-10** a *token-level attention indices* (no 2D mask) para causal multi-anchor sin padding en el drafter. | `cargo test sequence_pack` assert: masking causal exacto en 3 secuencias empaquetadas, 0 padding. |
| **DSP-4** | Hidden-state comm `O(d)` | En PCorePool, comunicar solo hidden-state previo a LM-head (no logits `O(V)`) entre hilos de entrenamiento del drafter. | Test de comm: ancho de banda mock `O(d)` vs `O(V)`; assert menor copies. |
| **DSP-5** | Spherical norm draft-target | Reusa **JEPA/mHC** (`F.normalize` sobre hidden) para alinear draft-target y evitar rejection cascade (RFC #52). | Test: tras spherical norm, tasa de rechazo del draft baja vs sin norm (sintético). |

---

## 8. Adyacencia útil (DeepSeek-V4 paper 2606.19348)

El paper V4 trae más que DSPARK, relevante al backlog existente de MUD:
- **Hybrid attention CSA/HCA** (compressed sparse + heavily compressed) → MUD YA tiene **CSA top-k (stream E)** y HCA; V4 usa *hash routing* en las primeras capas MoE → conecta con **J** (CSA LSH) del backlog.
- **FP4 MoE + TileLang DSL** → related a `MUD_KV_DTYPE=f16` (I) y ELUT 4-bit.
- **Muon + ZeRO híbrido** (BF16 grad sync, Newton-Schulz en BF16) → MUD YA tiene **Muon LIVE (L-01)** + `adam_state`; confirmación de que el camino es correcto.
- **Anticipatory routing** (router de `t-delta` para evitar loss spikes) → patrón de *runtime intervention* similar a como MUD usa `MUD_TRAIN_WCLAMP_K` para evitar colapso de escala.

---

## 9. Referencias

- DSpark paper: `arxiv.org/abs/2607.05147` (DeepSeek-AI, 2026).
- DeepSpec: `github.com/deepseek-ai/DeepSpec` (MIT) — train/eval drafters.
- DeepSeek-V4: `arxiv.org/abs/2606.19348` (hybrid attention, FP4 MoE, TileLang, Muon/ZeRO).
- RFC DeepSpec #52 (V22 white-box): spherical norm + 3-tier semantic cache (O(1) context compression alternativa a KV-cache).
- Checkpoints: HuggingFace `deepseek-ai/DeepSeek-V4-{Flash,Pro}-DSpark`, `dspark_qwen3_*`, `dspark_gemma4_12b`.

---

*Integrado en `GEMINI.md` §9 (nota DSpark open/MIT) y `AGENTS.md`
(historical + ledger). No contradictce políticas; es backlog de research F+.*
