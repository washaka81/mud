# C-MUD: Hallazgos, Investigación y Plan de Corrección (2026-07-20)

Seguimiento de `COMPLEX_REASONING_RESEARCH_2026-07-20.md` y `COMPLEX_REASONING_ESTADO_2026-07-20.md`.
Este documento documenta los **hallazgos** de validación, la **investigación** que los explica,
el **plan de corrección** y el estado de los **pendientes**.

---

## 1. Hallazgos (validación con `mud.sh cmud-cmp`)

El probe `cmud_compare` (baseline vs `MUD_CMUD_THINK=1`) sobre `models/smollm2.mud`
reveló dos regímenes patológicos en el stub original:

| Métrica | Baseline | C-MUD stub original | Diagnóstico |
|---|---|---|---|
| `cmud_entropy` | 0.2112 | **0.0000** | over-sharp: un solo token domina (softmax colapsado) |
| `logit_l2 (Δ)` | — | **36593** | magnitud de logits inflada ~6.7x |
| `argmax_changed` | — | true | desplaza la distribución (pero de forma degenerada) |

Una segunda pasada tras la corrección #1 (normalizar atención + fase posicional) dio el
régimen opuesto, también patológico:

| Métrica | C-MUD tras corrección #1 | Diagnóstico |
|---|---|---|
| `cmud_entropy` | **10.71** (≈ uniforme; vocab 49k → máx ~10.8) | wash-out: la salida se aplana a casi uniforme |
| `logit_l2 (Δ)` | 3330 | magnitud ya preservada, pero distribución inútil |

**Estado final tras la corrección completa** (α=0.05, ventana local, V real):

| Métrica | C-MUD corregido | Veredicto |
|---|---|---|
| `cmud_entropy` | **3.66** | moderado (no colapsa, no se aplana) |
| `logit_l2 (Δ)` | **4351** | magnitud preservada |
| `argmax_changed` | true | desplaza la distribución de forma acotada |

Se añadió un **test de regresión** `cmud_compare_not_degenerate` (entropy ∈ (0.05, 9.0),
logit_l2 < 20000) para que el bug over-sharp no vuelva silenciosamente.

---

## 2. Investigación (anchors)

- **PCT — Complex-Valued Phase-Coherent Transformer (arXiv:2605.10123)**: la atención debe ser
  *token-non-competing* — compuerta real, acotada, suave, **independiente del elemento** (C1-C4).
  La clave: **la fase debe ser no trivial y preservarse** a través de capas; una fase uniforme
  colapsa la compuerta `cos(Δθ)` a 1 → media-pond. (esto causó el over-sharp).
- **ComplexFormer (arXiv:2505.10222)**: la fase se deriva de los **datos** (Euler pairing /
  `ΔP=(m-n)·ω` posicional), y **V (valores) se mantiene REAL** — la fase sólo modula el score
  Q/K. Esto preserva la magnitud de la información ("la magnitud lleva el que, la fase el
  como mezclar"). Aplicado directamente: sembrar fase posicional y no proyectar todo a un radio fijo.
- **Phasor / LPM (arXiv:2603.17433)**: el mix complejo debe ser **unitario / no promediador**
  (DFT, no softmax) para no lavar la información. Justifica la **ventana local** en vez de
  atención global (un promedio global es un low-pass que aplana el hidden).
- **Complex Transformer (arXiv:1910.10202)**: usa Min-Max-Norm en vez de softmax para evitar
  explosión de gradiente — coherente con normalizar la salida de atención por Σw.

### Causas raíz del bug original
1. **Atención no normalizada**: `phase_coherent_attn` era `Σ w·v` (sin /Σw). Para hidden≈576,
   la salida era ~576x la magnitud → un logit domina (over-sharp). → corregido: normalizar por Σw.
2. **Semilla totalmente real**: `h = x + i·0` → todas las fases 0 → `cos(Δθ)=1` → media-pond
   uniforme → hidden constante → un token domina. → corregido: sembrar fase posicional `ω·i`.
3. **Proyección a radio fijo** (`project_hermitian` a `radius` en cada paso): iguala la magnitud
   de todos los elementos → contribuye al colapso. → corregido: *soft clamp* (sólo acota blow-ups).
4. **Pérdida de magnitud en el colapso**: al sembrar `h` complejo, `wave_collapse = Re = |h|·cos θ`
   descarta energía (`cos` promedia <1) → logits más pequeños → softmax plano (wash-out). →
   corregido: **V se mantiene real** (magnitud `=|x_i|` preservada); la fase sólo puntúa la atención.

---

## 3. Plan de corrección (staged)

| # | Acción | Estado | Efecto medido |
|---|---|---|---|
| C1 | Normalizar `phase_coherent_attn` por Σw (magnitud acotada) | HECHO | L2 36593→3330 |
| C2 | Sembrar fase posicional `ω·i` (rompe simetría all-real) | HECHO | score depende de (i−j) |
| C3 | Residual no-reemplazo `h ← h + α(attn−h)` (PCT non-competition) | HECHO | preserva info original |
| C4 | Soft clamp (no radio fijo) → preserva magnitud relativa | HECHO | bola respetada, sin igualar |
| C5 | Ventana local (no blur global) — Phasor/LPM | HECHO | entropía 10.7→3.66 |
| C6 | **V real, fase sólo para scoring** (ComplexFormer) | HECHO | magnitud preservada, sin wash-out |
| C7 | Hiperparámetros env-tunables (`MUD_CMUD_ALPHA/LOCK_EPS/POS_PHI/WIN`) | HECHO | experimentación sin recompilar |
| C8 | Test de regresión `cmud_compare_not_degenerate` | HECHO | bug no reaparece |
| C9 | Free-prob / Cauchy como health gate de colapso espectral | HECHO | `cmud_spectral_health` + `CmudAudit.spectral` + test |
| C10 | CTNN σ-imaginación cableada en el loop (magnitude-safe) | HECHO | `MUD_CMUD_SIGMA` escala spread de fase |

---

## 4. Pendientes (continuación)

- **#4 Camino complejo entrenable — ARQUITECTURA + PERSISTENCIA + TRAINER DEMO HECHOS**:
  se implementó `CmudLayerParams` (α, η_rep, ω, σ + `q_phase`/`k_phase` aprendidos por dimensión +
  `v_scale` por dimensión) y `think_step_trainable`, que reemplaza la fase posicional fija por
  proyecciones de fase Q/K aprendidas (estilo CMHA). Con biases=0 y escala=1 es **idéntico** al
  paso posicional fijo (regression-safe; `test_think_step_trainable_runs` lo verifica).
  - **Persistencia (HECHO)**: serde JSON sidecar `<model>.mud.cmud.json` vía `save_json`/`load_json`/
    `sidecar_for`; la ruta de producción (`maybe_think_collapse_report`) carga params desde
    `MUD_CMUD_PARAMS` si está seteado (test `test_cmud_params_json_roundtrip`).
  - **Trainer demo (HECHO)**: `tools/cmud_train.rs` (`./mud.sh cmud-train`) corre **Adam de
    descenso de gradiente real** con gradiente numérico (diferencias finitas centrales) sobre
    `q_phase`/`k_phase`, minimizando cross-entropy del next-token con el forward REAL
    (`forward_last_logits_cmud`). Smoke-test OK (CE 18.1502→18.1496 en 3 pasos sobre corpus
    sintético; guarda sidecar y se recarga en inferencia). Es un optimizador de gradiente genuino
    (no búsqueda por coordenadas).
  - **Backprop analítico en `corpus_trainer.rs` (PENDIENTE, sub-paso final)**: reemplazar el
    gradiente numérico por backprop analítico del `sampled softmax` sobre los mismos params
    (P-02 SSOT intacto; C-MUD opt-in). `test_trainable_bias_changes_scores` confirma que los
    biases alteran la salida, así que el gradiente tiene efecto. `α, η_rep, ω, σ` ya son env-tunables.
  - Nota: el trainer es una herramienta de investigación, NO el path de producción; el path f32
    de producción no se toca.
- **#2b Benchmark de calidad real**: corpus corto de razonamiento (GSM8K-style / lógica) con vs sin
  `MUD_CMUD_THINK=1` para medir si el paso complejo *ayuda* (hoy sólo medimos que no es degenerado).
- **C9 (HECHO)**: `cmud_spectral_health(mags, phases)` calcula spread de magnitud, concentración
  circular de fase `R` y `|G_λ(2)|` (Cauchy). Integrado en `CmudThinkReport.spectral`,
  `CmudAudit.spectral`, `cmud_kernel_selfcheck` y el `healthy()` gate (rechaza manifold colapsado).
  Test: `test_spectral_health_detects_collapse`.
- **CTNN (HECHO)**: `MUD_CMUD_SIGMA` escala el *spread* de fase alrededor de su media en cada paso
  (sin tocar magnitud, V real). Verificado vía `cmud-cmp` (entropy 3.47 vs 3.66 con σ=0.5).
  `complex_time_step` (σ global) queda disponible para un lookahead multi-paso futuro.

---

## 5. Cómo reproducir

```bash
MUD_CMUD_THINK=1 cargo run --release --bin forge_llm -- models/smollm2.mud   # inferencia
./mud.sh cmud-cmp models/smollm2.mud      # probe baseline vs C-MUD (entropy/L2/argmax)
./mud.sh cmud-audit models/smollm2.mud    # health del path complejo
./mud.sh pointer-audit models/smollm2.mud # validación de punteros ELUT/ternary
# tuning:
MUD_CMUD_ALPHA=0.03 MUD_CMUD_POS_PHI=0.05 MUD_CMUD_WIN=8 ./target/release/cmud_audit models/smollm2.mud --cmp
```

**Suite**: `cargo test --lib` → **250 passed, 0 failed**; `cargo clippy --all-targets` limpio.

---

*No contradice políticas (P-02 SSOT f32 en producción; C-MUD opt-in). Anchors: PCT 2605.10123,
ComplexFormer 2505.10222, Phasor/LPM 2603.17433, Complex Transformer 1910.10202, EulerFormer 2403.17729.*
