# C-MUD: Razonamiento Complejo — Estado y Documentación (2026-07-20)

**Módulo:** `src/mud/cmud.rs` (L-14) — kernel de investigación opt-in.
**Activación:** variable de entorno `MUD_CMUD_THINK=1`.
**Path de producción (`SlimeRegister` f32, P-02):** NO se toca; el paso complejo es
complementario y sólo se ejecuta cuando el flag está activo.

---

## 1. Qué tenemos implementado

### 1.1 Núcleo algebraico (ya existía, L-14)
| Símbolo | Rol |
|---------|-----|
| `ComplexF32 { re, im }` | estado complejo `re + i·im` |
| `GaussTernary { wr, wi }` | peso gaussiano-ternario ∈ {-1,0,1} → 9 estados |
| `gauss_mul` / `gauss_mac` | `Y_R = X_R W_R − X_I W_I`, `Y_I = X_R W_I + X_I W_R` |
| `project_hermitian(h, R)` | bala Hermitiana (mHC): `‖h‖²_C ≤ R²` |
| `phase_delta` | distancia angular L1 en el círculo |
| `wave_collapse(h)` | lectura a LM-head: `Re·(1+tanh|Im|)·cos θ` |
| `ThinkingState` | bucle de "thinking" τ sobre un vector oculto complejo |
| `think_step_stub` | paso stub: mezcla residual `h + α·i·h`, proyección Hermitiana |
| `ComplexSlimeRegister` | registro de investigación (estado + fase previa para ω) |

### 1.2 Kernels de razonamiento complejo (nuevos, esta sesión)
| Función | Sección doc | Qué hace |
|---------|------------|----------|
| `phase_coherent_attn(q,k,v,gate,out)` | §3.1 (PCT/CMHA) | atención fase-coherente: `score = cos(Δθ)`, compuerta real suave, mezcla de V preservando fase |
| `cue_phase_repulsion(phases, η)` | §3.2 (E1, Dyson CUE) | regularizador de repulsión de fase `−Σ log|e^{iθ_i}−e^{iθ_j}|²` + gradiente que separa fases (anti-colapso) |
| `cauchy_transform(λ, z)` | §3.3 (CFT/free prob) | `G(z)=1/N·Σ 1/(z−λ_j)` — lectura espectral del hidden |
| `r_transform_add(ra, rb)` | §3.3 | aditividad de R-transform `R_A⊕B = R_A + R_B` (convolución libre) |
| `contour_rotate(h, φ)` | §3.4 (E3) | rotación de contorno / continuación analítica; preserva norma Hermitiana |
| `complex_time_step(st, α, σ)` | §3.5 (CTNN) | eje de tiempo-complejo `σ` (memoria/imaginación) en el bucle de thinking |
| `think_step_phase_attn` (método) | §3.1+§3.2 | **paso de thinking real**: auto-atención fase-coherente + repulsión de fase + proyección Hermitiana |
| `maybe_think_collapse(x, R)` | orquestador | semilla desde oculto real, itera `think_step_phase_attn` hasta phase-lock, colapsa a real; no-op sin flag |

### 1.3 Gancho en el forward real
`src/mud/inference.rs` → `forward_last_logits` (línea ~364): tras construir el hidden final
`reg_f32`, llama `cmud::maybe_think_collapse(&mut reg_f32, 2·RMS)` **sólo si `MUD_CMUD_THINK=1`**.
El radio se escala al RMS del hidden para que la bala Hermitiana no recorte en la semilla.

---

## 2. Mapeo con la investigación (web, 2026)

| Fuente | Idea | Dónde vive en `cmud.rs` |
|--------|------|--------------------------|
| **PCT** arxiv 2605.10123 | atención L2-normalizada, compuerta suave, no competitiva | `phase_coherent_attn` |
| **ComplexFormer/CMHA** 2505.10222 | score `cos(Δθ)` + rotación adaptiva | `phase_coherent_attn` |
| **PRISM** 2512.01208 | tokens-fasor + colapso de onda al vocabulario | `wave_collapse` (ya existía) |
| **Dyson CUE / log-gas** | repulsión de fase en el círculo unidad | `cue_phase_repulsion` (E1) |
| **Free probability / Virasoro** | composición de capas complejas | `cauchy_transform`, `r_transform_add` |
| **CTNN** (tiempo-complejo) | `T=t+iτ`: memoria/imaginación | `complex_time_step` (E3) |

El item de orbit **C-MUD × log-gas** (antes DEFERRED) quedó parcialmente cubierto:
repulsión de fase (E1) y rotación de contorno (E3) ya tienen kernel y test.

---

## 3. Tests (17 en `cmud::`, todos verdes)

| Test | Valida |
|------|--------|
| `test_phase_coherent_attn_basis` | `cos(π/2)=0`, `cos(0)=1` |
| `test_cue_phase_repulsion_spreads` | fases iguales → R grande; separadas → R pequeña; empuja aparte |
| `test_cauchy_transform_far_field` | `G(10) ≈ 0.1` para autovalores en el círculo |
| `test_r_transform_add_additive` | `R_A+R_B` elemento a elemento |
| `test_contour_rotate_preserves_norm` | rotación preserva `‖h‖` |
| `test_think_step_phase_attn_runs` | paso real finito, dentro de la bala Hermitiana, colapsa a real |
| `test_complex_time_step_runs` | eje σ se aplica, τ avanza, finito |
| + 10 tests base (gauss_mul, 9 estados, proyección, wave_collapse, thinking stub, gemv, env…) | |

Además, **end-to-end**: `inference::tests::cmud_think_forward_smoke` corre el forward completo
con `MUD_CMUD_THINK=1` y exige logits finitos y con rango dinámico.

**Suite total:** `cargo test --lib` → **241 passed, 0 failed, 2 ignored**; `cargo clippy --all-targets` limpio.

### 3.1 Validación extendida (auditoría de punteros + probe de calidad)

Nuevas herramientas que validan el cálculo de direcciones de punteros crudos de todo el
proyecto (`pointer_audit`) y comparan baseline vs C-MUD (`cmud-cmp`):

- **`pointer_audit`** (`src/mud/pointer_audit.rs` + `tools/pointer_audit.rs`, `./mud.sh pointer-audit`):
  cruza `dequantize_ternary_row`/`pack_ternary_into`/`pack_elut_prq`/`unpack_ternary2bit_to_f32`
  contra la fórmula de extracción de `tools/training_healthcheck.rs`
  (`u32_idx = k/8; shift = (k%8)*4; bits = (*(ptr+u32_idx) >> shift) & 0xF`).
  **Resultado contra `models/smollm2.mud`: 210 tensores, 106 168 320 elementos, 0 mismatches,
  max_abs_err = 0.00e0** — el layout ELUT/ternary en mmap es idéntico al kernel de referencia.
- **`cmud_audit`** (`mud.sh cmud-audit`): `forward_ok=true`, `logit_range_min=1487.5`,
  `token0_dominant=false`, `τ=8`, `phase_locked=false`, `herm norm max 72.0/72.0` (bola respetada).
- **`cmud-cmp`** (`mud.sh cmud-cmp`): el stub original daba `cmud_entropy=0.0000`
  (over-sharp) y `logit_l2≈36593` (magnitud inflada). **Corregido** (ver
  `COMPLEX_REASONING_FINDINGS_2026-07-20.md`): normalizar atención por Σw, sembrar fase
  posicional `ω·i`, residual no-reemplazo, soft-clamp (no radio fijo), ventana local y
  **V real** (fase sólo para scoring, ComplexFormer). Resultado final:
  `cmud_entropy≈3.66`, `logit_l2≈4351` — perturbación acotada, no degenerada.
  Hay test de regresión `cmud_compare_not_degenerate`.
- Ambas herramientas entraron al battery CI (`mud.sh ci`): `pointer-audit` + `cmud-cmp`.

**Suite total (post-validación + #4):** `cargo test --lib` → **254 passed, 0 failed, 2 ignored**;
`cargo clippy --all-targets` limpio (0 warnings).

---

## 4. Cómo usarlo

```bash
# Inferencia con razonamiento complejo opt-in:
MUD_CMUD_THINK=1 cargo run --release --bin forge_llm -- models/smollm2.mud
```
El forward produce el hidden final, lo semilla en `ThinkingState`, itera `think_step_trainable`
(auto-atención `cos(Δθ)` con fase Q/K aprendida + repulsión de fase) hasta phase-lock
(`EMA(ω) < 1e-3`, tunable vía `MUD_CMUD_LOCK_EPS`) o `DEFAULT_THINK_ITERS=8`, y colapsa a
logits reales vía `wave_collapse`. Con `q_phase=k_phase=0`, `v_scale=1` el paso es idéntico al
positional-phase fijo (`think_step_phase_attn`). Los params entrenados se cargan desde
`MUD_CMUD_PARAMS` (sidecar JSON).

Parámetros ajustables (constantes en `cmud.rs`): `PHASE_LOCK_EPS=1e-3`,
`OMEGA_EMA_RATE=0.1`, `DEFAULT_THINK_ITERS=8`.

---

## 5. Limitaciones / deuda abierta
- **#4 entrenable (HECHO parcial)**: hay `CmudLayerParams` + `think_step_trainable` + persistencia
  (sidecar JSON) + trainer demo FD (`./mud.sh cmud-train`). Falta cablear descenso de gradiente
  real en `corpus_trainer.rs` (P-02 SSOT intacto). El FD trainer es herramienta de investigación.
- `cue_phase_repulsion` aplica el empuje con `η=0.01` fijo; el empuje en fases casi idénticas es
  singular (se protege con `is_finite`).
- Hay métrica de *calidad* (`cmud-cmp`, `mud.sh cmud-cmp`): compara baseline vs C-MUD en un prompt
  (argmax Δ, logit L2, entropía). Tras las correcciones el paso ya NO es sobre-pico
  (`cmud_entropy≈3.5`, `logit_l2≈4351`) — ver §3.1.
- El path AVX2/real sigue siendo el de producción; C-MUD es un paso post-forward adicional opt-in.

---

## 6. Próximos pasos sugeridos
1. **(HECHO) Arreglar el sobre-pico del stub** — ver `COMPLEX_REASONING_FINDINGS_2026-07-20.md`
   (causas raíz + correcciones C1–C8). El bug over-sharp y el wash-out están resueltos y
   cubiertos por `cmud_compare_not_degenerate`.
2. **Camino complejo entrenable (#4) — HECHO parcial:** `CmudLayerParams` + `think_step_trainable`
   (proyecciones de fase Q/K aprendidas por dimensión) + persistencia (sidecar JSON) + trainer demo
   FD (`./mud.sh cmud-train`). Falta cablear descenso de gradiente real del `sampled softmax` en
   `corpus_trainer.rs` (C-MUD opt-in, P-02 SSOT intacto).
3. **Comparación de calidad en corpus:** GSM8K-style / lógica con vs sin `MUD_CMUD_THINK=1`.
4. **(HECHO) C9 / CTNN:** `cmud_spectral_health` (Cauchy/free-prob) como health gate de colapso
   espectral en `CmudAudit` + `healthy()`; `MUD_CMUD_SIGMA` respira el spread de fase (CTNN).
   Ver `COMPLEX_REASONING_FINDINGS_2026-07-20.md` §3 (C9/C10).
5. **Trainer demo (HECHO):** `tools/cmud_train.rs` (`./mud.sh cmud-train`) corre **Adam de
   gradiente real** (gradiente numérico por diferencias finitas centrales) sobre `q_phase`/`k_phase`
   minimizando next-token CE con el forward real; guarda sidecar JSON recargable.

---

*Investigación base: `docs/research/COMPLEX_REASONING_RESEARCH_2026-07-20.md`
(ancoras arxiv PCT 2605.10123, ComplexFormer 2505.10222, PRISM 2512.01208, CTNN 2026,
Dyson CUE / Forrester). No contradice políticas (P-02 SSOT f32 en producción).*
