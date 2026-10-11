# Auditoría de Pérdida de Loss + Propuestas de Mejora (2026-08-03)

**Fecha:** 2026-08-03
**Contexto:** el log real muestra loss 5.0 → 2.86 en ~1100 steps con estancamiento posterior. Este documento audita el pipeline de entrenamiento y propone mejoras concretas para **bajar el loss**, priorizadas por impacto/esfuerzo. Complementa `MUD_AUDIT_FULL_2026-08-03.md` (auditoría general) y `MUD_AUDIT_ASM_KERNELS_2026-08-03.md` (kernels).

**Caveat histórico importante:** los reportes 2026-08-02/03 muestran que la mitad de los "estancamientos" observados fueron **bugs de telemetría** (doble división en `push_grad`, cabecera del log, placeholders). Verificar cada mejora con `cargo test --lib loss_cert` + run corto de 100 chunks antes de commits largos.

---

## 1. BUGs de entrenamiento verificados

### B-1 — `output_norm_w_grad` NUNCA se resetea entre chunks (acumulación sin límite) — ALTA

**Archivo:** `src/mud/corpus_trainer/train_step.rs:756, 766, 776` (acumulación) / `:1404-1421` (aplicación)
**Severidad:** BUG correcto — el gradiente de la norma de salida se acumula con `+=` sobre TODOS los pares de TODA la sesión:

```rust
// :756 — resize NO zeroes elementos existentes (solo los nuevos)
backward_ws.output_norm_w_grad.resize(hidden_size, 0.0);
// :766/:776 — += sobre el buffer (nunca se limpia tras el step)
```

Cada chunk aplica `shadow -= lr_norm × Σ_acumulado` (`:1416` `sgd_step`). Como el buffer jamás se pone a cero, el paso efectivo **crece linealmente con el nº de chunks** (comportamiento de controlador integral, no de SGD): la norma de salida se clava en el clamp [0.01, 10] (degenera a escala constante) en pocas docenas de chunks. Contraste: los gradientes por capa SÍ se resetean por chunk (`gradients[l_idx].reset()`, `:306`).

El comentario `:1406` ("accumulated across all tokens") lo presenta como intencional — es un bug con apariencia de diseño. **Fix:** zeroing del buffer tras el `sgd_step` de cada chunk (`mem::take` + `resize(0)` o `fill(0.0)`).

### B-2 — `MUD_GRAD_CLIP` es un env MUERTO — MEDIA

**Archivo:** `src/mud/constants.rs:40-46` (definición, documentado en `:32-39`) — **nunca llamado en ningún sitio**.
El clip real está hardcodeado en `train_step.rs:879-880` (`let clip = if grad_norm > 1.0 { 1.0/grad_norm } else { 1.0 }`). El comentario de constants promete "For fine-tuning an already-stable model, set MUD_GRAD_CLIP=0.1" — esa funcionalidad no existe. **Fix:** `let clip = ... if grad_norm > qat_grad_clip() { qat_grad_clip()/grad_norm }` (o exponer el umbral).

### B-3 — Embeddings congeladas por defecto = CABEZA LM congelada — ALTO IMPACTO EN LOSS

**Archivo:** `src/mud/constants.rs:145-148` (`train_freeze_emb()` default **true**) / `session.rs:40, 234-246` / gate en `train_step.rs:937-939, 960-991`
**Mecanismo:** el logit del sampled softmax es `dot(final_x, emb_row)` — las filas de embedding SON la cabeza (tied). Con `freeze_emb=true` la cabeza **nunca se entrena**: el modelo solo puede bajar loss moviendo las features de las últimas capas hacia filas fijas — un piso estructural. Esto explica el estancamiento en ~2.86 (≈ buen ajuste con readout estático). Coste de descongelar: ~1 GiB FP32 extra en 152k×2048 (el path emb usa SGD con clip per-example ya implementado en `:960-991`).

### B-4 — Solo las últimas 4-8 capas entrenan por defecto — MEDIA

**Archivo:** `sequence_pack.rs:87-123` (`train_last_n_layers`): `>16 capas → 4`, `>8 → 8`. Bonsai (28 capas) → **4**. SmolLM2 → 8. Combinado con B-3, la capacidad entrenable es una fracción pequeña del modelo. Default razonable para RAM UMA, pero es la palanca #2 para bajar loss.

### B-5 — GaLore es pass-through (sin Adam en `g_low`) — MEDIA

**Archivo:** `src/mud/galore.rs:118` — comentario explícito "(Adam would go here on g_low, but we just pass through…)". `galore_step` proyecta el gradiente a un subespacio rank-r y lo devuelve → "SGD con filtro de subespacio", no GaLore real. `select_optimizer` (matrices tall) promete más de lo que entrega.

### B-6 — Menores

- **LR off-by-one:** `apply_lr_schedule` se llama con `step_count` ANTES del `fetch_add` (`train_step.rs:1023` vs `:1455`) → el chunk N usa el LR del paso N-1.
- **`_weight_decay` sin usar** (`train_step.rs:1015`); el WD real es `decay=0.01` (`:1029`) que en la rama SGD es `1 - lr·wd ≈ 0.999995` — prácticamente nulo.
- **Negativos uniformes:** `train_step.rs:558-578` — sampling uniforme puro, sin hard negatives ni in-batch; con la cabeza congelada (B-3) el gradiente de negativos es poco informativo.

---

## 2. Propuestas priorizadas para BAJAR EL LOSS

> Escala: impacto ★ 0-5 (en el loss) · esfuerzo ● 0-5 (dificultad). Ordenadas por ROI.

| # | Propuesta | Archivo:línea | Mecanismo | Riesgo | Imp. | Esf. |
|---|-----------|---------------|-----------|--------|:---:|:---:|
| **P1** | **Descongelar embeddings** (`MUD_TRAIN_FREEZE_EMB=0`) | `constants.rs:147`, `train_step.rs:939/960` | La cabeza atada es el único punto de lectura; mover filas reduce loss directo sobre la clase correcta | RAM ~1 GiB FP32; riesgo de colapso si LR alta → usar `MUD_QAT_LR=2e-4` + clip emb existente | ★★★★★ | ● |
| **P2** | **Entrenar TODAS las capas** (`MUD_TRAIN_LAST_N_LAYERS=all`) | `sequence_pack.rs:87`, `session.rs:1025` | 4/28→28 capas con gradiente; el stack bajo hoy aporta solo features fijas | OOM en UMA → combinar con `MUD_GRAD_CKPT=1` + `MUD_TRAIN_STEPS_PER_CHUNK` menor | ★★★★☆ | ● |
| **P3** | **Fix B-1: resetear `output_norm_w_grad` por chunk** | `train_step.rs:1404-1452` | Corrige LR efectiva divergente; la norma de salida deja de clavarse en clamps | Bajo (test de equivalencia) | ★★★★☆ | ●● |
| **P4** | **Hard negatives + in-batch negatives** | `train_step.rs:558-578` | Reemplazar 255 neg uniformes por ~240 uniformes + 8-16 de la distribución predicha (argmax del batch anterior, cacheado) → gradiente softmax más informativo | Sesgo de sampling; costo extra pequeño | ★★★★☆ | ●●●● |
| **P5** | **Wire `MUD_GRAD_CLIP` (fix B-2)** | `train_step.rs:879, 943` | Activa el override documentado; permite clip 0.1 en fine-tune de modelos estables | Ninguno (ya está el código) | ★★☆ | ● |
| **P6** | **Deadzone shrink schedule** | `constants.rs:7` (0.7), `ezop.rs:1261`, `optim.rs:567` | Bajar `SPARSITY_THRESHOLD_RATIO` 0.7→0.35 tras el 40% del entrenamiento (o env): reactiva pesos muertos que hoy reciben ΔW=0 | Menos sparsity; reauditar modelo tras train | ★★★☆ | ●●● |
| **P7** | **Adam/SparseAdam para filas de embedding** | `train_step.rs:970-991` | Las filas usan SGD plano con clip; con momentos Adam bajan más rápido y estable | RAM extra (moments ≈ 2× emb); usar SparseAdam con filas activas | ★★★☆ | ●●● |
| **P8** | **STP loss ON por defecto (λ=0.02)** | `stp_loss.rs`, `train_step.rs:786-860` | Señal auxiliar de trayectoria (1-cos(d1,d2)) ya implementada y zero-alloc; hoy default OFF | Puede competir con CE; λ pequeña; testear 100 chunks | ★★☆ | ●● |
| **P9** | **Curriculum de negativos** (63→255→511) | `sequence_pack.rs:57`, `train_step.rs:249` | Iniciar con menos clases (loss fácil) y subir dureza; mejora señal temprana cerca de random | Reentrenar calendario | ★★☆ | ●● |
| **P10** | **Fix off-by-one LR (B-6)** | `train_step.rs:1023/1455` | Chunk N usa el LR correcto del paso N | Ninguno | ★☆☆ | ● |
| **P11** | **GaLore real (Adam en `g_low`) o descartar la rama** | `galore.rs:118` | Implementar momentos en el subespacio (GaLore correcto) o mapear tall→SparseAdam real | Costo alto, beneficio incierto | ★★☆ | ●●●●● |
| **P12** | **Micro-batching: acumular gradientes 2-4 chunks** | `train_step.rs:1013-1035` | Reducir varianza del gradiente antes del paso del optimizer | Memoria extra; más lento por paso | ★★☆ | ●●● |
| **P13** | **Entropy regularizer / piso de confianza** | `train_step.rs:687-700` | Añadir `−λ·H(p)` para empujar distribuciones no colapsadas (colapso de logits ya observado en step 3403) | Interfiere con CE; λ pequeño (0.01) | ★☆☆ | ●● |
| **P14** | **Self-play ON** (`MUD_SELF_PLAY=1`) | `session.rs:1644`, `self_play.rs` | Curriculum sintético ya implementado (22 tests); activarlo como regularizador en la fase final de cada ciclo SGDR | Más wall-clock | ★★☆ | ● |

---

## 3. Estado del scheduler SGDR (verificado 2026-08-03)

`apply_lr_schedule` (`train_step.rs:52-105+`):
- Warmup lineal inicial ✓
- Ciclos SGDR con mini-warmup del 20% del ciclo que parte del **50% del base** (no de 0) ✓ — la LR sube al pico en cada restart
- `MUD_LR_CYCLES` clamp [3, 16], default `√total` ✓
- El fix GRASSHOPPER (LR coincidente en step 2 y 1059) está resuelto: los restarts evitan el valle largo de LR insuficiente ✓

El schedule es correcto. El problema de pérdida de loss NO está en el scheduler sino en B-1 (acumulación de grad de output_norm) + B-3 (cabeza congelada) + B-4 (pocas capas entrenables).

---

## 4. Orden de ejecución sugerido (máximo ROI)

1. **P3 + P5 + P10** (fixes, ~1 día) → estabilidad correcta de la base; re-run loss_cert.
2. **P1 + P2** (envs existentes, ~1 día) → el salto grande de loss: cabeza + stack completo entrenable.
3. **P4 + P6** (2-3 días) → señal de gradiente de mayor calidad.
4. **P8/P9/P12** como experimentos A/B con `loss_cert` (modo Percentile) como gate objetivo.

**Gates de verificación:** `cargo test --lib loss_cert` + run corto de 100 chunks comparando la curva de loss contra el log actual (5.0→2.86).
