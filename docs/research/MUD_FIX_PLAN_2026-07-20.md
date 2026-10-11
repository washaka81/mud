# Plan de Correcciones y Mejoras — Post-Forensic (2026-07-20)

**Contexto:** forensic de `TRAIN_TELEMETRY_FORENSIC_2026-07-20.md` demostró que el checkpoint de
25 épocas es **MD5-idéntico** al modelo base (`ae15bdfe...`) → el trainer no persistió pesos.
Además el modelo base `models/smollm2.mud` ya está **vocabulary-colapsed** (logits planos,
ganador siempre token 0). Telemetría actual solo mide estabilidad de manifold, no calidad.

**Principio:** corregir el no-op de guardado PRIMERO (es un bug de infraestructura que invalida
cualquier entrenamiento), luego arreglar la escala del modelo base, y finalmente hacer la
telemetría honesta (rastrear ΔW). Nada de esto toca P-06 (clippy 0 warnings) ni los 222 tests.

**Orden de ejecución sugerido:** P0 → P1 → P2 → P3.

> **ESTADO 2026-07-20:** P0.1 ✅ (diagnóstico + `MUD_TRAIN_DEBUG_DW`), P1.1 ✅ (`scale_audit`
> confirmó cap 29 inflada 27.8×), P1.2 ✅ (reconvertido, `models/smollm2.mud` sano, colapsado en
> `.collapsed.bak`), P0.3 ✅ (`MUD_TRAIN_RESET_EPOCH` + warn epochs=0 + clamp `MUD_TRAIN_WCLAMP_K`).
> **CIRCUIT FIX (F3+):** `run_debate_session` tenía el MISMO error de escritura de aprendizaje
> (C1=sin PRQ scale, C2=no reescribe `*.prq_scale`) → corregido reusando `sync_shadow_to_mud`
> (corpus_trainer.rs). C3: `MUD_DEBATE_LEARN` default true + mud.sh. Ver forensic §8.
> **TLM ✅ (2026-07-20, follow-up):** TUI `train_telemetry` parsea por clave (`kv_f64`) +
> panel **Weight Δ**; trainer escribe `[TELEM]` a stderr Y `mud_train_metrics.log` (antes solo
> stderr → TUI vacío). Hot loops pointer-optimizados (P-00/P-01). `run_debate_session` ahora
> compara `hash_trained_weights` in/out (✓/⚠ NO-OP). Detalle en forensic §7.7.
> **Pendiente:** P3 gates automáticos (token-0 dominance, circuit rechaza base colapsado/no-op),
> **retrain de verificación** desde el base sano (`MUD_TRAIN_RESET_EPOCH=1`, MD5≠base, `conf`>~20%),
> y banner LR hardcoded (`3e-4` en corpus_trainer.rs:1703 vs `0.000500`, display-only).
> Detalle de ejecución en `TRAIN_TELEMETRY_FORENSIC_2026-07-20.md` §7-§8.

---

## P0 — Diagnosticar y corregir el no-op de guardado (BLOCKER)

**Objetivo:** que tras N épocas, `model_latest_checkpoint.mud` difiera del input.

### P0.1 — Aislar si el gradiente llega al shadow synced (H1)
- **Dónde:** `train_on_sequence` (corpus_trainer.rs:3524) → `apply_optimizer_cpu_step_and_pack`
  (4511) → `sync_shadow_to_mud` (3196) → `save_checkpoint` (3140/3162/3501).
- **Acción (diagnóstico temporal):** en `train_on_sequence`, justo antes de `save_checkpoint`,
  calcular `mean` y `Σ|Δ|` de `shadow_layers[0]` (o del primer tensor ternario) y comparar con el
  `mean` del tensor correspondiente en `mud` (vía `sync_shadow_to_mud` intermedio). Si son
  idénticos → el optimizer step no está mutando el shadow que se sincroniza.
- **Hipótesis a confirmar:** `train_on_sequence` recibe `layers: &mut [SlimeLayer]` y
  `shadow_layers: &mut [SlimeLayerShadowF32]`. El forward/backward puede operar sobre `layers`
  (unpack desde mmap) mientras el optimizer escribe a `shadow_layers`; si el weight que se
  *desempaca* para el forward no es el mismo que el shadow optimizado, el gradiente no se
  acumula de forma coherente → pesos no se mueven.

### P0.2 — Verificar que `mud.save` escribe el `owned_data` actualizado (H2)
- **Acción:** tras el primer epoch, escribir `weights/checkpoints/model_debug.mud` y dif contra
  `models/smollm2.mud`. Si idéntico → `sync_shadow_to_mud` no está pobl надando `owned_data` de
  todos los tensores modificados, o `mud.save` cae al path `mmap` (mod.rs:340) para algunos.
- **Fix si aplica:** asegurar que `sync_shadow_to_mud` setea `owned_data = Some(...)` + `data_ptr`
  para **cada** tensor entrenado (emb + todas las capas + scales), no solo algunos.

### P0.3 — Fix definitivo
- Una vez aislado (P0.1/P0.2), aplicar el parche mínimo: la identidad de los buffers
  shadow↔tensor debe coincidir en todo el ciclo
  `forward(unpack from mud) → backward → optimizer(shadow) → sync(shadow→owned) → save(owned)`.
- **Verificación:** tras 1 epoch mínimo (`MUD_TRAIN_MAX_CHUNKS=2`), el MD5 de
  `model_latest_checkpoint.mud` debe DIFERIR del base.

---

## P1 — Corregir la escala del modelo base (vocabulary collapse)

**Objetivo:** `models/smollm2.mud` genere logits con forma de distribución de lenguaje (no
planos, no colapsados a token 0).

### P1.1 — Auditar dequant de pesos ternarios
- **Dónde:** `unpack_ternary2bit_to_f32` + PRQ (`blk.N.*.weight` + `*.prq_scale`), y
  `logit_scale` en inferencia (`main.rs`).
- **Acción:** comparar la **magnitud** de los pesos ternarios desquantizados del `.mud` contra
  el BF16 source (`models/smollm2/model.safetensors`) capa por capa (p.ej. `blk.0.attn_q.weight`
  y `blk.29.attn_q.weight`). Si la RMS del `.mud` es ~100× menor que el source → la magnitud
  ternaria/PRQ está mal.
- **Hipótesis:** FIX D arregló las *normas* (BF16→F32 fiel), pero los **pesos** ternarios pueden
  haberse escrito con una escala PRQ incorrecta en la conversión original (previo a FIX D), o el
  `logit_scale` de inferencia no compensa.

### P1.2 — Reconvertir desde source si hace falta
- Si P1.1 confirma escala rota: regenerar `models/smollm2.mud` desde `models/smollm2/` con el
  converter ya corregido (FIX D + ECC fix), y validar con un reference HF load de SmolLM2-135M
  (logits de un prompt conocido) para confirmar fidelidad.
- **Gate:** el modelo reconvertido debe generar (greedy, temp=0) texto coherente de SmolLM2
  antes de cualquier re-entrenamiento.

---

## P2 — Telemetría honesta (rastrear ΔW)

**Objetivo:** que un entrenamiento no-op sea *visible* en los paneles.

### P2.1 — Añadir `Σ|ΔW|` / weight-delta a `mud_train_metrics.log`
- **Dónde:** writer en corpus_trainer.rs:3068. Añadir columna `dW` = suma de `|w_new − w_old|`
  sobre una muestra de tensores (o RMS del último paso de optimizador) por chunk.
- **Panel:** `train_telemetry.rs` debe leer la nueva columna y mostrar "Weight Δ" — si es ~0
  durante épocas, el entrenamiento es un no-op (como esta sesión).
- **Fix comentario `# cols`:** tras añadir la columna, actualizar el comentario (ya corregido una
  vez en §4 del session report; mantener alineado).

### P2.2 — Separar "manifold health" de "learning progress"
- Los paneles actuales (VarH/VarJ/σ/cog/JEPA) son estabilidad. Añadir explícitamente una métrica
  de **calidad de lenguaje** (p.ej. `conf` ya existe en el log pero no se grafica como alerta;
  si `conf < 5%` tras epoch 2 → warning de "model not learning").

---

## P3 — Mejoras de robustez (no bloqueantes)

| # | Mejora | Dónde | Por qué |
|---|--------|-------|---------|
| P3.1 | Fail-fast si `Σ|ΔW| ≈ 0` tras K épocas | `run_alignment_session` | Detecta no-ops temprano (evita 41h perdidas). |
| P3.2 | `diagnose_model` debe chequear colapso de vocabulario (token-0 dominance en logits sintéticos) | `tools/mud_full_audit.rs` | Detecta modelos base rotos antes de entrenar. |
| P3.3 | `mud.sh circuit` debe rechazar base colapsado/no-op con error claro | `corpus_trainer.rs` honors gate | El circuito F3+ no debe correr sobre un modelo que no aprende. |
| P3.4 | Logging de checksum de primer/último tensor en `save_checkpoint` (debug) | `corpus_trainer.rs:3501` | Forense futuro en segundos, no 41h. |

---

## Criterios de aceptación (Definition of Done)

1. **P0:** 1 epoch mínimo produce checkpoint con MD5 distinto al base; `Σ|ΔW| > 0` reportado.
2. **P1:** `models/smollm2.mud` (reconvertido) genera texto coherente greedy; logits no planos.
3. **P2:** `mud_train_metrics.log` incluye columna `dW`; panel la muestra; no-op es visible.
4. **P3:** `mud.sh circuit` rechaza base colapsado; `diagnose_model` detecta token-0 dominance.
5. Tras P0+P1: re-correr 25 épocas; el checkpoint debe (a) diferir del base y (b) mejorar `conf`
   por encima de ~20% en algún epoch → entonces SÍ habilitar circuito F3+.

**No iniciar el circuito RLVR (F3+) hasta P0+P1+P2 completos y verificados.**

---

*Plan derivado del forensic `TRAIN_TELEMETRY_FORENSIC_2026-07-20.md`. No se modificó código al
redactar este plan.*
