# Plan de Prioridades + Roadmap Extenso — Forge LLM / MUD (2026-07-20)

**Propósito:** hoja de ruta ejecutable y priorizada. Cada ítem tiene
**objetivo · por qué · acción concreta (archivo/función) · validación CORTA**
(segundos / pocos minutos — sin entrenamientos largos).

**Estado base (este día):** TUI telemetry vivo (TLM ✅), hot-loops pointer-opt (P-00/P-01 ✅),
debate writeback hash-check ✅, base reconvertido sano (scale_audit limpio). **Bloqueos abiertos:**
vocab perdió prefijo-espacio `Ġ` (palabras fusionadas) y el modelo base 135M genera
salaf (coherencia — issue de motor/quant aparte). Ver `MUD_SESSION_REPORT_2026-07-20.md` §7.

**Principio de validación:** todo ítem "DONE" requiere una check automática en
`./mud.sh ci` o `cargo test` que lo proteja de regresión. Las validaciones
largas (retrain de 25 épocas) se reemplazan por **proxies cortos** (MD5-diff +
ΔW>0 en 1–2 chunks, o `conf` en 1 chunk).

---

## 0. Matriz de prioridades

| Tier | Tema | Bloquea | Validación objetivo | Esfuerzo |
|------|-------|---------|----------------------|---------|
| **T0** | Correctitud (vocab + motor + gates) | SÍ — todo lo demás | tests unit cortos | M |
| **T1** | Entrenamiento que SÍ mueve pesos | SÍ — F3+/circuit | MD5-diff + ΔW en 1–2 chunks | S |
| **T2** | Coherencia de generación | calidad usable | greedy 20 tok con espacios + proxy repetición | S |
| **T3** | Robustez / ops / docs | no | `./mud.sh ci` + banner | XS |
| **T4** | Research F+ (backlog) | no | benchmarks opcionales | L |

---

## T0 — Correctitud (gate de lanzamiento)

### T0.1 — Vocab: restaurar prefijo-espacio `Ġ` (palabras fusionadas)
- **Por qué:** el `.mud` tiene **0** `Ġ`/`▁`; el source `models/smollm2/tokenizer.json`
  tiene 64.157. `decode` solo inserta espacios si `space_char` se detecta → con 0,
  `space_char=None` → `romance`+`inite` = `romancesinite`.
- **Acción:**
  1. Reconvertir preservando `Ġ` (el writer ya conserva UTF-8):
     `cargo run --release --bin universal_converter -- models/smollm2/ models/smollm2.mud`
     (y lo mismo para `weights/checkpoints/model_latest_checkpoint.mud` tras parar el train).
  2. **Test de regresión** (nuevo, corto) en `src/model/tokenizer_test.rs`: cargar
     `models/smollm2.mud` vía `from_mud_metadata` y `assert_eq!(tk.space_char, Some('Ġ'))`.
  3. **Test de round-trip** (corto): `decode(encode("hello world")) == "hello world"`.
- **Validación CORTA (segundos):**
  - `cargo test --lib tokenizer::` → pasa `space_char=='Ġ'` + round-trip.
  - `strings models/smollm2.mud | grep -c "Ġ"` → **> 1000** (era 0).
  - `./mud.sh chat` → "romances inite" (con espacio).

### T0.2 — Motor/quant: coherencia del 135M (root-cause de salaf)
- **Por qué:** el base (sano, no-op) genera `2034 15 life is and are…` — logits
  confiados-pero-erróneos. Template de chat应用 bien (main.rs:388). Magnitud
  ternaria ~0.4× es *esperada* (±1×`absmean·√½`), no es la causa.
- **Acción:**
  1. **Test `forward_sanity`** (corto, nuevo): cargar `models/smollm2.mud`, correr
     forward de 1 prompt de ~8 tok, assert: logits **finitos**, y el argmax **no es
     token 0 en >50% de posiciones** (no token-0 dominance), y **entropía > 0**
     (distribución no colapsada).
  2. **Proxy de referencia barata:** comparar logits del `.mud` ternario contra los del
     BF16 source **solo en la capa LM-head** (cargar `models/smollm2/model.safetensors`
     en un test `gguf`/lector BF16 mínimo → proyectar embedding→LM-head→logits en `f32`).
     Tolerancia: coseno > 0.9 en los top-k. Esto aísla "¿el motor ternario
     reproduce el BF16?" sin PyTorch.
- **Validación CORTA:**
  - `cargo test --lib forward_sanity` → sin dominance token-0.
  - `cargo test --lib lm_head_vs_bf16` → coseno > 0.9 (o documenta el delta).
  - Si falla → aislar: `unpack_ternary2bit_to_f32` / `pack_elut_prq` / RoPE / atención
    (tests unit de cada kernel ya existen; añadir assert de RMS vs BF16 por capa).

### T0.3 — Gates P3 (colapso / no-op)
- **Por qué:** el circuito F3+ no debe correr sobre base colapsado o no-op.
- **Acción:**
  1. `diagnose_model` (o `mud_full_audit`) debe chequear **token-0 dominance** en
     logits sintéticos y **PRQ-scale inflación** (reusa `scale_audit`).
  2. `run_training_circuit` rechaza base con `scale_audit` fuera de tolerancia (error claro,
     no panic en PCorePool).
- **Validación CORTA:**
  - `cargo test` con tensor colapsado inyectado → assert rechazo.
  - `./mud.sh scale-audit models/smollm2.collapsed.bak` → `VERDICT: SCALE BROKEN`.

---

## T1 — Entrenamiento que SÍ mueve pesos (prueba de no-op)

### T1.1 — Retrain de verificación (healthy base + LR alto)
- **Por qué:** confirmar que el trainer persiste cambios y mejora `conf`.
- **Acción:** desde `models/smollm2.mud` (ya `Ġ`-sano post-T0.1):
  `MUD_TRAIN_RESET_EPOCH=1 MUD_QAT_LR=0.03 MUD_TRAIN_MAX_CHUNKS=2 \
   MUD_TRAIN_STEPS_PER_CHUNK=16 ./mud.sh train models/smollm2.mud --epochs 1`
- **Validación CORTA (segundos, proxy no-op):**
  - `MUD_TRAIN_DEBUG_DW=1` → `[DW] moved>0` y MD5 de checkpoint **≠** base.
  - `conf` del último chunk **> 5%** (proxy de "aprende").
  - **Test automático** (nuevo, corto): harness que corre 1 chunk a LR alto sobre un
    modelo fixture y assert `hash_trained_weights` cambia.

### T1.2 — STE deadzone (default LR)
- **Por qué:** a LR=0.0005 (default) en base convergido, ΔW≈0 es *esperado*.
- **Acción:** documentar el umbral `s*0.7` y exponer `MUD_QAT_LR` ya existe.
  Opcional: subir el default a 0.01 para movimiento visible en bases sanas.
- **Validación CORTA:** test a LR=0.03 mueve pesos; a 0.0005 no (documentado).

---

## T2 — Coherencia de generación (calidad usable)

### T2.1 — Post-T0.1+T1.1: generación con espacios + proxy gramatical
- **Acción:** tras T0.1 y un retrain corto (T1.1), greedy gen de 20 tok sobre
  prompt fijo; assert: tokens tienen **límites de palabra** (espacios presentes) y
  **repetición < 40%** (proxy de no-bucle).
- **Validación CORTA:** `MUD_INFER_GREEDY=1 ./mud.sh chat` → texto con espacios,
  sin bucle de 1 palabra.

### T2.2 — Sampling / chat usability
- **Acción:** ajustar top-p/temp y el stop en `<|im_end|>`; verificar que el
  decode no imprima markers especiales.
- **Validación CORTA:** `./mud.sh chat` con 1 turno → respuesta acotada.

---

## T3 — Robustez / ops / docs

### T3.1 — Banner LR hardcoded (display-only) — XS
- **Acción:** `corpus_trainer.rs:1703` `lr_init=3e-4` → `qat_learning_rate()`.
- **Validación CORTA:** `cargo build` + grep del banner en un run → muestra `0.000500`.

### T3.2 — CI cubre los nuevos tests cortos
- **Acción:** `./mud.sh ci` ya corre `cargo test --lib` + clippy + health.
  Añadir `tokenizer` + `forward_sanity` + `lm_head_vs_bf16` al set.
- **Validación CORTA:** `./mud.sh ci` → verde.

### T3.3 — Sync docs
- Mantener `GEMINI.md` §0/§6 y `AGENTS.md` alineados tras cada cierre de tier.

---

## T4 — Research F+ (backlog, no bloquea)

| ID | Tema | Validación (cuando se retome) |
|----|------|-------------------------------|
| F/QKV | QKV multi-matrix one CB | **DONE (2026-07-20):** `tools/gemv_auto_bench.rs` break-even bench presente; corre `cargo run --release --bin gemv_auto_bench`. |
| G | Multi-expert STE (round-robin+hash) | **DONE:** `moe_train.rs` (`next_train_expert` round-robin + `begin_step_hash`), tests `test_round_robin_cycles`/`test_explicit_train_expert_wins`/`test_hash_route_picks_pool_member` (serializados con `EXPERT_TEST_LOCK`). |
| H | Long full-seq + residual bank | **DONE:** `grad_checkpoint.rs` (`CheckpointPolicy`, `ResidualBank` roundtrip, `recompute_from_residual_bank`, `MUD_GRAD_CKPT_RESIDUAL`); tests incl. `test_residual_bank_env_flag`. `MUD_GRAD_CKPT=1` + `_RESIDUAL=1` smoke validado. |
| I | KV f16 packs | **DONE:** `kv_dtype.rs` (`pack_f32_to_f16_bytes`/`store_row_f16` + round-trip), tests `test_f16_roundtrip_accuracy`/`test_store_load_row`/`test_resolve_aliases`. `MUD_KV_DTYPE=f16` round-trip OK. |
| J | CSA LSH prefilter | **DONE (2026-07-20):** `csa_indexer.rs` ya tenía `lsh_signature`/`hamming64` + rama LSH en `index_hca_blocks`; añadido `force_lsh: Option<bool>` (None=env) + test `test_lsh_prefilter_recall_vs_brute` (recall==1.0 vs brute). `cargo test --lib` 231 ok. |
| K | Loss cert CI gate | `./mud.sh cert-loss` + `cargo test loss_cert` |
| **DSP** | **DSpark open (MIT/DeepSpec):** drafter semi-autoregresivo + confidence head + anchor-bounded packing + hidden-state comm `O(d)` | **DONE (2026-07-20):** `speculative.rs` — `sequential_draft`+`markov_bias` (DSP-1), `schedule_draft_length` (DSP-2, +`schedule_draft_from_hidden` DSP-2/5), `anchor_boundaries`/`anchor_attention_indices` (DSP-3), `project_hidden_to_d`/`hidden_comm_bytes` (DSP-4), `spherical_norm`/`confidence_spherical` (DSP-5). 5 tests nuevos; `cargo test --lib` 230 ok, clippy `--all-targets` limpio. Trac detalle en `DEEPSEEK_DSPARK_RESEARCH_2026-07-20.md`. |

---

## Secuencia recomendada (orden de ejecución)

```
T0.1 (vocab Ġ)  → validación tokenizer_test + strings>1000
T0.2 (motor)    → forward_sanity + lm_head_vs_bf16
T0.3 (gates)    → scale_audit + reject test
────────────── GATE: correctitud OK ──────────────
T1.1 (retrain)   → MD5-diff + ΔW>0 + conf>5%
T1.2 (LR)        → test deadzone
────────────── GATE: entrena y persiste ──────────
T2.1 (gen)       → greedy 20tok con espacios + rep<40%
T2.2 (sampling)  → chat 1 turno acotado
T3.1–T3.3       → banner + ci + docs
T4 (F+)          → benchmarks bajo demanda
```

**Criterio de "GO de calidad":** T0.1–T0.3 + T1.1 en verde, y `./mud.sh chat`
produce texto con espacios y sin colapso de token-0. Sin eso, el circuito F3+ sigue
prohibido (ver forensic §7 / session report §7.2).

---

*Derivado de `MUD_SESSION_REPORT_2026-07-20.md` §7 (obs. inferencia) y
`TRAIN_TELEMETRY_FORENSIC_2026-07-20.md` §7.7–§7.8. No contradictce `GEMINI.md`.*
