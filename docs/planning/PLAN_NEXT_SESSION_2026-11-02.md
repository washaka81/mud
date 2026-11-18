# Plan — Siguiente sesión (2026-11-02 →)

**Objetivo:** pasar de “0/8 en el modelo real” a **top-1 > 0 verificable** en `qwen_coder_0.5b` (o en un base de mayor precisión), y corregir la medición de ΔW.

**Contexto:** [`docs/sessions/MUD_SESSION_REPORT_2026-11-02_AUDIT_PREDICTION.md`](../sessions/MUD_SESSION_REPORT_2026-11-02_AUDIT_PREDICTION.md) · [`PLAN_AUDITORIA_Y_MEJORAS.md`](../../PLAN_AUDITORIA_Y_MEJORAS.md) · backlog `R-##` [`MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) · protocolo de trazas [`MUD_TOKEN_PREDICTION_TRACE_PROTOCOL.md`](../architecture/MUD_TOKEN_PREDICTION_TRACE_PROTOCOL.md).

---

## 0. Baseline al cerrar la sesión

| Métrica | Valor |
|---|---|
| Tests lib release | **475/0/0** |
| Proptest | **482/0/0** |
| Clippy | **0** |
| Unsafe | **536/0** |
| Env audit | **0 divergencias** |
| `mud.sh ci` | **verde** |
| `ale.mud` factual | 0/8 top-1, PPL 167,471 |
| `ale.mud` copy | **2/6 (33.33 %)** tras QAT fuerte |
| `qwen_coder_0.5b` factual | 0/8 top-1, PPL 27.2M |
| `qwen_coder_0.5b` QAT 64 pasos | 0/8 top-1, PPL 4.73M, 30 `.weight` ternarios movidos |
| GEMV ASM vs escalar | cosine 1.000000, max_abs ≤ 1.7e-6 |
| Paralelización 1/8/16 hilos | logits idénticos |
| Atención K grad | 0 → 2.7e-2 (bug corregido) |

---

## 1. P0 — Verdad de medición y cierre de sesión (1–2 días)

### 1.1 Corregir el reporte ΔW

- **Problema:** `sync_shadow_to_mud` compara contra bytes que el optimizer ya empaquetó in-place; reporta `0` aunque el entrenamiento haya movido 53/30 pesos.
- **Acción:** guardar el hash/bytes del último checkpoint (o del inicio de sesión) y calcular ΔW contra ese baseline; o instrumentar el pack por paso y acumular `bits_changed`.
- **Aceptación:** un QAT que mueva pesos ternarios reporta `ΔW > 0` en el checkpoint, y los tests existentes de `checkpoint.rs` siguen verdes.
- **Comando:** `cargo test --lib --release checkpoint -- --test-threads=1`.

### 1.2 Commit de la sesión

- Revisar `git status`; incluir código, docs, tests, tools y la purga de `forge_autograd/target`.
- No commitear modelos (`models/` está gitignored).
- **Aceptación:** `git status` limpio; `git log -1` con mensaje de sesión.

### 1.3 R-61: 11 tools huérfanos

- Auditar: `avx_math_validator`, `bitwise_quadrature_audit`, `cmud_test_alpha`, `corpus_io_bench`, `debug_tokenizer`, `ezop_bench`, `gen_fixture`, `main_arithmetic_validator`, `quadrature_bench`, `token_pointer_validator`, `zerocopy_bench`.
- Decidir por cada uno: borrar (P-08) o declarar `[[bin]]` y añadir a `mud.sh tools`.
- **Aceptación:** `tools/*.rs` sin huérfanos; `cargo build --all-targets` verde.

### 1.4 Regresiones permanentes

- Mantener `test_forward_tape_populates_qkv_for_attention_backward` (K grad).
- Añadir un test/bench de determinismo de GEMV con 1 vs 8 hilos (puede ser `#[ignore]` por coste).
- Documentar `next_token_eval` y `gemv_parity` en `mud.sh` (`predict-eval`, `gemv-parity`).

### 1.5 Implementar el protocolo de trazas R-TP

- Implementar `next_token_eval --trace <path.jsonl>` con los campos R-TP-01…15, 18, 21, 22.
- Añadir `MUD_TRACE_TOKEN=1` y `MUD_TRACE_LAYERS=1` (opt-in, P-01: sin coste apagado).
- Validar gates: `Σp = 1 ±1e-5`, `NLL = -ln p`, `rank` recalculado, top-k ordenado, `PPL = exp(mean NLL)`.
- **Aceptación:** una traza de `Paris Paris Paris` pasa todos los gates y coincide con `next_token_eval` sin traza.
- **Referencia:** [`MUD_TOKEN_PREDICTION_TRACE_PROTOCOL.md`](../architecture/MUD_TOKEN_PREDICTION_TRACE_PROTOCOL.md).

---

## 2. P1 — Predicción real (3–7 días)

### 2.1 QAT largo sobre `qwen_coder_0.5b`

- **Setup:** `MUD_QAT_LR=0.1`, `MUD_TRAIN_GRAD_TOKEN_SCALE=0`, `MUD_TRAIN_LAST_N_LAYERS=all`, `MUD_TRAIN_FREEZE_EMB=0`, `MUD_TRAIN_DEADZONE=0.5` (annealing 0.5→0.7), `MUD_TRAIN_SEQ_LEN=16/32`.
- **Duración:** 2k–10k pasos, checkpoint cada 250–500; guardar `next_token_eval` en cada checkpoint.
- **Aceptación mínima:** top-1 > 0 en un set held-out de 20–50 prompts; PPL decreciente; `.weight` ternarios cambiados en cada checkpoint.
- **Si plateau:** implementar regla de actualización de trits (flip por error de cuantización > `0.5·scale`, o actualización directa por signo del gradiente) y repetir.

### 2.2 Base de mayor precisión (plan B)

- Si el QAT largo no alcanza, conseguir un base BF16/F32 (SmolLM-135M/360M o Qwen2.5-0.5B) y convertirlo con `universal_converter` (emb/head F32 por defecto).
- Evaluar **antes de ternarizar** con `next_token_eval`: debe dar top-1 > 0 y PPL sana.
- Después aplicar ternarización + QAT y comparar.
- **Aceptación:** paridad/top-1 del F32 > 0; luego medir la pérdida por ternarización.

### 2.3 Medición

- `bench-infer` (tok/s, ms/token, RSS) y `bench-energy` (J/token) — R-71/R-72.
- Guardar resultados en `docs/audits/`.
- **Aceptación:** tabla reproducible con comando, salida y fecha (R-73).

---

## 3. P2 — Ola 4 (después de P1)

| ID | Acción | Criterio |
|---|---|---|
| R-53 | F32 emb/head en `mud_forge` o modelo convertido real | `converter_auditor` confirma Float32; ablación F32 vs ternary |
| R-50 | Gate de paridad vs GGUF/safetensors | cosine/KL con umbral documentado en CI |
| R-52 | Rotación Hadamard/FWHT + PTQ | tabla PPL a igual bit-width |
| R-51 | Unpack I2_S según spec | vector de test independiente |

---

## 4. P3 — Eficiencia e higiene

- R-42 wire format 5 trits/byte.
- R-44 fusión de pases / P+E.
- R-61 tools (si no se cierra en P0).
- Limpiar modelos temporales de QAT en `models/` (gitignored) cuando ya no se necesiten.

---

## 5. Comandos canónicos

```bash
# Gates
cargo test --lib --release -- --test-threads=2
cargo test --lib --release --features proptest -- --test-threads=2
cargo clippy --all-targets --all-features -- -D warnings
bash mud.sh unsafe-audit
bash mud.sh env-audit
bash mud.sh ci

# Predicción
./target/release/next_token_eval models/<modelo>.mud tests/data/next_token_eval.txt --json
./target/release/next_token_eval models/<modelo>.mud /tmp/copy_eval.txt --json

# Paridad GEMV
./target/release/gemv_parity models/<modelo>.mud blk.0.attn_q.weight
```

---

## 6. Riesgos

- **Coste:** QAT de 0.5B en CPU ~1 tk/s; 2k pasos pueden ser horas. Usar background + checkpoints.
- **Memoria:** 0.5B + shadow + Adam puede acercarse al límite; `MUD_TRAIN_FREEZE_EMB=1` si hace falta.
- **Sobreajuste al corpus pequeño:** el corpus quick son 577 tokens; ampliar corpus antes de conclusiones de calidad.
- **Modelo PTQ base:** si el base ternario es irrecuperable, el plan B (base F32) es obligatorio.

---

## 7. Definición de “hecho” de la próxima sesión

1. `ΔW` del checkpoint refleja el movimiento real (test).
2. `git status` limpio y commit de la sesión.
3. R-61 cerrado.
4. `qwen_coder_0.5b` QAT: **top-1 > 0** en un set held-out (o plan B F32 con top-1 > 0).
5. Tabla de PPL/top-1/tok-s/J-token con evidencia y fecha.
