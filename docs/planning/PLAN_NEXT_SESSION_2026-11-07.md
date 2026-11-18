# Plan — próxima sesión (2026-11-07 →)

**Objetivo:** convertir el diagnóstico de fórmulas/calibración en una mejora
**medible** de PPL/top-1, y arreglar el modo determinista de inferencia.

**Contexto:**
- Informe de sesión: [`docs/sessions/MUD_SESSION_REPORT_2026-11-07.md`](../sessions/MUD_SESSION_REPORT_2026-11-07.md)
- Fórmulas/cálculos: [`docs/research/MUD_QUALITY_FORMULAS_2026-11-07.md`](../research/MUD_QUALITY_FORMULAS_2026-11-07.md)
- Plan de calidad largo: [`PLAN_QUALITY_2026-11-07.md`](PLAN_QUALITY_2026-11-07.md)
- Backlog `R-##`: [`MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md)

**Estado de partida (medido 2026-11-07):**

| Modelo | Top-1 | PPL | \(T^*\) | PPL(\(T^*\)) | ECE(\(T^*\)) |
|---|---:|---:|---:|---:|---:|
| `ale.mud` | 0/8 | 167,471.9 | 2.0 | 151,396 | 0.000027 |
| `qwen_coder_0.5b_trained` | 0/2 | 17,788.9 | 1.4 | **9,055** | **0.046** |

Hallazgo: \(s=\gamma/\sqrt2\) es 1.889× menor que el óptimo LSQ; el reescalado
post-hoc empeora PPL. La corrección va en conversión/QAT.

---

## 1. P0 — Determinismo de inferencia (0.5 día)

### T-01: `MUD_INFER_TEMPERATURE=0` debe ser greedy

- **Problema:** `src/main.rs` hace
  `.filter(|t| *t > 0.0).unwrap_or(0.8)`; `0` cae a `0.8`, no a greedy.
- **Acción:** interpretar `0.0` como greedy (argmax, sin muestreo) y documentar
  `MUD_INFER_TEMPERATURE=0|0.01|...`. Añadir test unitario de parseo
  (`test_temperature_zero_is_greedy`) y un test de que dos generaciones greedy
  con la misma prompt dan los mismos IDs.
- **Aceptación:** dos ejecuciones con `MUD_INFER_TEMPERATURE=0` producen la
  misma salida; `cargo test --lib --release temperature` verde.
- **Comando:**
  ```bash
  printf 'def fibonacci(n):\nquit\n' | MUD_INFER_TEMPERATURE=0 \
    ./target/release/forge_llm models/ale.mud
  ```

---

## 2. P0 — Set held-out y baseline honesto (1 día)

### T-02: `tests/data/heldout_quality.txt`

- 32–50 prompts variados: factual ES/EN, copy, código, matemática simple.
  **No** se usa en QAT/corpus.
- Evaluar con `--temperature-scan --json` y guardar tabla:
  - `ale.mud`, `ale_qat_512.mud`, `qwen_coder_0.5b_trained.mud`,
    `qwen_qat_strong.mud`.
- **Aceptación:** tabla con PPL, top-1/5, MRR, \(T^*\), ECE y tiempo; sin
  panic; el set queda versionado en `tests/data/`.

---

## 3. P1 — Fórmula de escala en conversión (2–3 días)

### T-03: `MUD_QUANT_SCALE=current|bitnet|lsq`

- Implementar en:
  - `tools/universal_converter/quantizer.rs`,
  - `tools/mud_forge.rs`,
  - `src/mud/checkpoint.rs` (repack).
- Fórmulas:
  - `current`: \(s=\gamma/\sqrt2\), \(\delta=0.7s\) (compatibilidad).
  - `bitnet`: \(s=\gamma\), \(\delta=0.5\gamma\).
  - `lsq`: códigos con \(\delta=0.7s_c\), luego
    \(s^*=\sum TW/\sum|T|\).
- Tests unitarios con vector conocido; comparar MSE/NMSE/SNR contra
  `quant_formula_audit`.
- **Aceptación:** `cargo test --lib --release quant_scale` verde; las 3
  variantes compilan y se pueden seleccionar por env.

### T-04: A/B real

- Si hay safetensors F32/BF16 (Qwen2.5-0.5B / SmolLM2-135M), convertir 3
  variantes y evaluar en held-out.
- Si no hay source, usar `mud_forge` con semilla fija y `--f32-emb` para una
  ablación controlada; documentar que no es un modelo real.
- **Criterio de adopción:** PPL ≥10 % menor o NMSE ≥25 % menor **y** top-1 no
  peor. Si no, mantener `current` y documentar.

---

## 4. P1 — QAT corto + deadzone (2–4 días)

### T-05: 256–512 pasos, deadzone 0.5 vs 0.7

- Modelo: `qwen_coder_0.5b_trained.mud` (o `ale` si el coste es alto).
- Setup: `MUD_QAT_LR=0.05–0.1`, `LAST_N=all`, `FREEZE_EMB=0`,
  `GRAD_TOKEN_SCALE=0`, `SEQ_LEN=16–32`, checkpoints cada 128–256 pasos.
- Evaluar held-out cada checkpoint: PPL, top-1, \(T^*\), ECE, `[DW] session`.
- **Aceptación:** PPL decreciente; top-1 > 0 en ≥1 prompt o NLL baja ≥20 %;
  `[DW] session` > 0 en cada checkpoint.

### T-06: R-53 — forge con emb/head F32

- `mud_forge --f32-emb` vs ternary, misma semilla/corpus.
- **Aceptación:** `diagnose_model` confirma Float32; tabla PPL/top-1 de la
  ablación. Si no hay diferencia, cerrar R-53 como “no aplica a forge”.

---

## 5. P2 — Documentación y gates (0.5 día)

- Actualizar:
  - `docs/research/MUD_QUALITY_FORMULAS_2026-11-07.md` con resultados T-03/T-05.
  - `docs/sessions/` con informe de la sesión.
  - `GEMINI.md` §0.3 / `AGENTS.md` con evidencia `(comando, salida, fecha)`.
- Gates obligatorios:
  ```bash
  cargo test --lib --release -- --test-threads=2
  cargo test --lib --release --features proptest -- --test-threads=2
  cargo clippy --all-targets --all-features -- -D warnings
  ./mud.sh env-audit
  ./mud.sh tools-audit
  ./mud.sh unsafe-audit
  ./mud.sh ci
  ```

---

## 6. Definición de “hecho” de la próxima sesión

1. `MUD_INFER_TEMPERATURE=0` es greedy y determinista (test).
2. Existe `tests/data/heldout_quality.txt` con baseline de 4 modelos.
3. `MUD_QUANT_SCALE` implementado con 3 variantes y tests.
4. A/B de conversión documentado; decisión `current|bitnet|lsq` tomada con
   PPL/top-1.
5. QAT corto con `[DW] session` > 0 y tabla PPL/top-1/ECE por checkpoint.
6. Gates verdes y docs sincronizados.

**Fuera de alcance:** QAT de 2k–10k pasos (Q3.1) y CAT-Q/PT²-LLM (Q5); se
planifican después de la decisión de escala.
