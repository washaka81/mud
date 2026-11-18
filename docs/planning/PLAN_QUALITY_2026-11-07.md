# Plan de calidad — 2026-11-07

**Objetivo:** pasar de **0 top-1** a **top-1 > 0 verificable** en un set
held-out, y reducir PPL/NLL de forma reproducible, sin romper los gates de
corrección/seguridad.

**Base:** [`docs/research/MUD_QUALITY_FORMULAS_2026-11-07.md`](../research/MUD_QUALITY_FORMULAS_2026-11-07.md)
· auditoría [`MUD_SESSION_REPORT_2026-11-07.md`](../sessions/MUD_SESSION_REPORT_2026-11-07.md)
· próxima sesión [`PLAN_NEXT_SESSION_2026-11-07.md`](PLAN_NEXT_SESSION_2026-11-07.md)
· backlog `R-##` [`MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md).

**Regla:** ningún ítem se marca hecho sin `(comando, salida, fecha)`. El MSE de
pesos no es criterio de calidad; el criterio es **PPL/top-1/ECE en held-out**.

---

## 0. Diagnóstico de partida (medido)

| Hecho | Evidencia |
|---|---|
| `ale.mud`: 0/8 top-1, PPL 167,471.9, H=16.914 (casi uniforme) | `next_token_eval` 2026-11-07 |
| `qwen_coder_0.5b_trained`: 0/2 top-1, PPL 17,788, H=7.084 (confiado/erróneo) | idem |
| Escala ternaria actual \(s=\gamma/\sqrt2\) es 1.889× menor que el óptimo LSQ en `token_embd` | `quant_formula_audit` → ratio 1.889, NMSE 0.391 vs 0.218 |
| BitNet exacto (\(s=\gamma,\delta=0.5\gamma\)) baja NMSE ~32 % en Gaussiana | cálculo numérico §2.5 |
| Escalar `prq_scale` post-hoc empeora PPL (ale 167k→244k; qwen 17.8k→20.7k) | `scale_rescale` + `next_token_eval` |
| `ale.mud` es forge 6L/128h; no hay teacher F32 equivalente | metadatos |

**Conclusión:** hay dos problemas distintos:
- `ale`: capacidad/datos/entrenamiento insuficientes.
- `qwen`: cuantización/ranking/calibración; la escala y la recuperación son
  atacables.

---

## 1. Fase Q0 — Verdad de medición (1–2 días)

### Q0.1 Evaluador held-out + calibración

- `next_token_eval` ya tiene `--temperature-scan` (\(T^*\) por rejilla 0.10–2.00)
  y ECE de 10 bins. **Hecho 2026-11-07.**
- Pendiente: `--save-logits <path.jsonl>` y Brier score.
- Añadir un set held-out de **20–50 prompts** variados (factual, copy, código,
  español/inglés) en `tests/data/heldout_quality.txt`, separado del set de
  entrenamiento.
- **Aceptación:** `./mud.sh predict-eval <model> tests/data/heldout_quality.txt --max 32 --temperature-scan`
  imprime \(T^*\), ECE y PPL sin panic; el set no se usa en QAT.

### Q0.2 Auditoría de kurtosis de activaciones

- Tool `activation_kurtosis` (o extender `next_token_eval --trace`): media de
  \(\mathrm{Kurt}(x)\) por capa y por proyección (Q/K/V/O/up/down).
- **Aceptación:** tabla por capa; identificar si los outliers están en
  activaciones (justifica Hadamard) o solo en pesos.

### Q0.3 Reproducibilidad

- `./mud.sh quant-audit models/qwen_coder_0.5b_trained.mud --max-rows 256 --json`
  debe reproducir ratio \(1.889 \pm 0.01\).
- **Aceptación:** JSON estable entre ejecuciones; documentado en §2.4 del
  research doc.

---

## 2. Fase Q1 — Escala y deadzone en conversión (3–5 días)

> **No** tocar `prq_scale` de modelos ya entrenados: medido que empeora PPL.

### Q1.1 Selector de fórmula de escala

- Añadir `MUD_QUANT_SCALE=current|bitnet|lsq` a:
  - `tools/universal_converter/quantizer.rs`,
  - `tools/mud_forge.rs` (para nuevos modelos),
  - `src/mud/checkpoint.rs` (repack de entrenamiento).
- Semántica:
  - `current`: \(s=\gamma/\sqrt2\), \(\delta=0.7s\) (compatibilidad).
  - `bitnet`: \(s=\gamma\), \(\delta=0.5\gamma\).
  - `lsq`: códigos con \(\delta=0.7s_c\), luego \(s^*=\sum TW/\sum|T|\).
- Test unitario por fórmula con vector conocido y comparación MSE.
- **Aceptación:** `cargo test --lib --release quant_formula` verde; el default
  se decide con Q1.3, no por opinión.

### Q1.2 Deadzone configurable/annealing

- `MUD_QUANT_DEADZONE` (0.5…0.9) y `MUD_QUANT_DEADZONE_ANNEAL` (p. ej.
  `0.5:0.7` durante QAT).
- **Aceptación:** el pack respeta el valor; test de frontera; no cambia el
  default sin A/B.

### Q1.3 A/B real de conversión

- Requiere un **F32/BF16 de referencia** (Qwen2.5-0.5B/SmolLM2-135M
  safetensors; descargar o usar el que exista en `models/`).
- Convertir 3 variantes (`current`, `bitnet`, `lsq`) con la misma semilla y
  evaluar en held-out:
  - PPL, top-1, top-5, ECE.
- **Criterio de adopción:** `bitnet` o `lsq` con PPL ≥10 % menor o NMSE ≥25 %
  menor **y** top-1 no peor; si no, mantener `current` y documentar.
- **Comando:**
  ```bash
  ./mud.sh convert <source.safetensors> /tmp/q_bitnet.mud   # con MUD_QUANT_SCALE=bitnet
  ./mud.sh predict-eval /tmp/q_bitnet.mud tests/data/heldout_quality.txt --max 32
  ```

### Q1.4 Re-entrenamiento corto post-conversión

- Si la conversión nueva baja MSE pero no PPL, hacer QAT corto (64–256 pasos,
  LR 0.03–0.1, deadzone annealing) sobre la variante ganadora.
- **Aceptación:** PPL baja monótonamente en ≥40 chunks; `[DW] session` > 0.

---

## 3. Fase Q2 — Embeddings y cabeza (R-53) (2–4 días)

### Q2.1 Forge con emb/head F32

- `mud_forge` ya tiene `--f32-emb`; usarlo en los perfiles nuevos y comparar
  `ale` ternary vs `ale` F32-emb con el mismo corpus/semilla.
- **Aceptación:** `converter_auditor`/`diagnose_model` confirman
  `token_embd`/`output` Float32; tabla PPL/top-1 de la ablación.

### Q2.2 Modelo real

- Convertir Qwen/SmolLM con emb/head F32 (default actual) y variante 8-bit
  (futuro) para medir el delta.
- **Aceptación:** el modelo F32-emb no pierde top-1 frente al ternario puro;
  si no hay diferencia, documentar y no invertir más.

---

## 4. Fase Q3 — QAT largo + destilación (1–2 semanas, CPU)

### Q3.1 QAT largo de Qwen 0.5B

- Setup: `MUD_QAT_LR=0.05–0.1`, `MUD_TRAIN_GRAD_TOKEN_SCALE=0`,
  `MUD_TRAIN_LAST_N_LAYERS=all`, `MUD_TRAIN_FREEZE_EMB=0`,
  `MUD_TRAIN_DEADZONE_ANNEAL=0.5:0.7`, `MUD_TRAIN_SEQ_LEN=32`,
  checkpoints cada 250–500 pasos.
- 2k–10k pasos; evaluar held-out cada checkpoint.
- **Aceptación:** top-1 > 0 en ≥1 prompt; PPL decreciente; `.weight` ternarios
  movidos en cada checkpoint (`[DW] session`).

### Q3.2 Destilación mínima (BitDistill-lite)

- Teacher: Qwen F32 (`qwen_coder_0.5b.mud`), student: ternario.
- Pérdida:
  \[
  \mathcal L=\mathcal L_{CE}+\lambda\mathcal L_{LD}
  +\gamma\mathcal L_{AD},
  \]
  con \(\tau=2..5\), \(\lambda=0.5..2\), \(\gamma=0\) primero (solo logit KD),
  luego \(\gamma>0\) (atención, 1 capa).
- Corpus: `training/corpus/project_corpus.txt` + `tests/data/heldout_quality.txt`
  **excluido** del entrenamiento.
- **Aceptación:** PPL del student baja ≥20 % frente a QAT sin KD a igual
  número de pasos; top-1 no empeora.

### Q3.3 TSLD (token-scaled)

- Si la KD media no basta, ponderar por token (TSLD) y comparar.
- **Aceptación:** NLL por token difícil baja sin colapso de entropía.

---

## 5. Fase Q4 — Rotaciones / outliers (R-52) (3–5 días)

### Q4.1 FWHT en conversión

- Implementar Hadamard normalizada \(H\) por bloque de 64/128 columnas:
  \[
  W'=H W,\qquad X'=X H^\top,
  \]
  y ternarizar \(W'\). Mantener \(H\) para inferencia (o plegarla en la
  siguiente capa si es posible).
- **Aceptación:** kurtosis de pesos/activaciones baja; PPL no empeora; test
  de ortogonalidad \(H^\top H=I\).

### Q4.2 Medición de outliers

- Comparar kurtosis antes/después y PPL de la variante rotada vs no rotada.
- **Criterio:** adoptar solo si PPL/top-1 mejora de forma reproducible.

---

## 6. Fase Q5 — PTQ avanzado (opcional, 1–2 semanas)

### Q5.1 CAT-Q-lite

- Implementar las 3 componentes: modulación aprendible
  \((\delta_\mu,\delta_\alpha,\delta_\Delta)\), transición
  \(\tanh\) con schedule \(s_0=30,\gamma=0.8\), y reconstrucción por ventana.
- 512 muestras de calibración; 60 épocas; sin gradientes de pesos completos.
- **Aceptación:** PPL < baseline ternario actual; top-1 > 0 en held-out.

### Q5.2 PT²-LLM ITF/AGA/SSR

- Implementar ATQ (Eq. 9/10), AGA (Eq. 13) y SSR (Eq. 15/16) del research doc.
- **Aceptación:** tabla PPL WikiText2-like vs baseline; si no hay corpus
  disponible, usar held-out propio y documentar la diferencia.

---

## 7. Fase Q6 — Calibración y limpieza (2–3 días)

- `temperature-scan` + ECE en `next_token_eval` (**hecho**).
- Reportar \(T^*\) y ECE en el informe de cada checkpoint.
- **Aceptación:** Qwen ya medido: \(T^*=1.4\), PPL 17,788→9,055, ECE
  0.264→0.046, top-1 sigue 0/2. Documentar que la temperatura no arregla
  ranking; el siguiente paso es KD/QAT.

---

## 8. Matriz de dependencias y esfuerzo

| ID | Tarea | Depende de | Esfuerzo | Impacto |
|---|---|---|---:|---|
| Q0.1 | Eval held-out + T*/ECE | — | 1–2 d | Verdad |
| Q0.2 | Kurtosis activaciones | — | 1 d | Decide R-52 |
| Q1.1 | `MUD_QUANT_SCALE` | Q0.1 | 1–2 d | Alto |
| Q1.2 | Deadzone annealing | Q1.1 | 1 d | Medio |
| Q1.3 | A/B conversión | Q1.1 + source | 2 d | Alto |
| Q1.4 | QAT corto | Q1.3 | 2–3 d | Alto |
| Q2.1 | Forge F32 emb/head | — | 1 d | Medio |
| Q3.1 | QAT largo Qwen | Q1.3/Q1.4 | 5–10 d | **Máximo** |
| Q3.2 | KD-lite | Q3.1 | 3–5 d | Alto |
| Q4.1 | FWHT | Q0.2 | 3–5 d | Medio |
| Q5.1 | CAT-Q-lite | Q1.3 | 5–10 d | Alto (si no hay QAT) |
| Q5.2 | PT²-LLM | Q5.1 | 5–10 d | Medio |
| Q6 | Calibración | Q0.1 | 1 d | Bajo |

---

## 9. Criterios de “hecho” de la fase

1. `./mud.sh quant-audit` reproduce el sesgo de escala \(1.889\pm0.01\).
2. La variante de conversión elegida mejora PPL ≥10 % o NMSE ≥25 % en
   held-out, sin bajar top-1.
3. `qwen_coder_0.5b` QAT/KD alcanza **top-1 > 0** en ≥1 prompt held-out.
4. `ale` con emb/head F32 y/o más datos mejora PPL de forma monótona.
5. Cada checkpoint reporta `[DW] session`, PPL, top-1, ECE y \(T^*\).
6. Gates verdes: `cargo test --lib --release`, proptest, clippy 0,
   `./mud.sh env-audit`, `./mud.sh tools-audit`, `./mud.sh unsafe-audit`,
   `./mud.sh ci`.

---

## 10. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| No hay F32 source para re-convertir | Descargar Qwen/SmolLM safetensors; o usar `mud_forge` con semilla fija para la ablación |
| El corpus quick (577 tokens) sobreajusta | Ampliar corpus y separar held-out; no entrenar con el set de eval |
| Cambiar escala rompe calibración | Re-entrenar/QAT después de convertir (medido: post-hoc empeora) |
| MSE baja pero PPL sube | Criterio PPL/top-1; no adoptar por MSE |
| CPU-only hace QAT caro | Empezar por KD-lite y CAT-Q-lite; checkpoints cada 250 pasos |
| Hadamard no ayuda en este modelo | Medir kurtosis primero; adoptar solo con PPL reproducible |
