# Plan — Llegar a predecir el token correcto

## Baseline (2026-10-29; corregido 2026-11-02)
- Modelo: `models/ale.mud` — **forge `mud_forge` perfil `nano`**: 6 capas, hidden 128, 2 heads, FFN 512, vocab 151643; `token_embd`/`output` en `Ternary2Bit`. No es una conversión de Qwen 0.5B (24 capas / 896 hidden).
- inference_coherence (2 prompts): success_rate 0.0%, mean_entropy 11.72 bits, coherence_healthy FAIL. Es un smoke test de 2 prompts, no una accuracy representativa.
- Objetivo: success_rate > 0 con top-1 correcto y entropía reducida; sin regresión de calidad.

## Pasos

### P1 — Paridad de conversión (R-50)
- Implementar `./mud.sh parity <ref> <mud>`: ppl/logits del .mud vs referencia sobre corpus fijo.
- Criterio: umbral documentado; falla si empeora.
- Esfuerzo: 2-3 d. Requiere modelo de referencia (BF16/GGUF) y corpus.

### P2 — Emb/head en mayor precisión (R-53)
- Estado: default (!--ternarize-emb) ya almacena emb/head como Float32 (mayor precisión).
- Acción: documentar y verificar con converter_auditor que token_embd/output son Float32 en modelos convertidos sin --ternarize-emb.
- Validación: converter_auditor sin orphans de tipo.

### P3 — Rotación + PTQ (R-52)
- Implementar opción --rotate (Hadamard/PeRQ) en universal_converter.
- Implementar PT²-LLM-style ITF+AGA para ternary grid.
- Criterio: tabla ppl por método a igual bit-width; adoptar ganador.
- Esfuerzo: 3-5 d.

### P4 — QAT corto + validación
- Entrenar con run_trainer (MUD_QAT_LR=0.03) por N chunks.
- Validar con inference_coherence: success_rate > 0, entropía < baseline.
- Gate: cert-loss monótono.

### P5 — Medición
- bench-infer (tok/s, ms/token, RSS) y bench-energy (J/token).
- Documentar números.

## Orden
P2 (verificar) → P1 (paridad) → P3 (rotación) → P4 (QAT) → P5 (medición).

## Estado
- P2: verificado — ale.mud tiene emb/head Ternary2Bit. Checkpoint post-1-epoch QAT: ale_latest.mud. inference_coherence: success_rate 0.0%, entropía 11.72 bits (sin mejora).
- P1/P3/P4/P5: pendientes.

## Resultados 2026-11-02 (next_token_eval + gemv_parity)

| Caso | top-1 | NLL | PPL | rank medio | entropía |
|---|---:|---:|---:|---:|---:|
| `ale.mud` (8 prompts) | 0/8 | 12.0286 | 167,471.9 | 75,919 | 16.914 |
| `qwen_coder_0.5b.mud` (8) | 0/8 | 17.1201 | 27,236,038 | 54,280 | 1.815 |
| `qwen_coder_0.5b` copy (6) | 0/6 | 18.5306 | 111,617,200 | 81,888 | 6.313 |
| `qwen_coder_0.5b` bypass capas (1) | **1/1** | 1.5702 | 4.808 | 1 | 13.943 |

- **GEMV ASM vs escalar**: cosine 1.000000, `max_abs ≤ 1.7e-6` en Q/K/V/O y FFN (7 tensores reales). El ASM no es la desviación.
- **Atención L0 pos0**: `scores[0]=1.0` y `o_act == v` (correcto).
- **Causa raíz**: el QAT de 3 épocas dejó **617/674 tensores idénticos** al base; solo cambiaron capas 20-23 + `output_norm`. El modelo evaluado es esencialmente el PTQ ternario sin entrenar.
- **QAT fuerte** (`ale.mud`, 64 pasos, todas las capas, LR 0.1, sin división por tokens, deadzone 0): **53 `.weight` ternarios cambiados**; copy task **2/6 top-1 (33.33 %)** vs **0/6** base; NLL 12.081→11.076. Con 512 pasos: 2/6 top-1, NLL 9.814.
- **El warning `ΔW 0/N` es sync-level**: el optimizer ya empaquetó el tensor en cada paso; el sync vuelve a empaquetar el mismo shadow y reporta 0. No significa que el entrenamiento no movió pesos. Para medirlo, comparar hashes antes/después.
- **Qwen QAT 64 pasos** (`qwen_qat_strong.mud`, LR 0.1, sin token scaling, deadzone 0.5, todas las capas): 198 tensores cambiados (30 `.weight` ternarios); factual **0/8**, copy **0/6**; PPL **27,236,038 → 4,725,659** (5.8×); entropía **1.815 → 12.395**. Mueve pesos y calibra, pero no alcanza top-1.
- **Siguiente experimento**: QAT largo (miles de pasos) sobre `qwen_coder_0.5b`, o un modelo base de mayor precisión; evaluar top-1/PPL.
EOF
echo done