# MUD — Protocolo de Trazabilidad de Predicción de Token (R-TP)

**Versión:** 1.0 — 2026-11-02  
**Estado:** especificación operativa.  
**Ámbito:** inferencia (`next_token_eval`, `inference_coherence`, REPL) y QAT (trainer).  
**Regla base:** ninguna afirmación de “predice el token correcto” es válida sin la cadena de cálculos y sus valores.

---

## 0. Principios

1. **Un token, una traza.** Cada predicción debe poder auditarse desde el token ID de entrada hasta el `rank` del token esperado.
2. **Cálculo antes que conclusión.** Primero se reportan logits/probabilidades/rank; después la interpretación.
3. **Fallo explícito.** Un NaN/Inf, una suma de probabilidades fuera de tolerancia o un rank inconsistente invalidan el prompt; no se promedian como si fueran válidos.
4. **Reproducible.** La traza incluye comando, fecha, hash del modelo, hash del tokenizer y flags de ejecución.
5. **Sin coste en producción.** La traza es opt-in (`--trace` / `MUD_TRACE_TOKEN=1`); apagada no debe añadir asignaciones al hot loop (P-01).

---

## 1. Reglas obligatorias (R-TP)

| ID | Regla | Evidencia mínima | Acción si falla |
|---|---|---|---|
| **R-TP-01** | Identificar modelo y tokenizer | `model_path`, SHA-256, `hidden_size`, `num_layers`, `n_heads`, `n_kv_heads`, `head_dim`, `vocab_size`, `mud:weight_type`, hash de `tokenizer.tokens/merges` | Abortar la traza |
| **R-TP-02** | Tokenización explícita | IDs de contexto, IDs decodificados, `target_id`, `target_text`, longitud de contexto, BOS/EOS | Abortar el prompt |
| **R-TP-03** | Embedding de entrada | fila usada, `L2`, `RMS`, `min/max`, finitos | Marcar prompt inválido |
| **R-TP-04** | Traza por capa | índice, `RMSNorm` in/out, `Q/K/V` norms, `attn` entropy/top-1, `residual` norm, `FFN` out norm | Marcar capa inválida |
| **R-TP-05** | RMSNorm verificable | `eps`, `sum_sq`, `rms_inv`, `w` min/max/mean, salida finita | Fallar la capa |
| **R-TP-06** | Proyecciones Q/K/V | `n_heads`, `n_kv_heads`, `head_dim`, `q_norm/k_norm/v_norm`, finitos | Fallar la capa |
| **R-TP-07** | RoPE verificable | `rope_theta`, `pos`, `cos[0..2]`, `sin[0..2]`, finitos | Fallar la capa |
| **R-TP-08** | Atención verificable | matriz/vector de scores, `softmax_sum`, entropía, top-k, mapeo GQA `kv_h = h / (n_heads/n_kv_heads)`, slots KV | Fallar la capa |
| **R-TP-09** | Salida de atención | `o_act` norm, `O` projection norm, finitos | Fallar la capa |
| **R-TP-10** | Residual/mHC | `alpha`, `beta`, `gate`, `radius`, norma pre/post, flag `projected` | Fallar la capa |
| **R-TP-11** | FFN verificable | `gate/up/down` norms, `silu(gate)` stats, `mid` norm, residual post-FFN | Fallar la capa |
| **R-TP-12** | Output norm | `output_norm` RMS, `w` stats, salida finita | Abortar el prompt |
| **R-TP-13** | Logits completos | `min/max/mean/std`, `finite_count`, `top-k` con id/texto/logit/probabilidad | Abortar el prompt |
| **R-TP-14** | Evaluación del token esperado | `rank`, `logit_target`, `p_target`, `NLL`, `margin` | Reportar fallo |
| **R-TP-15** | Agregados | top-1/5/10, MRR, mean NLL, PPL, mean p, mean rank, mean entropy | Reportar agregado |
| **R-TP-16** | Determinismo | `MUD_PCORE_THREADS`, `MUD_GPU_GEMV`, `MUD_USE_VULKAN`; repetir con 1/8/16 hilos y comparar logits | No certificar |
| **R-TP-17** | Paridad ASM | GEMV ASM vs escalar: `cosine`, `max_abs`, `rel_rms` | No certificar |
| **R-TP-18** | Invariantes numéricos | `Σp = 1 ± 1e-5`; `NLL = -ln p`; `PPL = exp(mean NLL)`; `rank` recalculado; top-k ordenado | Invalidar prompt |
| **R-TP-19** | Fail-fast | Cualquier no finito se marca; no se promedia silenciosamente | Invalidar prompt |
| **R-TP-20** | Traza de entrenamiento | `k_w_grad` no nulo; hash de `.weight` ternario antes/después; ΔW contra checkpoint, no contra sync | No certificar QAT |
| **R-TP-21** | Unidades y precisión | logits f32 6 decimales; probabilidades científicas; NLL nats; entropía bits; PPL adimensional | Rechazar traza |
| **R-TP-22** | Formato | JSONL: un objeto por prompt + un objeto `aggregate`; campos estables | Rechazar traza |
| **R-TP-23** | Reproducibilidad | comando, fecha, commit git, hash del modelo, env relevante | Rechazar traza |
| **R-TP-24** | Certificación | top-1 > 0 solo si R-TP-01…23 pasan; si no, indicar la primera etapa que falla | No certificar |

---

## 2. Fórmulas de referencia

### 2.1 RMSNorm

```text
mean_sq = (1/H) · Σ x_i²
rms_inv = 1 / sqrt(mean_sq + eps)
y_i = x_i · rms_inv · w_i
```

### 2.2 RoPE (half-split, Qwen/Llama)

```text
θ_i = pos · rope_theta^(-2i/d),  i = 0..d/2-1
x0' = x0·cos(θ) - x1·sin(θ)
x1' = x1·cos(θ) + x0·sin(θ)
```

### 2.3 Atención

```text
s_ij = (q_i · k_j) / sqrt(d_head)
α_ij = softmax_j(s_ij)
o_i = Σ_j α_ij · v_j
kv_h = h / (n_heads / n_kv_heads)   # GQA
```

### 2.4 Logits y evaluación

```text
logsumexp = max(z) + ln( Σ exp(z_i - max(z)) )
p_i = exp(z_i - logsumexp)
NLL = -(z_target - logsumexp) = -ln p_target
PPL = exp(mean(NLL))
rank = 1 + #{i : z_i > z_target}
margin = z_target - max_{i≠target}(z_i)
H = -Σ p_i · log2(p_i)
MRR = mean(1/rank)
```

### 2.5 mHC

```text
h' = α·gate·h + (1-gate)·β·f(h)
if ||h'|| > radius: h' = h' · (radius / ||h'||)
```

---

## 3. Formato de salida (JSONL)

### 3.1 Objeto por prompt

```json
{
  "type": "prompt",
  "prompt_id": 1,
  "model": {"path": "models/ale.mud", "sha256": "…", "weight_type": "ternary2bit"},
  "tokenizer": {"sha256": "…", "vocab_size": 151643},
  "context": {"ids": [59604, 12095], "text": "Paris Paris", "len": 2},
  "target": {"id": 12095, "text": " Paris"},
  "embedding": {"row": 12095, "l2": 0.4361, "rms": 0.0146, "finite": true},
  "layers": [
    {
      "i": 0,
      "rms_in": 1.0,
      "rms_out": 1.0,
      "q_norm": 0.0,
      "k_norm": 0.0,
      "v_norm": 0.0,
      "attn": {"entropy_bits": 0.0, "top1": 12095, "softmax_sum": 1.0},
      "residual_norm": 0.0,
      "ffn_out_norm": 0.0,
      "finite": true
    }
  ],
  "output_norm": {"rms": 1.0, "finite": true},
  "logits": {
    "min": -5.0, "max": 13.02, "mean": -0.8, "std": 0.9,
    "finite_count": 151643, "count": 151643
  },
  "topk": [
    {"id": 12095, "text": " Paris", "logit": 13.023862, "p": 0.208},
    {"id": 59604, "text": "Paris", "logit": 9.581469, "p": 0.0002}
  ],
  "target_eval": {"rank": 1, "logit": 13.023862, "p": 0.208, "nll": 1.5702, "margin": 3.4424},
  "gates": {"r_tp_18": true, "r_tp_19": true, "r_tp_16": true}
}
```

### 3.2 Objeto agregado

```json
{
  "type": "aggregate",
  "n": 6,
  "top1": 33.333333,
  "top5": 33.333333,
  "top10": 33.333333,
  "mrr": 0.333375,
  "mean_nll": 11.075988,
  "perplexity": 64601.210012,
  "mean_p": 4.368504e-5,
  "mean_rank": 46305.0,
  "median_rank": 17288.0,
  "mean_entropy_bits": 17.126864,
  "mean_margin": -1.128378,
  "invalid_prompts": 0,
  "command": "next_token_eval models/ale_qat_strong.mud /tmp/copy_eval.txt --trace",
  "date": "2026-11-02",
  "git_commit": "…",
  "env": {"MUD_PCORE_THREADS": "8", "MUD_GPU_GEMV": "0", "MUD_USE_VULKAN": "0"}
}
```

---

## 4. Reglas de reporte para el agente

Cuando el agente reporte una predicción, debe incluir:

1. **Comando exacto** y salida resumida.
2. **Tabla de etapas** con al menos: embedding, RMSNorm, Q/K/V, atención (entropía/top-1), residual/mHC, FFN, output norm, logits, top-k, rank del target.
3. **Valores matemáticos**: `p_target`, `NLL`, `rank`, `margin`, `top-1/5/10`, `PPL`.
4. **Estado de gates** R-TP-16/17/18/19.
5. Si falla una etapa: decir **cuál** y el valor que rompe el invariante.
6. No decir “predice” si `rank > 1`; decir “no predice” o “acierta en N/M”.

Plantilla mínima:

```text
Modelo: <path> (<sha256>)
Prompt: <texto> → target <id> "<texto>"
Embedding: row=<id> rms=<v>
Capa 0: rms_in=<v> q=<v> k=<v> v=<v> attn_H=<bits> top1=<id>
...
Logits: min=<v> max=<v> mean=<v> std=<v> finite=<n>/<N>
Top-k: <id>(<texto>)=<logit> p=<p> | ...
Target: rank=<r> p=<p> nll=<nats> margin=<v>
Agregado: top1=<x/N> MRR=<v> PPL=<v> entropía=<bits>
Gates: R-TP-16=<ok> R-TP-17=<ok> R-TP-18=<ok> R-TP-19=<ok>
```

---

## 5. Implementación prevista

| Fase | Cambio | Criterio |
|---|---|---|
| 1 | `next_token_eval --trace <path.jsonl>` | Emite R-TP-01…15, 18, 21, 22 |
| 2 | `MUD_TRACE_TOKEN=1` en inferencia | Traza por token en REPL sin coste si está off |
| 3 | `MUD_TRACE_LAYERS=1` | Añade R-TP-04…12 por capa |
| 4 | `gemv_parity --trace` | Añade R-TP-17 al JSONL |
| 5 | `mud.sh trace-eval` | Envuelve comando + validación de gates |

**No implementar en caliente:** usar `OnceLock`/buffers preasignados y `#[cfg(feature = "trace")]` si el coste lo exige (P-01).

---

## 6. Criterios de aceptación de una traza

Una traza se considera válida solo si:

- `finite_count == vocab_size` (o se reportan los no finitos y se invalidan).
- `softmax_sum` en `[0.99999, 1.00001]`.
- `NLL` coincide con `-ln(p_target)` en `±1e-5`.
- `rank` coincide con el recuento directo de logits mayores.
- `top-k` está ordenado descendente y sin IDs duplicados.
- `PPL == exp(mean_nll)` en `±1e-4` relativo.
- Repetición con 1/8/16 hilos da los mismos `logits` (bit a bit o ±1e-6).
- `gemv_parity` da `cosine ≥ 0.999999` y `max_abs ≤ 1e-5`.

Si cualquiera falla, la predicción **no se certifica**, aunque el top-1 sea correcto por casualidad.

---

## 7. Relación con el plan siguiente

Este protocolo se implementa en P0/P1 de [`docs/planning/PLAN_NEXT_SESSION_2026-11-02.md`](../planning/PLAN_NEXT_SESSION_2026-11-02.md):

- P0: `next_token_eval --trace` + validación de gates.
- P1: QAT largo y comparación de trazas antes/después.
- P2: paridad R-50 y rotación R-52 sobre la misma traza.
