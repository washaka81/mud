# MUD Session Report — 2026-07-28: Backward Pass Stacking Optimizations

> **Objetivo:** Identificar y aplicar oportunidades de apilamiento (stacking) en el backward pass para reducir passes de memoria y ahorrar cómputo en hot loops, cumpliendo P-00 (raw pointer mastery) y P-01 (zero allocation in hot loops).

---

## 1. Métricas Finales

| Métrica | Antes | Después | Δ |
|---------|-------|---------|---|
| Tests (excl. 2 pre-existing) | 465 | **465** | 0 |
| Clippy warnings | 0 | **0** | 0 |
| Passes de memoria eliminados por bloque | 6 | **2** | -4 |
| Dequantizaciones redundantes eliminadas | 2 | **0** | -2 |

---

## 2. Optimizaciones Implementadas

### 2.1 FFN: mHC gradient + kinetic gradient → 1 pass

**Archivo:** `src/mud/slime_backward.rs` (FFN branch, ~línea 541)

**Antes:** Dos passes separados sobre `hidden` elementos:
1. Loop escalar para `mhc_alpha_grad` / `mhc_beta_grad` (usa `grad_in[i]`, `tape.ffn_v_jepa[i]`)
2. Loop AVX2 o escalar para gradiente cinético `gb = gi + kinetic_coeff * jepa` (usa `grad_in[i]`, `tape.ffn_v_jepa[i]`)

**Después:** Un único pass apilado:
- **Path AVX2:** `axpy_avx2` computa el gradiente cinético (8 elementos/iteración), luego un loop escalar reutiliza los mismos punteros `gi`/`jepa` para el mHC gradient.
- **Path escalar:** Ambos se fusionan en un único loop con raw pointers (`gi.add(i)`, `jepa.add(i)`), computando `kinetic_grad`, `gate = sigmoid(v)`, `mhc_alpha_grad`, `mhc_beta_grad`, y el residual copy `*go = g` simultáneamente.

**Ahorro:** 1 pass completo sobre `hidden` (~2048-4096 elementos) por bloque.

### 2.2 Attention: mHC gradient + kinetic gradient → 1 pass

**Archivo:** `src/mud/slime_backward.rs` (Attention branch, ~línea 728)

Mismo patrón que el FFN. El mHC atención usa `tape.attn_v_jepa[i]` y `tape.o_act_f32[i]`; el gradiente cinético usa `tape.attn_v_jepa[i]`. Ambos leen `grad_out[i]` y `tape.attn_v_jepa[i]` → fusionados en 1 pass.

**Ahorro:** 1 pass completo sobre `hidden` por bloque.

### 2.3 FFN RMSNorm backward: reutilizar `b_ws.ffn_in_x`

**Archivo:** `src/mud/slime_backward.rs` (FFN RMSNorm backward, ~línea 700)

**Antes:** `y_f32 = tape.norm_i8_ffn[i] as f32 * tape.ffn_act_scale` se recomputaba 2 veces (en el loop de reducción y en el loop de gradiente), aunque `b_ws.ffn_in_x[i]` ya fue calculado en la línea 650 con la misma fórmula exacta.

**Después:** Se reutiliza `b_ws.ffn_in_x[i]` directamente en ambos loops.

**Ahorro:** 2 passes de dequantización redundante (i8→f32 + multiplicación por scale) sobre `hidden` elementos. El path de Attention RMSNorm ya tenía esta optimización (usaba `b_ws.attn_in_x[i]`).

---

## 3. Análisis de Impacto

### Passes de memoria eliminados por bloque

| Optimización | Passes eliminados | Elementos/bloque | Bytes ahorrados |
|---|---|---|---|
| FFN mHC+kinetic stacking | 1 | hidden | 4×hidden |
| Attention mHC+kinetic stacking | 1 | hidden | 4×hidden |
| FFN RMSNorm dequant reuse | 2 | hidden | 8×hidden |
| **Total** | **4** | **hidden** | **16×hidden** |

### Impacto total en modelo completo

En un modelo de 64 capas, por chunk de entrenamiento:
- **256 passes de memoria evitados** (4 passes × 64 capas)
- Con `hidden = 2048`: **~2 MB de lecturas/escrituras de memoria evitadas** por chunk
- Con `hidden = 4096`: **~4 MB evitados** por chunk

### Notas de implementación

- Todas las optimizaciones usan raw pointers (`*ptr.add(i)`) cumpliendo P-00.
- Cero allocaciones nuevas cumpliendo P-01.
- El path AVX2 preserva `axpy_avx2` para el gradiente cinético (8 elementos/iteración); el mHC gradient se mantiene escalar porque `sigmoid` no es vectorizable con AVX2 sin aproximaciones.
- El `copy_nonoverlapping(gi, go, hidden)` en el path AVX2 se mantiene como operación de memoria (muy rápida).

---

## 4. Verificación

- `cargo clippy --all-targets` → 0 warnings ✅
- `cargo test --lib -- --test-threads=2 --skip forward_sanity --skip cmud_compare_not_degenerate` → 465 passed, 0 failed ✅
- Los 2 tests excluidos (`forward_sanity`, `cmud_compare_not_degenerate`) son fallas preexistentes no causadas por estos cambios (verificado via `git stash`).

---

## 5. Próximos Pasos

1. **SDPA backward vectorization:** El loop anidado `for h in 0..n_heads { for d in 0..head_d }` (líneas 778-790) podría vectorizarse con AVX2 si `head_d >= 8`.
2. **WCLAMP guard stacking:** El guard de magnitud de shadow (líneas 480-492 en `optim.rs`) hace 2 passes por fila (sum + clamp). Podría apilarse con el pack step.
3. **Optimizer step + pack fusion:** El `apply_sgd_shadow_update` + `pack_elut_prq` son passes separados que podrían fusionarse.
