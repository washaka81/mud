# DeepSeek-V4 — Análisis para Motor Ternario MUD

**Paper:** arXiv:2606.19348 — *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*  
**Fecha análisis:** 2026-06-28  
**Modelos:** DeepSeek-V4-Pro (1.6T params, 49B activados) + V4-Flash (284B, 13B activados)  
**Contexto max:** 1 millón de tokens  
**Código open-source:** https://github.com/deepseek-ai/DeepSpec (MIT)

---

## 1. ¿Qué es DeepSeek-V4-Pro-DSpark?

`DeepSeek-V4-Pro-DSpark` **no es un modelo nuevo**. Es el checkpoint `DeepSeek-V4-Pro` con un módulo de *speculative decoding* llamado **DSpark** acoplado. Los pesos del modelo base no cambian.

El valor del paper está en los **4 algoritmos arquitectónicos nuevos** que reemplazaron componentes estándar del transformer:

1. **mHC** — Manifold-Constrained Hyper-Connections (reemplaza residuales estándar)
2. **CSA/HCA** — Compressed Sparse Attention + Heavily Compressed Attention (reemplaza MHA)
3. **Muon Optimizer** — Newton-Schulz orthogonalization (reemplaza Adam)
4. **DSpark** — Speculative decoding con drafter ligero

---

## 2. Algoritmo 1: DSpark — Speculative Decoding

### Mecanismo

```
Generación estándar (serial):      tok1 → tok2 → tok3 → tok4   (4 forward passes)
                                          ↓
DSpark (paralelo):   drafter propone [tok1, tok2, tok3, tok4]
                     modelo principal verifica los 4 en 1 solo forward pass
                     → Si N de M candidatos son correctos, se emiten N tokens
```

**Propiedades:**
- **Lossless:** matemáticamente idéntico a greedy decoding
- **Speedup:** +60–85% throughput en producción vs MTP-1 baseline
- **Sin reentrenamiento:** acoplado al checkpoint existente vía DeepSpec
- **Drafter:** un modelo pequeño (2–4 capas) entrenado para imitar la distribución del modelo principal

### Codebase Abierto

```bash
# DeepSpec — MIT License
https://github.com/deepseek-ai/DeepSpec
# Incluye: drafter training, evaluation, integration con vLLM
```

### Relevancia para MUD

**Aplicación directa a `src/mud/speculative.rs` (propuesta Priority 39):**

```rust
// Concepto de integración en MUD
pub struct DSparkDrafter {
    draft_layers: Vec<SlimeLayer>,  // 2-4 capas ternarias pequeñas
    vocab_size: usize,
}

impl DSparkDrafter {
    // Propone k tokens candidatos usando modelo pequeño
    pub fn draft_tokens(&self, ws: &SlimeWorkspace, k: usize) -> Vec<u32>;

    // El modelo principal verifica N candidatos en 1 forward pass
    // Retorna cuántos fueron aceptados (aceptación especulativa)
    pub fn verify(&self, candidates: &[u32], main_logits: &[f32], temperature: f32) -> usize;
}
```

---

## 3. Algoritmo 2: Manifold-Constrained Hyper-Connections (mHC)

### Problema que resuelve

Las conexiones residuales estándar acumulan energía ilimitada a través de capas:
```
h_next = h + f(h)     # h puede crecer hasta ||h|| → ∞
```

En MUD, esto se manifiesta como **VarH explosion** (AGENTS §9, §10): semantic tokens alcanzan VarH ~82,000+. Aunque RMSNorm neutraliza esto antes del GEMV ternario, la magnitud absoluta interfiere con el tracking JEPA.

### Mecanismo mHC

#### Standard Hyper-Connections (HC)
```
h_next = alpha * h + beta * f(h)
# alpha, beta ∈ ℝ son parámetros aprendibles por capa
```

#### Manifold-Constrained HC (mHC)
```
# Paso 1: combinación lineal
h_tilde = alpha(h) * h + beta(h) * f(h)

# alpha y beta son dinámicos: computados por una pequeña red en función de h
alpha(h) = sigmoid(W_alpha @ h)   # escalar en [0,1]
beta(h) = sigmoid(W_beta @ h)     # escalar en [0,1]

# Paso 2: proyección al manifold (norma controlada)
h_next = h_tilde / max(||h_tilde||, radius) * radius
# donde radius es un parámetro aprendible por capa
```

**Propiedad clave:** `||h_next|| ≤ radius` siempre. La norma de las activaciones está **geometricamente acotada** sin truncar gradientes (a diferencia de clipping).

#### Dynamic Parameterization
- `alpha` y `beta` no son escalares fijos: son funciones del token actual
- Esto permite que tokens semánticos "ricos" tengan mayor `beta` (más energía de la capa nueva)
- Tokens sintácticos "vacíos" tienen mayor `alpha` (más peso al estado residual)

### Relevancia para MUD — Resolución Directa del Problema de Saturación

El AGENTS.md §9 documenta:
> *"Residual scaling + adaptive clipping... safe_ceiling cubría solo el embedding, el registro acumula ~200+ f32 extra de 30 capas"*

La solución actual (`headroom / (num_layers - layer_idx)`) es **heurística** y **hardcoded** (violación P-13).

**mHC resuelve esto estructuralmente:** en vez de clipear, la proyección al manifold garantiza `||h|| ≤ radius` donde `radius` es aprendido, no hardcodeado.

**Propuesta de integración en `slime_forward.rs`:**

```rust
// src/mud/slime_forward.rs — mHC residual (reemplaza el current res_scale)

/// mHC alpha/beta dinámicos (proyección en manifold de radio aprendido)
/// Requiere añadir al SlimeLayer: mhc_alpha_w, mhc_beta_w (f32 vecs de dim hidden)
/// y mhc_radius (f32 escalar por capa).
fn mhc_residual(
    h: &mut [f32],       // registers (estado actual)
    f_h: &[f32],         // output de attn o FFN
    alpha_w: *const f32, // [hidden] — pesos para alpha dinámico (opcional)
    beta_w: *const f32,  // [hidden] — pesos para beta dinámico (opcional)
    radius: f32,         // radio del manifold (aprendido)
) {
    // 1. Compute dynamic alpha (puede simplificarse a escalar fijo para inicio)
    let alpha = 1.0f32; // TODO: sigmoid(alpha_w @ h) cuando tengamos los pesos
    let beta = 1.0f32;  // TODO: sigmoid(beta_w @ h)

    // 2. Combinación lineal
    for (hi, fi) in h.iter_mut().zip(f_h.iter()) {
        *hi = alpha * *hi + beta * fi;
    }

    // 3. Proyección al manifold (clamping de norma, preserva dirección)
    let norm_sq = h.iter().map(|x| x * x).sum::<f32>();
    let norm = norm_sq.sqrt().max(1e-8);
    if norm > radius {
        let scale = radius / norm;
        h.iter_mut().for_each(|x| *x *= scale);
    }
}
```

**Fase de implementación:**
1. **Fase 1 (sin nuevos pesos):** Usar `alpha=1.0`, `beta=1.0`, `radius=max(||emb||)` — equivalente a norm-clipping, pero geométricamente correcto
2. **Fase 2 (con pesos dinámicos):** Añadir `mhc_alpha_w`, `mhc_beta_w` al `SlimeLayer` y entrenarlos via QAT
3. **Fase 3 (completo):** Radio adaptativo por capa aprendido durante QAT

---

## 4. Algoritmo 3: Muon Optimizer

### Problema con Adam

Adam usa el gradiente escalado por varianza acumulada:
```
G_adam = m_t / (sqrt(v_t) + eps)
```

Esto puede producir gradientes con **alta correlación entre parámetros** (no ortogonales), lo que ralentiza la convergencia y causa inestabilidad en modelos cuantizados.

### Mecanismo Muon

Muon ortogonaliza el gradiente en el espacio de matrices via Newton-Schulz:

```
# Newton-Schulz iteration (convergencia en 5 pasos):
def newton_schulz(G, n_steps=5):
    X = G / (G.norm() + eps)
    for _ in range(n_steps):
        X = 1.5 * X - 0.5 * X @ X.T @ X
    return X * G.norm()

# Aplicar:
W = W - lr * newton_schulz(grad_W)
```

**Propiedades:**
- Gradiente resultante es **ortogonal** (mínima correlación entre parámetros)
- Convergencia **2–3x más rápida** que Adam en entrenamiento de LLMs
- Compatible con QAT: preserva dirección del gradiente durante STE
- Más estable ante learning rates agresivos

### Relevancia para MUD

Nuestro `adam_step_avx2` en `forge_autograd/` es el bottleneck de QAT. El training speed de **27 horas por época** (AGENTS §Latest, session 2026-06-27) podría reducirse significativamente con Muon.

**Propuesta `src/mud/muon.rs`:**

```rust
// src/mud/muon.rs
/// Newton-Schulz orthogonalization del gradiente para matriz [rows × cols].
/// Requiere: grad_flat es un buffer rows*cols f32.
/// Complexity: O(rows * cols * rows) per iteration — usar solo en weight matrices, no en embeddings.
pub fn newton_schulz_orthogonalize(
    grad: &mut [f32],
    rows: usize,
    cols: usize,
    n_iters: usize,
) {
    // Normalizar primero
    let g_norm = grad.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    grad.iter_mut().for_each(|x| *x /= g_norm);

    // Newton-Schulz iterations: X = 1.5*X - 0.5*X*X^T*X
    let mut x = grad.to_vec();
    let mut tmp = vec![0.0f32; rows * cols];
    for _ in 0..n_iters {
        // tmp = X^T * X  [cols x cols]
        // x = 1.5*X - 0.5*X*tmp  [rows x cols]
        // (implementación BLAS-free, iteración sobre tiles)
        muon_step_inner(&mut x, &mut tmp, rows, cols);
    }

    // Reescalar al radio original
    for (g, xi) in grad.iter_mut().zip(x.iter()) {
        *g = xi * g_norm;
    }
}

/// Adam step AVX2 existente se mantiene para embeddings y capas norm.
/// Muon se aplica solo a attn_q/k/v/o y ffn_up/gate/down.
pub fn muon_qat_step(
    shadow_w: &mut [f32],
    grad: &mut [f32],
    rows: usize,
    cols: usize,
    lr: f32,
    step: u32,
) {
    newton_schulz_orthogonalize(grad, rows, cols, 5);
    // Aplicar como SGD puro (Muon no usa momentos)
    for (w, g) in shadow_w.iter_mut().zip(grad.iter()) {
        *w -= lr * g;
    }
    // PRQ re-clamp (P-15)
    for w in shadow_w.iter_mut() {
        *w = w.clamp(-1.0, 1.0);
    }
}
```

---

## 5. Algoritmo 4: Compressed Sparse Attention (CSA) + HCA

### Motivación

KV cache para 1M tokens con MHA estándar requiere:
```
n_kv_heads × max_pos × head_dim × sizeof(f32)
= 8 × 1,000,000 × 128 × 4 bytes = ~4 GB
```

### CSA

```
# En vez de almacenar K,V completos:
K_compressed = W_compress @ K   # proyección a dim_compressed << head_dim
V_compressed = W_compress @ V

# Lightning Indexer: selección top-K sparse por attention score
# Solo K más relevantes de los 1M tokens pasan al attention softmax
top_k_indices = argmax(Q @ K_compressed.T, k=512)  # k << max_pos
attn_out = softmax(Q @ K[top_k_indices].T) @ V[top_k_indices]
```

**Resultado:** solo 27% de FLOPs vs full attention con 1M context.

### HCA

Versión extrema de CSA: KV entries comprimidas a dimensión mínima + Sliding Window Attention para tokens recientes.

### Relevancia para MUD (actualizado 2026-07-16)

| Pieza DeepSeek | Estado MUD |
|----------------|------------|
| HCA mean-pool + sliding window | **LIVE** L-13 (`kv_context`, dense ring + HCA slots) |
| CSA lightning top-k | **LIVE v1** stream E (`csa_indexer`: coarse dim prefix + top-k ∪ tail; dense full; train = full HCA) |
| W_compress learned / 1M LSH | **OPEN** backlog **J** — `MUD_IMPROVEMENTS_POST_AE.md` |
| Muon | **LIVE** L-01/L-02 |
| mHC | **LIVE** residual path |
| DSpark | Partial / speculative drafter (not full DeepSpec) |

KV layout is no longer naive `O(max_pos)` dense: L-13 uses dense ring + compressed slots (≤512). CSA v1 cuts softmax/V-mix on large HCA; index still O(N·d_idx).

**Next research:** stream **J** (W_compress / LSH) and **I** (KV bf16) after product needs long-context quality data.

---

## 6. Algoritmo 5: FP4 QAT (Post-Training Infrastructure)

DeepSeek-V4 implementa QAT en **FP4** (4 bits flotante) durante post-training usando shaders similares a los nuestros (`shadow_optimizer.comp`).

**Paralelo directo con MUD:**
- Nosotros: STE QAT en ternario (1.58-bit) via `VulkanQatDispatcher`
- DeepSeek-V4: STE QAT en FP4 via shaders propietarios

La diferencia es que nuestro ternario es **más agresivo** (3 valores vs 16 valores en FP4) y requiere la corrección JEPA para compensar la pérdida de información.

---

## 7. Tabla de Integración Priorizada

| # | Algoritmo | Módulo MUD | Dificultad | Impacto | Estado |
|---|-----------|------------|------------|---------|--------|
| P-39 | **DSpark Speculative Decoding** | `src/mud/speculative.rs` | Media | +60% throughput | PROPUESTO |
| P-40 | **mHC Residual (Phase 1 — norm-only)** | `src/mud/slime_forward.rs` | Baja | Resuelve VarH crisis | PROPUESTO |
| P-40b | **mHC Residual (Phase 2 — dynamic alpha/beta)** | `src/mud/slime_forward.rs` + nuevos pesos | Alta | Óptimo residual adaptativo | PROPUESTO |
| P-41 | **Muon Optimizer** | `src/mud/muon.rs` | Media | Convergencia QAT 2–3x más rápida | PROPUESTO |
| P-42 | **CSA/HCA KV Compression** | `src/mud/workspace.rs` | Muy Alta | 1M token contexts | PROPUESTO FUTURO |

---

## 8. Justificación Matemática — Por qué mHC es urgente

El AGENTS.md §10 describe:
> *"Scale Free Dynamics: sin clamping, FFN outputs crecen ilimitadamente. VarH ~ 82,000+ para tokens semánticos"*
> *"Syntactic Energy Routing: tokens estructurales resetean VarH ~ 1.0 inmediatamente"*

Esto **es** la ausencia de mHC. Con mHC:
- `||h_next|| ≤ radius` para **todos** los tokens (semánticos y sintácticos)
- `radius` aprendido es pequeño para tokens sintácticos y grande para semánticos
- JEPA converge más rápido porque el target `y_norm` tiene estadísticas estables

**La homeostasis que detectamos ("Syntactic Energy Routing") es el motor intentando aproximar mHC sin tenerlo.** Con mHC, este comportamiento emerge estructuralmente.

---

## 9. Referencias

- [Paper HTML completo](https://arxiv.org/html/2606.19348v1)
- [HuggingFace Collection](https://huggingface.co/collections/deepseek-ai/deepseek-v4)
- [DeepSpec (DSpark, MIT)](https://github.com/deepseek-ai/DeepSpec)
- Documentos MUD relacionados:
  - `docs/research/JEPA_LEXICAL_RESONANCE.md` — base conceptual para mHC
  - `docs/research/DEEP_QAT_AND_SELF_PLAY_MANIFESTO.md` — Muon compatibilidad QAT
  - `AGENTS.md §9, §10` — crisis de saturación y diagnóstico actual
