# TRAINING_SPEED_ADAPTIVE.md
## Técnicas de Aceleración para Entrenamiento Ternary QAT en MUD (CPU-Only)

**Fecha:** 2026-06-28 | **Investigador:** Research Agent  
**Modelo target:** SmolLM2-135M → BitNet b1.58 Ternary | **Restricción:** CPU-only, 1 época = 20h actual  
**Goal:** Reducir a < 2 horas/época mediante técnicas adaptadas al tamaño de matriz

---

## 📊 TABLA COMPARATIVA DE TÉCNICAS

| # | Técnica | Speedup Estimado | Complejidad Rust | Compat. Ternary QAT | Fuente | Prioridad |
|---|---------|-----------------|-----------------|---------------------|--------|-----------|
| 1 | **FairyFuse Fused Ternary Kernel** | **4–30x kernel** | Alta (AVX2/AVX-512 ASM) | ✅ Nativa | arXiv:2604.20913 | 🔴 CRÍTICA |
| 2 | **Sequence Packing (No-Pad)** | **1.5–2.0x** tokens útiles/step | Baja (~50 líneas Rust) | ✅ Transparente | Unsloth, HuggingFace | 🔴 CRÍTICA |
| 3 | **Rayon Parallel Backward** | **2–4x** en CPU multi-core | Media (rayon par_chunks) | ✅ Compatible | Rust rayon docs | 🔴 CRÍTICA |
| 4 | **WSD LR Schedule** | **15–30%** menos pasos | Muy Baja (5 líneas) | ✅ Ideal para QAT | arXiv WSD paper | 🟡 ALTA |
| 5 | **Tequila Anti-Deadzone** | **Convergencia cualitativa** | Media (modificar STE grad) | ✅ Soluciona JEPA collapse | arXiv:2509.23809 | 🔴 CRÍTICA |
| 6 | **Gradient Checkpointing** | **1.15–1.25x** (mem → batch) | Media | ✅ Compatible STE | Estándar DL | 🟡 ALTA |
| 7 | **T-MAC LUT Kernel** | **4x throughput** CPU | Muy Alta (TVM codegen) | ⚠️ Indirecto | arXiv T-MAC 2025 | 🟢 MEDIA |
| 8 | **BF16 Shadow Weights** | **1.3–1.5x** mem+bw | Baja (cast f32→bf16) | ✅ Verificado BitNet | scitepress 2024 | 🟡 ALTA |
| 9 | **Muon Newton-Schulz** | **30–50%** menos pasos | Ya implementado | ✅ Ya en MUD | Kelsey McAuley 2024 | 🟡 YA ACTIVO |
| 10 | **Curriculum / Token Importance** | **18–45%** menos steps | Media | ✅ Compatible | ACL 2024 | 🟢 MEDIA |
| 11 | **Flash Attention CPU Tiling** | **30–50%** prefill largo | Alta (cache-tiled SDPA) | ✅ Con GQA | llama.cpp -fa | 🟡 ALTA |
| 12 | **CAT-Q Diferenciable** | **Convergencia cualitativa** | Alta (STE suavizado) | ✅ Para ternary | arXiv CAT-Q 2024 | 🟢 MEDIA |
| 13 | **NUMA Pinning (numactl)** | **5–10%** sistemas NUMA | Nula (env var) | ✅ Transparente | Linux numactl | 🔴 TRIVIAL |

---

## 🧩 PRINCIPIO: TÉCNICA ADAPTADA AL TAMAÑO DE MATRIZ

**REGLA FUNDAMENTAL:** No existe una única técnica de optimización óptima para todas las matrices.
El optimizador, batch size, estrategia de gradientes y kernel SIMD deben seleccionarse
**dinámicamente** en función de la forma de la matriz objetivo.

### Tabla de Decisión por Forma de Tensor (SmolLM2-135M)

| Tensor | Forma | Clasificación | Técnica Óptima |
|--------|-------|---------------|----------------|
| `attn_q.weight` | 576×576 | Cuadrada pequeña | **Muon** (Newton-Schulz, ≤5 iter) |
| `attn_k.weight` | 192×576 | Rectangular KV | **Adam** (Muon inestable con GQA) |
| `attn_v.weight` | 192×576 | Rectangular KV | **Adam** |
| `attn_output.weight` | 576×576 | Cuadrada | **Muon** |
| `expert.0.w1.weight` | 1536×576 | Tall (rows>>cols) | **GaLore** (rank=cols/4) |
| `expert.0.w3.weight` | 1536×576 | Tall (rows>>cols) | **GaLore** |
| `expert.0.w2.weight` | 576×1536 | Wide (cols>>rows) | **Chunked Adam** (512 cols) |
| `token_embd.weight` | 49152×576 | Gigante vocab | **Sparse Adam** (solo filas activas) |

### Función de Selección — Rust

```rust
pub enum OptimizerStrategy {
    Muon { ns_iters: usize },
    GaLore { rank: usize, update_freq: usize },
    ChunkedAdam { chunk_cols: usize },
    SparseAdam { only_active_rows: bool },
    Adam,
}

/// Selecciona la estrategia óptima según la forma de la matriz.
/// DEBE llamarse durante la inicialización de shadow_layers en corpus_trainer.rs
pub fn select_optimizer(rows: usize, cols: usize) -> OptimizerStrategy {
    let ratio = rows as f32 / cols as f32;
    let total = rows * cols;

    if total < 100_000 && ratio >= 0.5 && ratio <= 2.0 {
        OptimizerStrategy::Muon { ns_iters: 5 }
    } else if ratio > 2.5 {
        let rank = (cols / 4).max(8);
        OptimizerStrategy::GaLore { rank, update_freq: 100 }
    } else if ratio < 0.4 {
        OptimizerStrategy::ChunkedAdam { chunk_cols: 512 }
    } else if total > 10_000_000 {
        OptimizerStrategy::SparseAdam { only_active_rows: true }
    } else {
        OptimizerStrategy::Adam
    }
}
```

---

## 🏆 TOP 5 TÉCNICAS CON PSEUDOCÓDIGO

### 🥇 #1 — Sequence Packing Sin Padding
**Speedup: 1.5–2.0x | Complejidad: Baja | Impacto inmediato**

```rust
pub fn pack_sequences(
    token_sequences: Vec<Vec<u32>>,
    context_len: usize,
) -> Vec<(Vec<u32>, Vec<usize>)> {
    let mut bins: Vec<(Vec<u32>, Vec<usize>)> = Vec::new();
    for seq in token_sequences {
        let mut placed = false;
        for (bin_tokens, bin_bounds) in bins.iter_mut() {
            if bin_tokens.len() + seq.len() + 1 <= context_len {
                let boundary = bin_tokens.len();
                bin_tokens.extend_from_slice(&seq);
                bin_tokens.push(EOS_TOKEN_ID);
                bin_bounds.push(boundary);
                placed = true;
                break;
            }
        }
        if !placed {
            let mut new_tokens = seq.clone();
            new_tokens.push(EOS_TOKEN_ID);
            bins.push((new_tokens, vec![0]));
        }
    }
    bins
}
// Reset position_ids a 0 en cada boundary para RoPE correcto
```

### 🥈 #2 — Fused Ternary GEMV (FairyFuse arXiv:2604.20913)
**Speedup: 4–30x kernel | Complejidad: Alta | Impacto máximo**

```
// ANTES: conditional multiply (2 ops)
//   accum += act[j] * w[j]   donde w[j] ∈ {-1, 0, +1}
//
// DESPUÉS FairyFuse: conditional add/sub sin multiply
// Si w[j] == +1: accum += act[j]   (_mm256_add con mask)
// Si w[j] ==  0: no-op
// Si w[j] == -1: accum -= act[j]   (_mm256_sub con mask)
//
// AVX2:
// mask_pos = _mm256_cmpeq_epi8(w_byte, ones)
// mask_neg = _mm256_cmpeq_epi8(w_byte, neg_ones)
// accum += _mm256_and_si256(acts, mask_pos)
// accum -= _mm256_and_si256(acts, mask_neg)
// → 0 multiplicaciones float, 0 dequantización
```

### 🥉 #3 — Rayon Parallel Backward
**Speedup: 2–4x | Complejidad: Media | Impacto inmediato**

```rust
use rayon::prelude::*;
// Token-level: cada token forward+backward independientemente
let local_grads: Vec<LayerGrads> = tokens
    .par_iter()
    .map(|&tok| { forward_and_backward_single(tok, layers) })
    .collect();
let total_grads = local_grads.into_iter()
    .fold(LayerGrads::zero(), |a, g| a.add(g));
apply_muon_step(shadow_weights, &total_grads, lr);
```

### 🏅 #4 — Tequila Anti-Deadzone STE (arXiv:2509.23809)
**Speedup: Cualitativo — resuelve VarH=0 / JEPA collapse | Complejidad: Media**

```rust
fn ternary_ste_grad_tequila(shadow_weight: f32, grad_out: f32, threshold: f32) -> f32 {
    let in_deadzone = shadow_weight.abs() < threshold;
    if in_deadzone {
        // Fuerza de escape: rompe el ciclo grad≈0 → peso no escapa
        grad_out * (1.0 + shadow_weight.abs() / threshold)
    } else {
        if shadow_weight.abs() <= 1.0 { grad_out } else { 0.0 }  // STE estándar
    }
}

fn adaptive_threshold(jepa_var_ema: f32, base: f32) -> f32 {
    if jepa_var_ema < 0.1 { base * 0.5 } else { base }  // Liberar pesos en colapso
}
```

### 🎖 #5 — WSD Learning Rate Schedule
**Speedup: 15–30% menos pasos | Complejidad: Mínima**

```rust
pub fn wsd_lr(step: usize, total: usize, lr_max: f32, lr_min: f32) -> f32 {
    let warmup = (total as f32 * 0.05) as usize;
    let decay  = (total as f32 * 0.90) as usize;
    if step < warmup {
        lr_max * (step as f32 / warmup as f32)
    } else if step < decay {
        lr_max
    } else {
        let t = (step - decay) as f32 / (total - decay) as f32;
        lr_min + (lr_max - lr_min) * 0.5 * (1.0 + (std::f32::consts::PI * t).cos())
    }
}
```

---

## 📋 PLAN DE IMPLEMENTACIÓN EN 3 FASES

### FASE 1 — Quick Wins (1–3 días) → ~2.5–4x speedup

| Tarea | Archivo | Tiempo | Efecto |
|-------|---------|--------|--------|
| 1a. NUMA pinning | `mud.sh` | 30 min | +10% |
| 1b. WSD LR Schedule | `corpus_trainer.rs` | 2h | +20% pasos |
| 1c. Sequence Packing | `corpus_trainer.rs` | 1 día | +60–100% |
| 1d. Tequila STE | `slime_backward.rs` | 3h | Fix JEPA collapse |

### FASE 2 — Medium Impact (1–2 semanas) → ~2–3x adicional

| Tarea | Archivo | Tiempo | Efecto |
|-------|---------|--------|--------|
| 2a. Rayon parallel backward | `slime_backward.rs` | 3 días | +100–300% |
| 2b. Gradient checkpointing | `slime_forward.rs` | 2 días | batch×4 |
| 2c. BF16 shadow (FFN only) | `corpus_trainer.rs` | 1 día | +30–50% mem |
| 2d. Flash Attention CPU tiling | `slime_forward.rs` | 3 días | +30–50% attn |
| 2e. select_optimizer() | `corpus_trainer.rs` | 2 días | Arquitectural |

### FASE 3 — Arquitectural (2–4 semanas) → ~2–10x adicional

| Tarea | Archivo | Tiempo | Efecto |
|-------|---------|--------|--------|
| 3a. FairyFuse fused GEMV | `src/asm/` | 1 semana | +4–30x kernel |
| 3b. Curriculum Learning | `corpus_trainer.rs` | 1 semana | +18–45% steps |
| 3c. GaLore FFN tall matrices | `muon.rs` | 1 semana | +4–8× mem FFN |

---

## ⚡ ESTIMACIÓN DE SPEEDUP TOTAL

| Fase | Factor | Tiempo |
|------|--------|--------|
| Baseline | 1.0x | **20h/época** |
| + Fase 1 | ~2.5x | **~8h** |
| + Fase 2 | ~2.5x adicional | **~3.2h** |
| + Fase 3 | ~2x adicional | **~1.5h** |

**Speedup total estimado: 13–15x → de 20h a ~1.3–1.5h/época**  
Estimado conservador realista: **8–10x (≈2–3h/época)**

---

## 🔬 DIAGNÓSTICO ACTUAL (SmolLM2 post-conversión)

Los síntomas observados (σ=0.067, VarH=0.00, E_JEPA=3.91, tokens incoherentes):

1. **Deadzone trapping masivo** → VarH=0 significa pesos atrapados en zona cero sin gradiente STE → **Fix: Tequila**
2. **E_JEPA no convergiendo** → gate no se cierra porque pesos no aprenden → **Fix: Tequila + WSD**
3. **σ=0.067 muy bajo** → distribución estrecha alrededor de 0, LR insuficiente o threshold mal → **Fix: WSD warmup**
4. **ρ=0.29** → baja correlación, sin propagación coherente → consecuencia de VarH=0

**Secuencia correcta:** Tequila → WSD → más épocas con corpus mayor

---

## 📚 REFERENCIAS

1. FairyFuse — arXiv:2604.20913 (2026) — Fused ternary kernels sin float multiplications
2. Tequila — arXiv:2509.23809 (2025) — Anti-deadzone ternary quantization
3. T-MAC — ICML 2025 — LUT-based ternary, 4x CPU throughput
4. BitNet b1.58 — arXiv:2402.17764 (2024) — Ma et al., Microsoft Research
5. GaLore — arXiv:2403.03507, ICML 2024 — Gradient Low-Rank Projection
6. Muon Optimizer — Kosson et al. 2024 — Newton-Schulz orthogonalization
7. WSD Schedule — "Scaling Laws with Learning Rate Annealing" arXiv 2024
8. Sequence Packing — HuggingFace TRL docs + Amazon Science 2024
9. DSIR — NeurIPS 2023 — Data Selection via Importance Resampling
10. Flash Attention CPU — llama.cpp -fa flag, GQA support

**Repos:** github.com/microsoft/BitNet | github.com/microsoft/T-MAC | github.com/ggml-org/llama.cpp
