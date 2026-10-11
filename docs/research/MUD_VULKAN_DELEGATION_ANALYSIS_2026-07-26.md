# MUD Vulkan/ash Compute Delegation Analysis (2026-07-26)

> **Fecha:** 2026-07-26  
> **Hardware target:** Intel i7-1260P + Iris Xe (ADL GT2 UMA)  
> **Contexto:** Análisis profundo de funciones y variables del trainer para identificar cálculos complejos que podrían beneficiarse de delegación a Vulkan/ash, considerando la restricción de SIGSEGV en Iris Xe.

---

## 1. Variables de entorno críticas

| Variable | Default | Función | Archivo |
|----------|---------|---------|---------|
| `MUD_USE_VULKAN` | `1` | Activa/desactiva ash completamente | `ash_backend.rs:229` |
| `MUD_GPU_GEMV` | `auto` | Política GEMV: `0`/`1`/`auto` | `gemv_policy.rs:68` |
| `MUD_GPU_GEMV_MIN` | — | Override de umbral min work | `gemv_policy.rs:93` |
| `MUD_MUON_NS_ITERS` | `5` | Iteraciones Newton-Schulz | `muon.rs:38` |
| `MUD_PIPELINE_OVERLAP` | `0` | Double-buffer CPU→GPU→CPU | `session.rs` |
| `MUD_TRAIN_FREEZE_EMB` | `true` | Congela embeddings | `constants.rs:113` |
| `MUD_TRAIN_STP` | `false` | Activa STP loss | `stp_loss.rs:121` |

---

## 2. Funciones críticas del trainer (por complejidad)

| Función | Archivo:Línea | Complejidad | GPU path? |
|---------|---------------|-------------|-----------|
| `train_on_sequence` | `train_step.rs:34` | O(hidden × ffn_mid × layers × steps) | Parcial (GEMV) |
| `backward_slime_block` | `slime_backward.rs:506` | O(hidden × ffn_mid × layers) | No |
| `apply_optimizer_cpu_step_and_pack` | `optim.rs:377` | O(n_weights) | Parcial (NS) |
| `newton_schulz_orthogonalize` | `muon.rs:75` | O(n² × n_iters) | ✅ Sí (`dispatch_newton_schulz_sync`) |
| `stp_loss_and_grad` | `stp_loss.rs:56` | O(hidden × triples) | No |
| `jepa_stabilizer` | `slime_jepa.rs:173` | O(hidden × layers) | No |
| `mhc_scale_sgd_step` | `optim.rs:221` | O(hidden) | No |
| `evaluate_slime_block` | `slime_forward.rs:1007` | O(hidden × ffn_mid × layers) | Parcial (GEMV) |
| `ternary_gemv_backward` | `slime_backward.rs:341` | O(hidden × ffn_mid) | No |

---

## 3. Dispatch functions disponibles en ash_backend.rs

| Función | Archivo:Línea | Usa `submit_and_wait`? | Shader |
|---------|---------------|----------------------|--------|
| `dispatch_gemv_host_sync` | `ash_backend.rs:743` | ✅ Sí | `gemv_pipe` |
| `dispatch_gemv_host_sync_ex` | `ash_backend.rs:763` | ✅ Sí | `gemv_pipe` |
| `dispatch_gemv_qkv_host_sync` | `ash_backend.rs:855` | ✅ Sí | `qkv_pipe` (**SIGSEGV Iris Xe**) |
| `dispatch_rms_norm_sync` | `ash_backend.rs:1093` | ✅ Sí | `rms_norm_pipe` |
| `dispatch_mha_sync` | `ash_backend.rs:1155` | ✅ Sí | `mha_pipe` |
| `dispatch_newton_schulz_sync` | `ash_backend.rs:1264` | ✅ Sí | `ns_step1_pipe` / `ns_step2_pipe` |
| `dispatch_optimizer_batch_async` | `ash_backend.rs:658` | ❌ No (`frame.acquire`) | `optimizer_pipe` |

---

## 4. Análisis de viabilidad por cálculo

### 4.1 Newton-Schulz (Muon NS) — `muon.rs:75`

**Estado:** ✅ Ya tiene GPU path via `dispatch_newton_schulz_sync`

**Complejidad:** O(rows × cols² × n_iters) — para 2048×576 con 5 iteraciones ≈ 3.4B ops

**Patrón de cómputo:**
- Normalización L2 (reducción)
- Gram matrix: `tmp = X^T X` (cols×cols)
- Iteración NS: `X_{k+1} = X_k * (a * tmp + b * I)` (matmul)
- 5 iteraciones con barriers

**Viabilidad Iris Xe:** ⚠️ **Riesgo** — usa `submit_and_wait`, pero el shader NS (`ns_pipe`) es diferente al QKV (`qkv_pipe`). El SIGSEGV es específico del shader QKV GEMV.

**Recomendación:** Agregar check de Iris Xe a `try_vulkan_ns` (muon.rs:53). Probar si el shader NS funciona en Iris Xe.

### 4.2 STP Loss — `stp_loss.rs:56`

**Estado:** ❌ CPU-only

**Complejidad:** O(hidden × num_triples) = O(2048 × 58) ≈ 119K ops/step

**Patrón de cómputo:**
- Vector differences (element-wise)
- Dot products (reducción)
- Gradient computation (element-wise)
- Chain rule (element-wise)

**Viabilidad Iris Xe:** ✅ **Alta** — patrón element-wise + reducción. No requiere shader QKV GEMV.

**Recomendación:** Crear shader genérico usando `dispatch_optimizer_batch_async` (no usa `submit_and_wait`).

### 4.3 mHC Scale SGD — `optim.rs:221`

**Estado:** ❌ CPU-only

**Complejidad:** O(hidden) por step, por capa

**Patrón de cómputo:** Element-wise clamp: `updated = (w - lr * g * inv_tok).clamp(0.0, 4.0)`

**Viabilidad Iris Xe:** ✅ **Alta** — operación element-wise trivial.

**Recomendación:** Incluir en `AshOptimizerUpdate` batch. El shader `optimizer_pipe` ya existe y es seguro.

### 4.4 JEPA Stabilizer — `slime_jepa.rs:173`

**Estado:** ❌ CPU-only (AVX2 batch-8 para mean/var)

**Complejidad:** O(hidden × num_layers) = O(2048 × 12) ≈ 25K ops/step

**Patrón de cómputo:**
- Z-score normalization (reducción + element-wise)
- EMA update (element-wise)
- Gate computation sigmoid (element-wise)

**Viabilidad Iris Xe:** ✅ **Alta** — reducción + element-wise.

**Recomendación:** Crear shader genérico usando `dispatch_optimizer_batch_async`.

### 4.5 Backward Pass — `slime_backward.rs:506`

**Estado:** ❌ CPU-only

**Complejidad:** O(hidden × ffn_mid × layers) ≈ 14M ops/step

**Patrón de cómputo:**
- mHC gradient (element-wise)
- FFN down projection backward (ternary GEMV)
- FFN activation backward (element-wise)
- FFN up/gate projection backward (ternary GEMV)

**Viabilidad Iris Xe:** ⚠️ **Riesgo parcial** — usa `ternary_gemv_backward` (GEMV). El GEMV regular usa `gemv_pipe` (diferente a `qkv_pipe`).

**Recomendación:** Probar si `dispatch_gemv_host_sync_ex` funciona en Iris Xe.

---

## 5. Tabla de viabilidad resumida

| Cálculo | Complejidad | GPU path | `submit_and_wait`? | Shader ≠ QKV? | Viabilidad Iris Xe |
|---------|-------------|----------|-------------------|---------------|-------------------|
| **Newton-Schulz** | O(n²×iters) | ✅ Sí | ✅ Sí | ✅ Sí (`ns_pipe`) | ⚠️ Probar |
| **STP Loss** | O(h×triples) | ❌ No | N/A | N/A | ✅ Crear shader genérico |
| **mHC Scale SGD** | O(h) | ❌ No | N/A | N/A | ✅ Batch en optimizer_pipe |
| **JEPA Stabilizer** | O(h×layers) | ❌ No | N/A | N/A | ✅ Crear shader genérico |
| **Backward Pass** | O(h×ffn×layers) | ❌ No | N/A | N/A | ⚠️ Probar GEMV regular |
| **RMSNorm** | O(h) | ✅ Sí | ✅ Sí | ✅ Sí (`rms_pipe`) | ⚠️ Probar |
| **MHA** | O(h²×heads) | ✅ Sí | ✅ Sí | ✅ Sí (`mha_pipe`) | ⚠️ Probar |
| **GEMV (QKV)** | O(h×ffn) | ✅ Sí | ✅ Sí | ❌ No (`qkv_pipe`) | ❌ SIGSEGV |
| **GEMV (regular)** | O(h×ffn) | ✅ Sí | ✅ Sí | ✅ Sí (`gemv_pipe`) | ⚠️ Probar |
| **Optimizer batch** | O(n_weights) | ✅ Sí | ❌ No (`frame.acquire`) | ✅ Sí (`optimizer_pipe`) | ✅ **Seguro** |

---

## 6. Recomendaciones prioritizadas

### 🟢 Inmediato — Batch optimizer con mHC + STP
- El shader `optimizer_pipe` ya existe y es seguro en Iris Xe
- Incluir mHC scale updates y STP gradient injection en el batch
- No requiere `submit_and_wait` → evita SIGSEGV
- Archivo: `ash_backend.rs:658` (`dispatch_optimizer_batch_async`)

### 🟡 Mediano plazo — Shader genérico para JEPA + STP
- Usar `frame.acquire` + DoubleFrame (no `submit_and_wait`)
- Operaciones element-wise + reducción (mean/var)
- Safe en Iris Xe

### 🔴 Investigación — Probar shaders no-QKV en Iris Xe
- NS, RMSNorm, MHA, GEMV regular usan shaders diferentes al QKV
- Si el SIGSEGV es solo en el shader QKV, estos podrían funcionar
- Requiere testing en hardware real

### 🔵 Documentar el límite de la Iris Xe
- El SIGSEGV es en el shader QKV GEMV (`qkv_pipe`), no en `submit_and_wait` general
- `probe_gpu()` ya blacklistea Iris Xe, pero esto es conservador
- Podría relajarse para permitir shaders no-QKV

---

## 7. Implementación ya completada (2026-07-26)

Las siguientes optimizaciones de rendimiento del trainer ya fueron implementadas y verificadas:

1. **Fix crítico: ΔW tracking para el embedding** (`checkpoint.rs`)
   - Agregada comparación old-vs-new bytes para `token_embd.weight`
   - Agregada comparación old-vs-new para `token_embd.prq_scale`
   - Agregada comparación old-vs-new para path F32 del embedding

2. **Pre-allocación de buffers en `train_on_sequence`** (P-01)
   - 15 buffers pre-allocados en `SlimeBackwardWorkspace`
   - `std::mem::take` para extraer/restaurar buffers
   - Actualizados todos los call sites

3. **Eliminación de `ternary_data` Vec intermedio** (P-01)
   - Packing directo en buffer de salida via raw pointers
   - Ahorro: ~1.2 GB por sync (embedding) + ~4.7 MB por tensor

4. **Zero-copy scale comparison** (P-01)
   - Comparación directa contra `&[f32]` slices
   - Evita `.to_vec()` clonación

5. **Test de equivalencia de packing**
   - `test_direct_pack_matches_reference` verifica bytes idénticos

**Verificación:** 397 tests ✅ | clippy 0 errors/0 warnings ✅ | `cargo check --release` verde ✅
