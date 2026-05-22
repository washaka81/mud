# MUD Optimization Log & Performance Roadmap
## Last Updated: 22 de mayo de 2026

This document registers identified bottlenecks and optimization opportunities for the Forge LLM (MUD) engine.

### 1. Vulkan Performance (Critical)
| Opportunity | Description | Estimated Impact |
| :--- | :--- | :--- |
| **Persistent Buffer Caching** | Currently, `run_ternary_gemv` creates and copies weight buffers on every call. Weights should be uploaded once to `DeviceLocal` memory during model load or skill activation. | **5x - 10x Speedup** in inference. |
| **Kernel Fusion** | Combine `RMSNorm`, `RoPE`, and `Projection` into single dispatches to minimize CPU-GPU synchronization and command buffer overhead. | **20% Latency Reduction**. |
| **Command Buffer Recording** | Pre-record static parts of the inference graph (e.g., within a layer) to avoid recording overhead on every token. | **10% Latency Reduction**. |

### 2. Core Logic & CPU (High)
| Opportunity | Description | Estimated Impact |
| :--- | :--- | :--- |
| **SIMD RAG (AVX2)** | Optimize `cosine_similarity` in `graph.rs` using AVX2. This is critical for autonomous retrieval in large knowledge bases. | **8x Speedup** in retrieval. |
| **Optimized Vocab Projection** | The final logit calculation in `inference.rs` uses a manual loop with dequantization. It should use `ternary_gemv_avx2` or a dedicated Vulkan kernel. | **Significant reduction** in time-to-first-token. |
| **KV-Cache Quantization** | Quantize KV-Cache to FP16 or INT8. Intel Iris Xe has specialized hardware for half-precision math. | **50% VRAM / 20% Speedup**. |

### 3. Architecture & Memory (Medium)
| Opportunity | Description | Estimated Impact |
| :--- | :--- | :--- |
| **String Interning** | Use `Arc<str>` or a string pool for node content in the Knowledge Graph to reduce allocation pressure during massive ingestion. | **Memory stability** during long runs. |
| **Async Knowledge Bridge** | Move `recalculate_ranks` and bridge construction to a background thread so they don't block the ingestion of the next chunk. | **Smoother ingestion** of large files. |
| **Parallel Expert Execution** | Run multiple experts in parallel via Vulkan subgroups if the GPU has enough occupancy. | **15% Speedup** in MoE layers. |

### 4. Implementation Status
- [ ] Vulkan Buffer Caching (Planned)
- [x] AVX2 Cosine Similarity (Completed)
- [x] Fused Vocab Projection (Completed)
- [ ] FP16 KV-Cache (Investigating)

### 4. SIMD & Pipeline Mastery (Mayo 2026)
| Opportunity | Description | Estimated Impact | Status |
| :--- | :--- | :--- | :--- |
| **Multi-Row Ternary Kernel** | New ASM kernel processing 4 rows at once to maximize L1 cache reuse. | **15% Speedup** | ✅ Completed |
| **Memory-Efficient DataLoader** | Lazy-loading of corpus to prevent RAM saturation. | **Stability Fix** | ✅ Completed |
| **Neural Kick (Jitter)** | Random ε-perturbation to prevent MoE expert collapse. | **Cognitive Boost** | ✅ Completed |
| **Full Metadata Injection** | Automatic arch descriptors in .mud files. | **Safety Fix** | ✅ Completed |
| **Parallel MoE (Rayon)** | Multithreaded expert execution in Rust. | **25% Speedup** | ⏳ Planned |

---

## Sesión 2026-05-21 — MoE 256 Expertos + Overflows + DB

### Arquitectura MoE Hiper-Granular
- **Nuevo universo:** 256 micro-expertos organizados en 16 clústeres funcionales
- **Tabla Maestra:** Creada `docs/MUD_MOE_EXPERTS.md` con descripción completa
- **Invariantes:** NUM_EXPERTS=256, CLUSTER_SIZE=16, TOP_K=4, AUX_COEFF=0.01
- **Clústers definidos:** Planificación/CoT, Lógica Formal, Evaluador Interno, Razonamiento Difuso, Gramática AST, Optimización Bajo Nivel, Algoritmia, Álgebra Lineal, Cálculo, Estadística, Física Cuántica, Mecánica Clásica, Química, Bioinformática, Sistemas Complejos, Taxonomías

### Correcciones de Overflow (inference.rs)
- `kv_cache` allocation: `num_layers * 4096 * hidden_size` → `checked_mul()` con `expect()`
- Expert indexing: `layer.experts[expert_id]` → `layer.experts.get(expert_id)` (filter_map)
- Logit bounds: `ws.logits[prev_id as usize]` → `ws.logits.get_mut(idx)`
- `embed_token`: `id * (hidden_size/16)` → `checked_mul()` + `debug_assert!(hidden_size % 16 == 0)`

### Correcciones de Overflow (mod.rs)
- `mmap load`: `data_start + tensor.offset` → `checked_add()` + bounds assert vs mmap_len
- `dequantize_ternary_row`: truncado silencioso si `n % 16 != 0` → loop residual separado + `debug_assert!(out.len() >= n)`

### Correcciones Router (routing.rs)
- Empty logits → early return `vec![]`
- `debug_assert_eq!(logits.len(), self.num_experts)` para detectar mismatch gate/router
- `sum_exp == 0.0 || !sum_exp.is_finite()` → fallback a mejor experto
- Re-normalización: `if new_sum > 0.0 && new_sum.is_finite()` antes de dividir

### Hardening de Base de Datos (store.rs)
- **PRAGMA WAL mode** + `busy_timeout=5000ms` para lectura concurrente inferencia/auto-trainer
- **4 índices automáticos:** `status`, `rank DESC`, `timestamp DESC`, `learning_mark`
- **Mutex envenenado:** todos los `unwrap()` → `map_err(|_| anyhow!(...))` (10 sitios)
- **Blob alignment:** `aligned_len = blob.len() - (blob.len() % 4)` antes de leer f32
- **Sanitización de rank:** `f64` → `.clamp(0.0, 1e6)` para evitar NaN/Inf en PageRank
- **`get_unassimilated()` sin LIMIT** → `LIMIT 1000` para proteger RAM
- **`update_rank()`** sanitiza NaN antes de persistir con clamp

### Herramienta de Auditoría (tools/moe_audit.rs)
- Reescrita para 256 expertos en 16 clústeres
- Test 1: Anatomía de pesos por clúster (W1/W2/W3 magnitudes, estado: ÓPTIMO/ACTIVO/DÉBIL/MUERTO)
- Test 2: Balance de carga del router (histograma por clúster, score de balance %)
- Test 3: Stress-test Vulkan 50 pasos con detección de NaN
- Test 4: Coherencia cuantizador ternario (distribución +/-/0, asimetría)

### Trainer Local (training/mud_fast_trainer.py) — v2.0
- NUM_EXPERTS: 8 → 256 | NUM_LAYERS: 1 → 6 | TOP_K: 2 → 4
- `MoELayer` con telemetría de activación por clúster (`cluster_activations` buffer)
- `log_cluster_balance_to_db()`: persiste stats MoE en tabla `moe_balance_log` de SQLite
- AUX_COEFF = 0.01 (era 10.0 — reducción 1000x para evitar que domine el entrenamiento)
- `_pack_ternary()`: corregido manejo de residuo (`len % 16 != 0`)
- Checkpoint: `load_state_dict(strict=False)` para compatibilidad entre versiones
- `weights_only=True` en torch.load (seguridad)
- Argumentos CLI: `--experts`, `--top-k`, `--log-balance`

### Trainer Kaggle (training/kaggle_trainer.py)
- EXPERTS: 8 → 256 | NUM_LAYERS: 1 → 6 | TOP_K: 2 → 4
- `balance_loss`: `importance.var() * 10.0` → `importance.var() * 0.01 * self.num_experts`

### Script Maestro (scripts/train_master.sh) — v2.0
- `set -euo pipefail` (fail-fast estricto)
- Parámetros: `--experts`, `--top-k`, `--steps`, `--quick`, `--resume`
- Modo `--quick`: solo moe_audit + weight_audit
- Modo `--test-all`: suite completa de 10 herramientas con PASS/FAIL tracking
- Modo `--kaggle`: inyecta NUM_EXPERTS/TOP_K en kernel-metadata.json
- Verificación de corpus y vocabulario antes del entrenamiento
- Post-training: `moe_audit` automático sobre modelo exportado

### Estado del Proyecto
- **cargo check:** ✅ 0 errores, 0 warnings
- **Compilación:** limpia con `RUSTFLAGS="-C target-cpu=native"`
- **Próximos pasos:** implementar `experts.rs` con tabla const de 256 ExpertDescriptor

---

## Sesión 2026-05-21 — Vulkan zero-copy + AVX2 backward + 200-step validation

### Vulkan Persistent Buffers (mod.rs)
- **Weight copy eliminado del hot path:** `copy_from_slice()` de pesos movido dentro del bloque `if recreate`. Los pesos ternarios se suben **una sola vez** a VRAM, no en cada forward call. Impacto estimado: 5-10x en inferencia.
- **Device selection:** prioriza `IntegratedGpu` (score 0) sobre `DiscreteGpu` (score 1) para aprovechar UMA zero-copy en iGPU Intel.
- **`is_available()`:** ahora retorna `self.available` en lugar de `true` hardcodeado.
- **`vb_clear_caches()`:** nueva función FFI para reset explícito de buffers Vulkan cuando los pesos cambian entre pasos de training.

### AVX2 Backward Pass (vulkan_backend.rs)
- **`gemv_transpose_avx2_row()`:** nueva función que procesa la transpuesta GEMV del backward usando AVX2 intrinsics. Desempaqueta 16 pesos ternarios por bloque con `_mm256_srlv_epi32` + `_mm256_blendv_ps`, multiplica por `dy_i` y acumula en `dx` con `_mm256_add_ps`.
- **`outer_product_avx2_row()`:** actualización de gradientes con 8 floats por iteración usando `_mm256_mul_ps` + `_mm256_add_ps`.
- **Forward 4-rows:** `gemv_cpu()` ahora usa `ternary_gemv_4rows_avx2` para procesar 4 filas simultáneamente, reduciendo dispatch overhead 4x.

### Thread + resource_tracker Fix (todos los trainers)
- **Semaphore leak eliminado:** `multiprocessing.set_start_method('spawn')` + `torch.multiprocessing.set_sharing_strategy('file_system')` al inicio de cada trainer.
- **Thread config unificada:** `OMP_WAIT_POLICY=PASSIVE`, `MKL_DYNAMIC=FALSE`, `KMP_BLOCKTIME=1`, `KMP_AFFINITY=granularity=fine,compact,1,0` en todos los trainers.
- **Inter-op parallelism:** `torch.set_num_interop_threads(1)` para evitar overhead.
- **MKL threads:** limitado a 8 threads (por escalado sub-lineal en CPUs híbridas).

### torch.compile Fix
- `@torch.compiler.disable` añadido a `MoELayer.forward()` para evitar graph breaks que causaban `RuntimeError: backward through graph a second time`.

### Entrenamiento 200 pasos — Validación
- **Config:** 16 expertos, 4 capas, hidden=384, 119.4M params, BF16, no-compile
- **Tiempo:** 26 min (7.9s/it), 21,497 tokens procesados
- **Loss:** 8.4 → 5.4 (descenso estable, sin NaN, sin OOM)
- **Export:** `models/core_skills.mud` (59.5 MB)
- **Auditoría:** 10/10 PASS
  - weight_audit: distribución ternaria simétrica (37% +1, 26% 0, 37% -1)
  - moe_audit: balance 79.0%, NaN stress 0/50
  - ternary_audit: bit-exact precision vs Rust reference
  - deep_math_audit: skew ~0, kurtosis ~-1.6 (ideal)
  - truth_auditor: veracidad baja (esperado con 200 pasos)

### Auto-Config + Hardware Profiler
- `tools/hardware_profiler.py`: detecta CPU/RAM/GPU/Vulkan, corre micro-benchmark de 10 pasos, persiste en `knowledge.db`
- `training/auto_config.py`: lee DB y retorna config óptima para cualquier trainer
- `scripts/train_master.sh` paso [1/6]: integra profiler + auto-config + `--dry-run`
- `mud.sh profile`: nuevo comando para perfilado manual

### Issues Conocidos
- **torch.compile** incompatible con MoELayer por graph breaks en dispatch loop. Solución: `@disable` + `--no-compile`
- **Balance MoE débil (79%)** con 200 pasos: solo 1 clúster activo dominante. Requiere más pasos o aumentar aux_coeff.
- **resource_tracker warning** persiste esporádicamente (1 vez cada ~115 pasos) aunque muy reducido.
- **5/7 trainers quedaron atrás**: language, cognitive, ultra, final, distillation, kaggle — no tienen `aux_coeff` configurable, `_step_ratio` para noise annealing, ni el balance loss de 3 componentes. Solo `mud_fast_trainer.py` fue actualizado completamente.
- **`clear_caches()` nunca se llama**: definido en `vulkan_backend.py` pero ningún entrenador lo invoca en el loop de training.
- **Coeficientes hardcodeados**: 4 trainers usan `10.0` (excesivo), 1 usa `0.01`. Solo `mud_fast_trainer.py` y `auto_config.py` usan constantes nombradas escaladas por RAM.

---

### 7. `aux_coeff` Propagación + `_step_ratio` (22 de mayo de 2026)

**Contexto:** El balance loss de `mud_fast_trainer.py` usaba `AUX_COEFF` como constante del módulo. Al overridear `--experts`, el coeficiente no se ajustaba. Además, el ruido de exploración (noisy top-k gating) no tenía un mecanismo de annealing controlado.

**Cambios en `training/mud_fast_trainer.py`:**
- `MoELayer.__init__`: nuevo parámetro `aux_coeff: float = 0.05` (per-instancia, no constante global)
- `MoELayer.__init__`: nuevo buffer `_step_ratio` para annealing de ruido
- `MoELayer.forward`: `_step_ratio` controla `noise_std = 0.1 * (1.0 - self._step_ratio.item())`. La exploración se desvanece linealmente de 10% a 1%
- `MudModel.__init__`, `MudBlock.__init__`: propagan `aux_coeff` por cadena de constructores
- `train()`: computa `_eff_coeff` según número real de expertos si `--experts` difiere del default:
  - ≤16 expertos → 0.5 | ≤64 → 0.1 | >64 → 0.05
  - Actualiza banner: muestra `_eff_coeff` en lugar de `AUX_COEFF`
- Loop de training: itera `model.modules()` cada paso para setear `MoELayer._step_ratio = step_ratio`

**Cambios en `training/auto_config.py`:**
- Small mode: `top_k=3` (antes 2), `aux_coeff=0.5` (antes 0.2)

**Auditoría Completa (22 mayo):**
- Rust: `cargo check` — ✅ pasa
- Python syntax check — 16/16 archivos en `training/` — ✅ todos compilan
- Todos los `.py` compilados con `py_compile` — ✅ sin errores

**Oportunidades de mejora documentadas en audit:**
| Archivo | Problema | Severidad |
| :--- | :--- | :--- |
| `mud_language_trainer.py` | Sin `aux_coeff`, `_step_ratio`, balance hardcodeado 10.0 | Alta |
| `mud_cognitive_trainer.py` | Sin `aux_coeff`, `_step_ratio`, balance hardcodeado 10.0 | Alta |
| `mud_ultra_trainer.py` | Sin `aux_coeff`, `_step_ratio`, balance hardcodeado 10.0 | Alta |
| `mud_final_trainer.py` | Sin balance loss en absoluto | Crítica |
| `kaggle_trainer.py` | Sin `aux_coeff`, `_step_ratio`, balance hardcodeado 0.01 | Media |
| `distillation_trainer.py` | Sin `aux_coeff`, `_step_ratio`, balance hardcodeado 10.0 | Alta |
| Todos | `clear_caches()` nunca se invoca desde ningún trainer | Baja |
| `mud_fast_trainer.py`, `mud_ultra_trainer.py` | No importan `auto_config.py` (usan RAM detection propia) | Media |
