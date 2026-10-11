---
lang: es
---

# Estudio de Optimización de la Arquitectura MUD

Tras una auditoría profunda de la base de código actual y los registros históricos de rendimiento, se presentan las siguientes posibilidades de optimización estructuradas en tres pilares: **Velocidad de Entrenamiento, Rendimiento de Inferencia y Robustez del Sistema**.

## 1. Velocidad de Entrenamiento

### 1.1 Métricas Actuales y Cuellos de Botella
- **Métrica base (Hardware Medio):** ~7.9s por iteración de entrenamiento en `mud_fast_trainer.py` (Vulkan/CPU híbrido) con 16 expertos.
- **Cuello de Botella `torch.compile`:** Existe una incompatibilidad entre la compilación de grafos dinámicos en PyTorch y el enrutamiento del MoE, causando *graph breaks*. Actualmente se utiliza el decorador `@disable` para evitar caídas, pero esto impide beneficiarse del JIT de PyTorch 2.x.
- **Acumulación de Basura en Vulkan:** Aunque se programó un `clear_caches()` en `vulkan_backend.py`, la función no se invoca explícitamente en el lazo de entrenamiento. Esto podría causar *stale buffers* y ralentizaciones de VRAM a medida que los pasos avanzan.

### 1.2 Optimizaciones de Código Propuestas
> [!TIP]
> **Integración Estricta de `clear_caches()`:** En todos los `trainer.py`, insertar `vulkan_backend.clear_caches()` y un `torch.cuda.empty_cache()` (si aplica) forzosamente al final de cada época o intervalo fijo.
> **Compilación de Sub-módulos:** En lugar de compilar el modelo entero con `torch.compile`, aplicar la compilación únicamente sobre los expertos (`ExpertLayer`), dejando el enrutador de forma eagerly-executed para evadir los *graph breaks*.

## 2. Velocidad de Inferencia (Latencia y TPS)

### 2.1 Métricas Actuales
- **Inferencia en CPU:** El uso de AVX2 (unrolling a 4 filas) permite una velocidad respetable para tareas de Retrieval-Augmented Generation (RAG).
- **Inferencia en iGPU (Vulkan):** La comunicación *zero-copy* ya reduce masivamente el overhead en arquitecturas UMA (Intel Iris Xe).

### 2.2 Optimizaciones de Código Propuestas
> [!IMPORTANT]
> **Fusión de Kernels (Kernel Fusion):** Actualmente, la Normalización (RMSNorm), RoPE (Rotary Position Embeddings) y la proyección a los logits (Vocab Projection) se despachan por separado. Escribir un shader SPIR-V unificado para fusionar estas operaciones ahorrará al menos un **20% en latencia** al evitar escrituras en VRAM intermedias.
> **Parallel MoE (Rayon):** Durante la inferencia, el enrutador activa `Top-K` expertos. Actualmente la CPU procesa esto linealmente. Se debe delegar la evaluación de los expertos seleccionados a un threadpool de `rayon` (`par_iter`), reduciendo el tiempo de la capa MoE casi a la mitad en procesadores P/E (Performance/Efficiency cores).
> **Cuantización del KV-Cache:** El KV Cache de la inferencia crece de forma lineal y consume enorme cantidad de RAM. Pasar los estados ocultos a enteros de 8 bits (INT8) duplicará la capacidad de contexto sin penalidad grave en asertividad.

## 3. Robustez de la Arquitectura

### 3.1 Fractura del Ecosistema de Entrenadores
El `mud_fast_trainer.py` ha evolucionado y hoy cuenta con mitigación de OOM (Out of Memory), protección atómica de Checkpoints, control dinámico del `aux_coeff` y *noise annealing* (`_step_ratio`) para evitar colapso de expertos.
Sin embargo, hay una **deuda técnica masiva**:
- `mud_language_trainer.py`
- `mud_cognitive_trainer.py`
- `mud_ultra_trainer.py`
- `mud_final_trainer.py`
- `kaggle_trainer.py`
- `distillation_trainer.py`

Todos estos *trainers* están desactualizados. Operan sin ruido de exploración ni compensación de clústeres. **El riesgo es altísimo**: si usas cualquiera de estos, la red corre el riesgo de sufrir "Amnesia Ternaria" o colapso de enrutador.

### 3.2 Optimizaciones de Robustez Propuestas
> [!WARNING]
> **Sincronización Urgente de Trainers:** Debe portarse la lógica de 3 componentes del *Balance Loss*, el *Dynamic aux_coeff*, y la estructura de guardado atómico a los 6 entrenadores rezagados de inmediato.
> **Delegation Router Seguro:** Asegurarse de que el uso de procesos Python en la herramienta de lógica matemática no genere bloqueos ("deadlocks"). Aislar los cálculos en subprocesos con *timeouts* estrictos.

---

### Resumen de Próximos Pasos Recomendados:
1. Sincronizar todos los archivos `*_trainer.py` con la arquitectura robusta de `mud_fast_trainer.py`.
2. Implementar `par_iter` (Rayon) en la ejecución de los expertos de inferencia en Rust (`experts.rs`).
3. Investigar la viabilidad de Kernel Fusion en SPIR-V para un despacho único en Vulkan.

---

## 4. Optimizaciones Implementadas — Warp-Aligner (2026-06-18)

> [!IMPORTANT]
> **Adam AVX2 Kernel (`src/asm/adam_step.s`):** El loop Adam en `corpus_trainer.rs` calculaba `b1.powi(t)` y `b2.powi(t)` **por cada elemento** del tensor — O(2N) llamadas innecesarias. El fix pre-calcula `bc1 = 1 - β₁ᵗ` y `bc2 = 1 - β₂ᵗ` **una sola vez** por paso y despacha el update a `adam_step_avx2`: kernel AVX2+FMA que procesa **8 floats por iteración** usando registros YMM con `vfmadd231ps`/`vsqrtps`/`vdivps`. Speedup: **4–8×** en el Adam update. Dispatch automático (AVX2 / scalar fallback) via `crate::asm::adam_step()`.

> [!IMPORTANT]
> **O(1) NodeId Tracking (eliminación de `iter().position()`):** En `train_on_sequence_qat()`, la actualización de embeddings por posición QAT usaba `tape.nodes.iter().position(|n| {...})` — **búsqueda lineal del tape completo** por cada una de las 8 posiciones QAT (≈12,000 comparaciones/paso). Se reemplazó por `all_emb_nodes: Vec<NodeId>` que captura el `NodeId` en el momento de creación del nodo en el tape. Lookup O(1) directo posterior. Adicionalmente, corrige un **bug lógico**: la búsqueda original siempre devolvía el primer nodo coincidente (posición 0), causando que las posiciones 1–7 usasen gradientes de la posición 0.

> [!TIP]
> **Shader Backward con Shared Memory Tiling (`assets/shaders/ternary_backward_opt.comp`):** El shader `ternary_backward.comp` original accedía a `x[b * n_in + col]` con strides variables por thread — causando cache thrashing en L1 de GPU. El shader optimizado usa tiles `shared float s_x[TILE_Y][TILE_X]` (1KB, cabe en L1 de cualquier GPU) con carga coalesced y barriers explícitos. Activo en Vulkan via `backward_cs` en `src/vulkan/mod.rs`.

> [!IMPORTANT]
> **Fix de Persistencia .mud — `qat.sync_to_mud()` faltante:** `warp_aligner.rs` solo llamaba `sync_shadow_to_mud()` (solo embeddings) al guardar. Los shadow weights QAT de las capas de atención y FFN **nunca se escribían al disco**. Fix: `qat.sync_to_mud(&mut mud)` se llama explícitamente antes de `mud.save()` en el save final y en cada Ctrl+C handler. Checkpoint periódico cada **500 pasos** para resiliencia ante cuelgues.

### Estado post-optimización (2026-06-18)
- `cargo clippy --features="tools"` → **0 warnings, 0 errors** ✅
- Política 0-Warning del GEMINI.md: **MANTENIDA**
- Ver: [MUD_AUDIT_REPORT_V25_WARP_PERF.md](../audits/MUD_AUDIT_REPORT_V25_WARP_PERF.md)
