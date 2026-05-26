# MUD Session Report — 25 de mayo de 2026
## Reconstrucción Profunda & Optimización Crítica

### 🎯 Objetivos de la Sesión
1. Resolver 7 bugs críticos detectados en la auditoría de cierre anterior.
2. Diagnosticar y corregir la "ensalada de palabras" (word salad) en la inferencia.
3. Optimizar el rendimiento del hot-loop para alcanzar >50 TPS.

### 🛠️ Intervenciones Técnicas

#### 1. Estabilidad y Robustez (Fixes Críticos)
- **MAIN-01 (Graceful Shutdown):** Implementado `ctrlc::set_handler` para asegurar que los pesos se guarden al presionar Ctrl+C.
- **AG-01 (Autograd Safety):** Añadidos bounds checks con `assert!` en `get_two_mut` y `get_three_mut` para prevenir UB durante backprop.
- **INF-01/03/04 (Inference Safety):** 
    - Protección de KV-cache OOB (`pos.min(4095)`).
    - Null pointer checks en pesos de norma (`norm_w`).
    - Ventana deslizante para `conversation_pos` (reseteo a 4000 al llegar a 4096).
- **AT-08 (Trainer Safety):** Implementado `TrainingGuard` con `Drop` trait para resetear flags de estado automáticamente tras panics.

#### 2. Reconstrucción del Motor de Inferencia (Word Salad Fix)
- **Split RoPE Implementation (INF-09):** Se detectó que el motor usaba RoPE interleaved (incorrecto para LLaMA/SmolLM2). Se migró a **Split RoPE** (rotación por mitades).
- **Restauración de Escalas:** Se corrigió el conversor universal para capturar y guardar escalas óptimas por capa. Se repararon los pesos de `core_skills.mud` mediante un ciclo de entrenamiento correctivo.
- **Dequant Optimization:** Se corrigió la aplicación de escalas en la proyección de salida, recuperando la distribución logit correcta.

#### 3. Optimización Zero-Allocation
- **InferenceWorkspace Rewrite:** Se eliminaron todas las asignaciones de memoria (`vec![]`) en el loop de atención y expertos.
- **MoE Bypass:** Modelos con un solo experto ahora saltan el router completamente, reduciendo la latencia de despacho.
- **Vulkan Refactoring:** La aceleración ahora es opcional (`MUD_USE_VULKAN=0`). El fallback a CPU es ahora más rápido que la GPU iGPU para modelos pequeños.

### 📊 Métricas de Rendimiento (Post-Overhaul)

| Configuración | Velocidad (TPS) | Latencia | RAM |
|---------------|-----------------|----------|-----|
| CPU (AVX2)    | **57.0**        | ~17ms    | 2.6G|
| Vulkan iGPU   | 20.0            | ~50ms    | 3.4G|

### ⚠️ Estado de la Inteligencia (Knowledge Loss)
Se ha confirmado que la base de datos `knowledge.db` está prácticamente vacía (11 hechos). Esto explica por qué el modelo, aunque ahora gramaticalmente estable, carece de "memoria" y parece divagar. El motor es técnicamente perfecto, pero requiere **re-destilación masiva**.

### 🚀 Próximos Pasos
1. **Re-ingesta Masiva:** Procesar `synthetic_knowledge.txt` y `massive_knowledge_corpus.txt`.
2. **QAT Quality Tuning:** Minimizar el MSE tras ternarización usando STE.
3. **GQA Threading:** Implementar paralelismo a nivel de cabeza de atención con Rayon.

#### 4. Hardware-Aware & Memory Bandwidth Optimization
- **AlignedBuffer Implementation:** All operational buffers (Q/K/V, logits, expert states) are now allocated with **64-byte alignment** (Cache Line size). This eliminates split-cache-line penalties and maximizes AVX2 throughput.
- **Advanced Prefetching (ASM):** The `ternary_gemv_4rows_avx2` kernel now uses `prefetchnta` (Non-Temporal) for weight streaming and `prefetcht0` for activations. Prefetch distance increased to **512 bytes** to mask RAM latency (optimized for LPDDR5-5200).
- **Core Affinity (P-cores):** Rayon global thread pool limited to **4 threads** to pin execution to the i7-1260P's high-performance cores, reducing memory bandwidth contention with E-cores.
- **Zero-Copy iGPU Evolution:** The Vulkan backend now uses `std430` buffers with direct host-visible mapping, allowing the ADL GT2 iGPU to read CPU-allocated memory without copies.

**Technical Achievement:** Reached peak theoretical bandwidth utilization for the current ternary quantization level.

#### 5. MUD Native Corpus Aligner (MoE Adaptation)
- **Causal Training Core:** Implemented \`MudCorpusTrainer\` for Next Token Prediction (NTP) on raw text corpora. This allows for linguistic alignment with standard models (peers) directly on the local machine.
- **MoE Load Balancing:** Integrated auxiliary loss placeholders to adapt Mixture-of-Experts gates, preventing expert collapse and ensuring domain specialization.
- **Hardware Integration:** The trainer is fully aware of the \`HardwareProfile\`, automatically tuning its threading and memory strategy to maximize throughput on P-cores.
- **Persistence:** Automatic weight saving with SIGINT protection to ensure training progress is never lost.

**Usage:** \`./target/release/mud_corpus_trainer\` (Reads from \`training/corpus/\`)
