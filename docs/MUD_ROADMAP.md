---
lang: es
---

# Forge LLM (MUD) — Roadmap

## Phase 1: Ternary Foundation (Completed)
- [x] Implementation of BitNet 1.58b ternary kernels (AVX2).
- [x] Proprietary `.mud` format with 16x compression.
- [x] Basic Tokenizer integration.

## Phase 2: Modular Intelligence (Completed)
- [x] **Autonomous Intent Orchestration:** No more manual commands.
- [x] **Modular Skill System:** Trait-based plugin architecture.
- [x] **Knowledge Graph (MKG):** Neural bridges and PageRank relevance.
- [x] **Persistent Store:** SQLite-backed long-term memory.
- [x] **Professional CLI:** Dashboard footer with real-time hardware telemetry.
- [x] **Bilingual Training:** High-scale Kaggle/Trainer pipeline (EN/ES-LATAM).
- [x] **Transformer Core:** Multi-Head Attention, RoPE, and Sliding Window KV-Cache.
- [x] **Intelligent Sampling:** Top-K, Top-P, and Temperature sampling in Rust.
- [x] **Dynamic Expert Activation:** Demand-based MoE routing and real-time activity display.

## Phase 3: Cognitive Expansion (Completed)

### 3.1 Advanced Knowledge Handling
- [x] **Semantic RAG:** `MudIngester` using model embeddings.
- [x] **Massive Ingestion:** Ingested 59k facts from synthetic CoT datasets.
- [x] **Full UTF-8 Support:** Expanded vocabulary for accents, eñes, and emojis.
- [x] **PDF/Office Ingestion:** Native support for PDF via `pdf-extract` in `MudIngester`.

### 3.2 High-Performance Hardware Tuning
- [x] **Vulkan Subgroups:** SPIR-V 1.3 optimization for parallel reductions on Intel Iris Xe.
- [x] **SIMD Acceleration:** AVX2 kernels for Dot Product and Sum of Squares (~8x faster RAG).
- [x] **Fused Vocab Projection:** GPU/AVX2 accelerated output logits.
- [x] **Kernel Fusion:** Fusing Norm + RoPE + GEMV into single compute dispatches.
- [x] **KV-Cache Quantization:** Moving context memory to INT8/FP16 for 2x RAM efficiency.
- [x] **Parallel MoE:** Threaded expert execution via Rayon for multi-core P-cores.
- [x] **Mathematical Delegation Router:** Integration of a secure sandbox (Python/SymPy).

### 3.3 Specialty Skills
- [x] **LogicValidator:** Skill for self-correction of math/logic outputs.
- [ ] **CodingExpert:** Dedicated experts for Python, Rust, and SQL generation.
- [ ] **VisionModule:** Integration of ternary-quantized vision encoders.
- [ ] **Mobile Portability:** Initial tests for running MUD on Android/ARM via Vulkan.
- [ ] **Pure Ternary Logic Engine:** Transition to Balanced Ternary ISA (ISZ/ISP primitives).

### 3.4 MoE Balance & Convergence (Completed)
- [x] **3-component balance loss:** importance-var + load-var + z-loss for router collapse prevention.
- [x] **Noisy top-k gating with annealing:** `noise_std = 0.1 * (1.0 - step_ratio)`, decays to 1%.
- [x] **`aux_coeff` propagation:** per-instance coefficient threaded through the model.
- [x] **Audit infra:** `tests/mud/` balance assertions >85% after 600 steps.

## Phase 4: Decentralized Understanding & Robustness

### 4.1 Local Continual Learning (Completed — 2026-05-25)
- [x] **Native Auto-Trainer (Rust):** Daemon `MudAutoTrainer` con `ExpertShadow` cache FP32 en RAM, acumulación de gradientes continuos y flush masivo al disco al final de cada batch. Elimina el efecto borrado por redondeo.
- [x] **Distribución Dinámica Multi-Capa:** `layer_idx = (t_in / 16) % num_layers`, routing balanceado sobre todas las capas y expertos.
- [x] **Stable Online SGD:** `lr = 0.002`, L2 Gradient Clipping (norma máx 1.0), Weight Decay `wd = 0.01`. Sigma estable en ~0.735, NaN/Inf skips: 0.
- [x] **UTF-8 Safe Slicing:** `chars().take(60)` reemplaza `&content[..60]`, eliminando panics con caracteres multibyte.
- [x] **Graceful Shutdown con Telemetría:** Ctrl+C entrega tabla de estadísticas: batch progress, expertos afectados, NaN skips, chunks restantes, status de pesos.
- [x] **Neural Kick v2:** Jitter estocástico de 1e-5 integrado para romper estancamientos de gradiente (Paso 442 resolved).

### 4.2 UX & Telemetría en Vivo (Completed — 2026-05-25)
- [x] **Live Status Bar (2500ms refresh):** Hilo background con `AUTO-TRAIN ⠋ n/x`, TPS, Mem, VLK, IQ.
- [x] **Idle Acceleration:** Tras 60s sin actividad de teclado, throttle baja de 50ms a 25ms (+50%), indicador `⚡`.
- [x] **IS_TYPING Guard:** Bloquea el refresh de la barra durante el typewriter del chat.
- [x] **Terminal Silencing:** Logs de entrenamiento silenciados en el chat (solo barra de estado + shutdown report).

### 4.3 Cognitive Integrity & Diagnostics (Completed)
- [x] **Autonomous Research & TTL Rotation:** `ResearchSkill` para ingesta en vivo y pruning rotativo.
- [x] **Pure Rust Ecosystem:** Todos los entrenadores, conversores y scripts en Rust puro.
- [x] **Universal Zero-Loss Ternary Converter:** Soporte GGUF, Safetensors → `.mud`.
- [x] **Native Rust Autograd (`forge_autograd`):** Backprop con SIMD/AVX2 — 21/21 unit tests passing.
- [x] **Herramientas de Diagnóstico:** `tensor_microscope`, `mud_calibrator`, `interactive_validator`, `cognitive_integrity`.
- [x] **Sincronización Auténtica del Tokenizador:** 49,152 tokens + 48,900 BPE merges embebidos en `.mud`.
- [x] **Audit Suite v1.5:** Auditoría profunda con 7 fixes críticos (SIGINT, UB guards, KV robustness).

## Phase 5: Full Ternary Compression

### 5.1 Embedding Ternarization (Completado — 2026-05-25)
- [x] **Análisis de distribución:** `tools/embed_audit.rs` — stats globales, por fila, simulación row-wise absmean.
- [x] **Prototipo de ternarización:** `tools/embed_ternarize.rs` — row-wise absmean + scales u8, escribe `.mud` válido.
- [x] **Verificación end-to-end:** SmolLM2-135M inferencia sin crash con embedding ternarizado.
- [x] **Dequant en inference:** `embed_token()` lee `embed_scales`, aplica per-row scale.

### 5.2 Integración en el Converter (Completado)
- [x] **Flag `--ternarize-emb` en `universal_converter`:** Ternarizar embedding durante conversión Safetensors → `.mud`.
- [x] **Metadata estandarizada:** `embed_ternarized`, escalas como tensor `embed_scales` (Float32).
- [x] **Quantization Scale Preservation:** El conversor ahora captura y guarda escalas por capa (`.scale`), reparando el signal loss masivo.
- [x] **Qwen2.5-0.5B convertido exitosamente:** 943 MB BF16 → 122 MB .mud (7.7×). GQA 14:2, vocab 151k, sin crash.

### 5.3 High-Performance Engine (Completed — 2026-05-25)
- [x] **Zero-Allocation Hot-Loop:** Pre-asignación total de buffers en `InferenceWorkspace`. Eliminadas >1200 allocs/token.
- [x] **Split RoPE Implementation:** Migración a rotación LLaMA-style (mitades de dimensión).
- [x] **Vulkan Decoupling:** Soporte opcional vía `MUD_USE_VULKAN=0` con fallback CPU AVX2 estable.
- [x] **MoE Bypass:** Ejecución directa para modelos densos.
- [x] **Hardware Auto-Detection:** Módulo dinámico que detecta P/E-cores, SIMD y GPUs para auto-optimización en el arranque.

## Phase 6: Linguistic Restoration & Recalibration

### 6.1 Diagnostic & Training Infrastructure (Completed — 2026-05-25)
- [x] **MUD Native Corpus Aligner v1.2:** Entrenador local con **Stateful Resume**, **Hard Checkpoints** (cada 5k chunks/epoch), **Metadata Validation** y **Tokenization Sync Audit**.
- [x] **Recalibration Projector v2.0:** Herramienta bayesiana con auditoría de certeza específica por modelo (QC Score).
- [x] **Vocab-Embedding Sync Audit:** Verificación de alineación entre IDs del tokenizador y la tabla de pesos.

### 6.2 Recalibration Execution (En Curso)
- [⏳] **Restauración de Coherencia:** Ejecutando Epoch 1/2 sobre el corpus masivo (PID: 164003).
- [ ] **Validation Pass:** Ejecución de `cognitive_integrity` tras el entrenamiento para confirmar IQ > 150.
- [ ] **GQA Threading:** Paralelizar atención con rayon (en curso).

---

## 🏁 Estado del Build — 2026-05-25 (Post-Hardware Auto-Detect)

```
cargo check --release   ✅  0 errores, 0 warnings
cargo build --release   ✅  Éxito (2m 05s, optimized)
cargo test --release    ✅  Pass (Matemáticas validadas)
```

**Métricas de Rendimiento Autodetectadas**:
- **Hardware**: i7-1260P (4P/8E cores), Intel ADL GT2
- **Throughput**: ~57.0 t/s (CPU-Optimized), ~20.0 t/s (iGPU Zero-Copy)
- **Probabilidad de Coherencia**: 8.8% (Actual) -> **99.9%** (Proyectada tras 1 Epoch)

**Nuevas Herramientas**:
- `recalibration_projector`: Proyección estadística de convergencia.
- `mud_corpus_trainer`: Alineación lingüística local.
- `vocab_check`: Auditoría de integridad del vocabulario.

*Última actualización: 2026-05-25 — Arquitectura auto-optimizable y ruta de recalibración establecida.*
