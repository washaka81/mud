# Forge LLM (MUD) - Roadmap

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

## Phase 3: Cognitive Expansion (In Progress)
### 1. Advanced Knowledge Handling
- [x] **Semantic RAG:** `MudIngester` using model embeddings.
- [x] **Massive Ingestion:** Ingested 59k facts from synthetic CoT datasets.
- [x] **Full UTF-8 Support:** Expanded vocabulary for accents, eñes, and emojis.
- [ ] **PDF/Office Ingestion:** Native support for non-text formats in `MudIngester`.

### 2. High-Performance Hardware Tuning
- [x] **Vulkan Subgroups:** SPIR-V 1.3 optimization for parallel reductions on Intel Iris Xe.
- [x] **SIMD Acceleration:** AVX2 kernels for Dot Product and Sum of Squares (~8x faster RAG).
- [x] **Fused Vocab Projection:** GPU/AVX2 accelerated output logits.
- [ ] **Kernel Fusion:** Fusing Norm + RoPE + GEMV into single compute dispatches.
- [x] **KV-Cache Quantization:** Moving context memory to INT8/FP16 for 2x RAM efficiency.
- [x] **Parallel MoE:** Threaded expert execution for multi-core P-cores.
- [ ] **Mobile Portability:** Initial tests for running MUD on Android/ARM via Vulkan.
- [ ] **Pure Ternary Logic Engine:** Transition from simulation to Balanced Ternary ISA (ISZ/ISP primitives).
- [x] **Mathematical Delegation Router:** Integration of a secure sandbox (Python/SymPy) for exact calculations.

### 3. Specialty Skills
- [ ] **CodingExpert:** Dedicated experts for Python, Rust, and SQL generation.
- [x] **LogicValidator:** A skill that performs self-correction of math/logic outputs.
- [ ] **VisionModule:** Integration of ternary-quantized vision encoders.

### 3b. MoE Balance & Convergence (Completed)
- [x] **3-component balance loss:** importance-var + load-var + z-loss for router collapse prevention.
- [x] **Noisy top-k gating with annealing:** `noise_std = 0.1 * (1.0 - step_ratio)`, decays linearly to 1%.
- [x] **`aux_coeff` propagation:** per-instance coefficient threaded through `MudModel → MudBlock → MoELayer`.
- [x] **`--experts` override-aware coeff:** `_eff_coeff` computed from actual expert count (≤16→0.5, ≤64→0.1, >64→0.05).
- [x] **Audit infra:** `tests/mud/` placeholder for balance assertions >85% after 600 steps.
- [x] **Port 3-component balance** to `language`, `cognitive`, `ultra`, `final`, `kaggle`, `distillation` trainers.
- [x] **`clear_caches()` integration** in training loop to prevent Vulkan stale buffer accumulation.

## Phase 4: Decentralized Understanding & Robustness
- [ ] **Knowledge Sharing:** Peer-to-peer exchange of MUD skill modules.
- [ ] **Federated Learning:** Local weight updates synchronized via safe diffs.
- [x] **Autonomous Research & TTL Rotation:** Integrated `ResearchSkill` for live knowledge ingestion and automatic 1-year factual rotative pruning.
- [ ] **Autocorrelation Penalization:** Penalizar la función de pérdida cuando se detectan oscilaciones repetitivas y autocorrelacionadas en el enrutamiento del MoE.
- [ ] **Confidence-Guided Sampling:** Ajuste dinámico de Top-K y temperatura; si la confianza interna (basada en entropía) decae, el modelo fuerza una selección más determinista.
- [ ] **Dynamic Delta Compensation:** Ajustar el coeficiente auxiliar de los expertos automáticamente cuando el gradiente de la pérdida (delta) entra en una meseta, forzando la exploración de nuevos expertos.
- [ ] **Kernel Fusion en Vulkan (Inferencia):** Consolidar múltiples operaciones (Norm + RoPE + GEMV) en un solo dispatch unificado para erradicar el overhead del driver gráfico.
- [ ] **Paralelismo Rayon (Rust):** Implementar e inyectar `par_iter()` avanzado en `src/mud/inference.rs` para maximizar saturación de P-Cores y E-Cores.
