# Forge LLM (MUD) - Roadmap

## Phase 1: Ternary Foundation (Completed)
- [x] Implementation of BitNet 1.58b ternary kernels (AVX2).
- [x] Proprietary `.ai` format with 16x compression.
- [x] Basic Tokenizer integration.

## Phase 2: Modular Intelligence (Completed)
- [x] **Autonomous Intent Orchestration:** No more manual commands.
- [x] **Modular Skill System:** Trait-based plugin architecture.
- [x] **Knowledge Graph (MKG):** Neural bridges and PageRank relevance.
- [x] **Persistent Store:** SQLite-backed long-term memory.
- [x] **Professional CLI:** Dashboard footer with real-time hardware telemetry.
- [x] **Bilingual Training:** High-scale Kaggle/Unsloth pipeline (EN/ES-LATAM).
- [x] **Transformer Core:** Multi-Head Attention, RoPE, and Sliding Window KV-Cache.
- [x] **Intelligent Sampling:** Top-K, Top-P, and Temperature sampling in Rust.

## Phase 3: Cognitive Expansion (In Progress)
### 1. Advanced Knowledge Handling
- [x] **Semantic RAG:** `MudIngester` using model embeddings.
- [ ] **PDF/Office Ingestion:** Native support for non-text formats in `MudIngester`.
- [ ] **Recursive Learning:** Model periodically "dreams" (assembles) its persistent store into new skill blocks.

### 2. High-Performance Hardware Tuning
- [x] **Vulkan Subgroups:** SPIR-V 1.3 optimization for parallel reductions on Intel Iris Xe.
- [ ] **Kernel Fusion:** Fusing Norm + RoPE + GEMV into single compute dispatches.
- [ ] **KV-Cache Quantization:** Moving context memory to INT8/FP16 for 2x RAM efficiency.
- [ ] **Parallel MoE:** Threaded expert execution for multi-core P-cores.
- [ ] **Mobile Portability:** Initial tests for running MUD on Android/ARM via Vulkan.

### 3. Specialty Skills
- [ ] **CodingExpert:** Dedicated experts for Python, Rust, and SQL generation.
- [ ] **LogicValidator:** A skill that performs self-correction of math/logic outputs.
- [ ] **VisionModule:** Integration of ternary-quantized vision encoders.

## Phase 4: Decentralized Understanding
- [ ] **Knowledge Sharing:** Peer-to-peer exchange of MUD skill modules.
- [ ] **Federated Learning:** Local weight updates synchronized via safe diffs.
