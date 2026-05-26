---
lang: en
---

# MUD: Cognitive Assimilation Plan (CAP)

## Goal: Moving from Retrieval (RAG) to Intrinsic Knowledge (Weights)
MUD doesn't just "read" information via the Knowledge Graph; it **assimilates** facts into its ternary weights through a hardware-aware, automated training pipeline.

---

## The Unified Training Lifecycle (Auto-Edit Phase)
Training is no longer manual. It is governed by a central orchestrator that aligns architecture with hardware capability.

### 1. Hardware Profiling & Auto-Config
- **Tool:** `tools/hardware_profiler.py` & `training/auto_config.py`
- **Logic:** Upon execution, MUD detects CPU (AVX2/512), RAM, and GPU. It calibrates a "Mode" (Tiny, Small, Medium, Colab, Big) and persists optimal parameters (`hidden`, `num_layers`, `num_experts`) in `models/knowledge.db`.
- **Source of Truth:** All trainers (`mud_fast_trainer.py`, `mud_ultra_trainer.py`, `kaggle_trainer.py`) MUST inherit their architecture from `load_training_config()`.

### 2. Massive Ingestion & Learning Marks
- **Status:** Active
- **Mechanism:** Ingested facts in `knowledge.db` are tagged with `learning_mark`.
    - **0 (Raw):** Ingested but not yet distilled.
    - **1 (Learned):** Integrated into weights via training.
    - **2 (Master):** Critical verified knowledge.

### 3. Stability & Neural Health (Neural Kick)
- **Mechanism:** To prevent MoE expert collapse and break gradient plateaus (like the Step 442 bottleneck), a "Neural Kick" (Epsilon Jitter) is applied.
- **Intensity:** `1e-5` perturbation every 100 steps.
- **Impact:** Forces expert specialization and prevents weight stagnation in ternary space.

### 4. Weight Consolidation (Local & Cloud)
- **Local:** `./mud.sh train` launches the fast orchestrator using optimized Rust-compiled binaries and AVX2-accelerated trainers.
- **Kaggle:** `./mud.sh train --kaggle` dispatches the SAME configuration to the cloud for high-scale training (V1-MASTER).

---

## Phase Roadmap
1. **[DONE]** Automated hardware-to-architecture mapping.
2. **[DONE]** Fix state_dict size mismatch bugs in auto-config.
3. **[ACTIVE]** Scaling to 256 MoE experts on high-memory environments.
4. **[FUTURE]** PageRank-based loss weighting (Facts with higher connectivity get higher priority in weight updates).
5. **[FUTURE]** Local Trainer in Rust (Entrenador nativo en Rust para procesar chunks de la base de datos de conocimiento y asimilar información localmente sin depender de Python).
