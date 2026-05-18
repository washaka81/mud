# MUD: Cognitive Assimilation Plan (CAP)

## Goal: Moving from Retrieval (RAG) to Intrinsic Knowledge (Weights)
Currently, MUD "reads" books via the Knowledge Graph. To make it truly intelligent, it must **assimilate** this information into its ternary weights through a structured training pipeline.

---

## Phase 1: Massive Ingestion (The Library)
- **Status:** In Progress
- **Action:** Continue using `/ingest tests/data/books/` to build a high-quality SQLite database of Science, Math, Logic, and Programming.
- **Metric:** Reach > 10,000 knowledge chunks in `knowledge.db`.

## Phase 2: Synthetic Dreaming (Dataset Generation)
- **Status:** Planned
- **Logic:** A script will iterate through the SQLite facts and generate **Chain of Thought (CoT)** training pairs.
- **Format:**
  `Q: Explain the concept of [Subject] A: <thinking> [Logic steps derived from fact] </thinking> <answer> [Synthesized conclusion] </answer>`
- **Tool:** `tools/dreamer.py` (To be implemented).

## Phase 3: Weight Consolidation (Kaggle Training)
- **Status:** Running (v32)
- **Action:**
  1. Upload the generated synthetic dataset to Kaggle as a private dataset.
  2. Perform **Knowledge Distillation**: The model trains on its own "digested" version of the books.
  3. **Specialized Experts:** Lock specific MoE experts to specific domains (Expert 0-2: Logic, Expert 3-5: Code, Expert 6-7: Math).

## Phase 4: Autonomous Validation
- **Status:** Future
- **Logic:** The model is presented with unseen technical problems from the same books. It must solve them using its internal weights first, then verify via RAG if confidence is low (< 0.7 similarity).

---

## Technical Strategy: Index-to-Weight Mapping
1. **Embedding Alignment:** Ensure the training embedding layer uses the exact same vocabulary IDs as the Rust ingester.
2. **Loss Weighting:** Facts with high **PageRank** in the Knowledge Graph will have a 2x higher loss weight during training (Importance-based learning).
3. **BitNet Preservation:** Use Quantization-Aware Training (QAT) to ensure the newly learned facts are stable in a `{-1, 0, 1}` environment.
