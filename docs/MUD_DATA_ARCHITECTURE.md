# MUD Data Architecture: The Absolute Truth Lifecycle

This document defines the end-to-end flow of knowledge within the MUD (Modular Understanding Dynamics) ecosystem, from raw ingestion to neural assimilation.

---

## 1. Storage Layer: The Knowledge Base (`knowledge.db`)
- **Technology:** SQLite (Persistent disk-based storage).
- **Schema:** `facts` table storing `content`, `source`, `embedding`, `rank` (PageRank), and `timestamp`.
- **Integrity:** The `MudStore` module handles all SQL transactions, ensuring atomic updates and duplicate prevention.

## 2. Topological Layer: MUD Synapse Mesh
Instead of a flat list of facts, MUD organizes data into a **Neural Mesh**:
- **Primary Synapses:** Automatic connections based on high semantic similarity (>0.85).
- **Secondary Synapses:** Indirect connections between neighbors of neighbors, creating a dense logical web.
- **Connectivity Reward:** Facts that act as "knowledge hubs" (high synapse count) receive an automatic boost in their PageRank, prioritizing them for both inference and training.

## 3. Temporal Policy: Dynamic Retention (TTL)
To prevent cognitive stagnation and ensure relevance:
- **1-Year Rotation:** A strict Time-To-Live (TTL) of 365 days is enforced.
- **Purge Process:** The `enforce_ttl` worker automatically deletes facts older than one year, forcing the system to re-learn or update its knowledge via the `ResearchSkill`.
- **Goal:** Keep the "Absolute Truth" aligned with current real-world state.

## 4. Ingest & Assimilation: The Data Bridge
The transition from "External Memory" to "Intrinsic Intelligence" follows this pipeline:
1. **Ingestion:** `/ingest` reads local files, chunks text into 800-char blocks, and populates the Synapse Mesh.
2. **Dreaming (CoT):** `tools/dreamer.py` transforms facts into reasoning pairs (`<thinking> ... </thinking> <answer>`).
3. **v42 Bridge:** `tools/db_to_training.py` extracts the high-rank facts into a massive 223k+ sequence corpus.
4. **Final Assimilation:** The model is trained on this corpus, embedding the database's facts directly into the ternary MoE weights.

## 5. Autonomous Access: Indirect Synapse Injection
During inference, MUD accesses its data autonomously:
- **Interval:** Every 20 tokens (optimized for >20 TPS).
- **Mechanism:** The motor monitors its internal hidden state, performs a `jump_search` on the mesh, and injects the top-2 facts as a context bias (5% weight) into the neural trajectory.
- **Visual Feedback:** Confirmed by `[Synapse Activated]` logs.

---
**Status:** IMPLEMENTED | **Policy:** ROTATIVE (365d) | **Target:** 100% VERACITY
