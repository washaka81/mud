---
lang: en
---

# MUD: Training Protocols & Cognitive Assimilation

**Last Updated:** 30 de mayo de 2026
**Status:** Living Document — synced with code as of Audit V5

## Goal: Moving from Retrieval (RAG) to Intrinsic Knowledge (Weights)
MUD doesn't just "read" information via the Knowledge Graph; it **assimilates** facts into its ternary weights through a hardware-aware, automated training pipeline.

---

## The "Restore IQ" Pipeline (v1.5)
When a model is converted to ternary format, it undergoes "Ternary Shock". To restore semantic intelligence, we follow a standardized pipeline.

### Step 1: Convert (PRQ)
- **Goal:** Convert FP16/BF16 weights to 1.58-bit ternary with Per-Row Quantization.
- **Tool:** `universal_converter --ternarize-emb`
- **Method:** Row-wise `absmean` scale + round + clamp to {-1, 0, +1}.

### Step 2: Analyze (Bayesian QC)
- **Goal:** Assess signal health and quantization certainty.
- **Tool:** `recalibration_projector output.mud`
- **Method:** Statistical analysis of sigma, sparsity, and scale variance per layer.

### Step 3: Calibrate (Scale Boost)
- **Goal:** Optimize scales based on activation data.
- **Tool:** `recalibration_projector output.mud --boost`
- **Method:** Adjusts per-row scales to minimize SNR collapse.

### Step 4: Train (Live SGD — Ternary Manifold Seating)
- **Goal:** Let the weights settle into the discrete ternary manifold.
- **Tool:** `./mud.sh restore-iq` or `mud_autotrainer seating`
- **Method:** Short-burst SGD using `forge_autograd` tape-based backprop.
- **Optimizer:** Vanilla SGD with gradient clamping — **NOT AdamW**.
- **Key Parameters:**

| Parameter | Value | Source |
|-----------|-------|--------|
| Learning Rate | 0.002 | `auto_trainer.rs:L14` |
| Gradient Clamp | per-element `[-1.0, 1.0]` | `auto_trainer.rs:L544` |
| Gradient Sanitization | `is_finite()` check per element | `auto_trainer.rs:L544` |
| Grad Accumulation | 8 steps | `auto_trainer.rs:L18` |
| Layers per Token | 3 consecutive | `auto_trainer.rs:L16` |
| Max Facts per Cycle | 10 | `auto_trainer.rs:L341` |

---

## Two Training Paths

### A. Auto-Trainer (Hot Ternary SGD — `auto_trainer.rs`)
- **Scope:** Trains embedding table + MoE expert FFN weights (w1/w2/w3) + Mamba projections.
- **Data Source:** Unassimilated facts from `knowledge.db` (SQLite).
- **Forward Pass:** Embedding → 3 consecutive FFN/Mamba layers → logit projection → CE loss.
- **Loss:** Full Cross-Entropy (vocab ≤50k) or Contrastive with 5 negatives (vocab >50k).
- **Persistence:** Forced Hot PRQ re-quantization with per-row absmean scales → atomic `.mud` save.

### B. Corpus Aligner (Linguistic Recalibration — `corpus_trainer.rs`)
- **Scope:** Trains **only** the embedding table — no deep layers.
- **Data Source:** `.txt` files in `training/corpus/`.
- **Forward Pass:** Embedding → contrastive projection (7 negatives) → CE loss.
- **Learning Rate:** 0.001 (different from auto-trainer).
- **Persistence:** Saves embeddings as Float32 in `.mud` (⚠️ known issue: should re-ternarize).
- **Features:** Stateful resume (epoch/file/chunk), hard checkpoints every 5k chunks.

---

## Learning Marks System
Facts in `knowledge.db` are tagged with `learning_mark`:
- **0 (Raw):** Ingested but not yet distilled.
- **1 (Learned):** Integrated into weights via training.
- **2 (Master):** Critical verified knowledge.

---

## Health Monitoring

### Ternary Distribution Targets (BitNet 1.58b)
| Metric | Ideal | Healthy Range | Tool |
|--------|-------|---------------|------|
| σ (sigma) | ~0.735 | 0.50 – 0.80 | `training_health` |
| Fraction +1 | 37% | 25% – 49% | `training_health` |
| Fraction 0 | 26% | 14% – 38% | `training_health` |
| Fraction -1 | 37% | 25% – 49% | `training_health` |
| BUG-6 check | — | σ<0.3 ∧ zero>70% → COLLAPSE | `training_health` |
| Inter-layer σ | — | Δσ < 0.2 between adjacent layers | `training_health` |

### Gradient Safety (Mandated by GEMINI.md)
- All gradients MUST pass `is_finite()` check before application.
- All gradients MUST be clamped to `[-1.0, 1.0]` per element.
- Shadow weights in FP32 MUST be re-quantized via Forced Hot PRQ before disk save.

---

## Known Issues (Training-Specific)
| ID | Description | Status |
|----|-------------|--------|
| TRAIN-01 | PRQ scales ignored during expert dequantization | 🟢 RESOLVED |
| TRAIN-02 | Mamba in_proj backward pass is dead-end | 🟢 RESOLVED |
| TRAIN-07 | Corpus trainer: no gradient sanitization on target/neg updates | 🟢 RESOLVED |
| TRAIN-08 | Corpus trainer saves embeddings as Float32, not ternary | 🟢 RESOLVED |
| TRAIN-21 | Corpus trainer ignores embed_scales during load | 🟢 RESOLVED |
| BUG-6 | Weight decay may collapse ternary weights to zero | 🟢 RESOLVED |

---

## Phase Roadmap
1. **[DONE]** Automated hardware-to-architecture mapping.
2. **[DONE]** Fix state_dict size mismatch bugs in auto-config.
3. **[DONE]** Implementation of Per-Row Scaling (PRQ).
4. **[DONE]** Native Rust Trainer for direct knowledge assimilation.
5. **[DONE]** Gradient sanitization and Forced Hot PRQ (V4 Audit).
6. **[DONE]** Stateful resume and hard checkpoints in corpus trainer.
7. **[DONE]** Fix TRAIN-01/02/07/08/21 (see Audit V5).
8. **[ACTIVE]** Knowledge Distillation (KD) for stable ternary seating.
9. **[FUTURE]** Learning rate schedules (warmup + cosine decay).
