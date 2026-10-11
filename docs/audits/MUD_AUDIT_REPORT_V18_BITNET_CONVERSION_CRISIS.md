# MUD Audit Report V18: BitNet Conversion Crisis & Semantic Aphasia
**Date:** 15 de junio de 2026
**Status:** CRITICAL FAILURE | NO COHERENCE

## 1. Executive Summary
The BitNet 2B model, despite passing mathematical audits with a score of 112%, suffers from total Semantic Aphasia during live inference. This has been traced to a critical architectural discrepancy in how GQA (Grouped Query Attention) models are handled in the inference engine, specifically regarding the KV cache stride and memory safety.

## 2. Identified Conversion & Engine Problems

### A. KV Cache Stride Mismatch (Engine)
- **Problem:** The engine assumes `hidden_size` as the stride for the KV cache across all models.
- **Impact:** For GQA models like BitNet (5 KV heads vs 20 Q heads), the KV cache stride (2560) is 4x larger than the actual projection output (640). 
- **Consequences:** 
  1. **Buffer Overflow:** `ptr::copy_nonoverlapping` reads 2560 elements from a 640-element buffer.
  2. **Data Displacement:** Attention mechanism reads garbage from incorrect offsets.
  3. **Corruption:** Hidden states are corrupted in the hot-loop, causing token soup.

### B. Sub-Norm Regression (Conversion/Engine)
- **Problem:** "Audit V12" mandated bypassing `attn_sub_norm` and `ffn_sub_norm`.
- **Impact:** BitNet 2B uses these layers with non-identity learned weights. Bypassing them destroys activation magnitude stability.

### C. Logits Scaling Paradox (Tied Weights)
- **Problem:** Tied embeddings sharing Float32 memory might produce large logit magnitudes without proper temperature or scale adjustment.

## 3. Technical Verification (Llama-3/BitNet)
| Parameter | Value | Impact |
| :--- | :--- | :--- |
| `num_attention_heads` | 20 | Correct |
| `num_key_value_heads` | 5 | **STRIDE ERROR** (Mismatch with Q-stride) |
| `head_dim` | 128 | Correct |
| `KV Stride` | 2560 | **CORRUPTION** (Should be 640) |

## 4. Remediation Mandate
- [ ] **KV Stride Normalization:** Refactor `inference.rs` and `forward.rs` to use `n_kv_heads * head_dim` as the stride for KV cache.
- [ ] **Rollback Audit V12:** Restore mandatory `SubLN` processing for BitNet architectures.
- [ ] **Standardize Epsilon:** Fix all divisional floors to `1.1e-8` for boundary consistency.

---
*Audit conducted by Gemini CLI - MUD Engine Core.*
