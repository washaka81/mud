---
lang: en
---

# MUD Technical Spec: Ternary KV Cache (Working Memory)

## 1. Overview
The **Ternary KV Cache** is MUD's short-term "working memory." It allows the model to maintain context across a long sequence of tokens (up to 4096) by storing the Key (K) and Value (V) vectors of every processed token.

## 2. Architecture
Unlike standard engines that use FP16 or FP32 for the KV Cache, MUD utilizes a **Hybrid Cache**:
- **Activations:** Kept in FP32 during the current token's computation for maximum precision.
- **K/V Storage:** Can be dynamically quantized to **INT8** or kept as **FP32** depending on hardware pressure.
- **Attention Type:** Grouped-Query Attention (GQA) to save memory while maintaining multi-head quality.

## 3. Integration with Ternary Weights
The attention mechanism in MUD bridges the **Ternary Weights** (used for projections) and the **Linear Attention** (used for memory retrieval).
- **Q, K, V Projections:** Executed via `ternary_gemv_avx2` (Additions/Subtractions only).
- **Positioning:** Rotary Position Embeddings (RoPE) are applied to Q and K to ensure MUD understands the order of words.

## 4. Impact on Coherence
Without this cache, MUD has "amnesia" and repeats itself. With the KV Cache:
- **No more loops:** The repetition penalty can look back at the actual generated history.
- **Grammar Stability:** MUD can finish a sentence started 20 words ago.
- **Bilingual Flow:** Maintains the chosen language's structure by referencing previous tokens.

## 5. Memory Footprint (MUD 512 Hidden Size)
- **K-Cache:** 1 layer * 512 dim * 4096 tokens * 4 bytes ≈ 8 MB.
- **V-Cache:** 1 layer * 512 dim * 4096 tokens * 4 bytes ≈ 8 MB.
- **Total:** ~8 MB per layer. Extremely efficient on the Intel i7-1260p.
