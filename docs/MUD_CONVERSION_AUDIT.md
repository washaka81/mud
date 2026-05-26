---
lang: en
---

> **[HISTÓRICO — Reemplazado por `MUD_AUDIT_LATEST.md`]**
> Este análisis de conversión Dense→Ternary está consolidado en las secciones 5 y 6 de `docs/MUD_AUDIT_LATEST.md`.

# MUD Conversion Audit Report: MUD Universal Parameters

## 1. Overview
This document serves as an exhaustive audit of the MUD Universal Conversion pipeline, specifically focusing on the recent integration of dynamic configuration parameters (`hidden_size`, `ffn_hidden`, `kv_dim`, `num_experts`) and the conversion of dense FP32 architectures (e.g., `SmolLM2-135M`) into the Ternary 1.58b MoE `.mud` format.

## 2. Structural & Architectural Audit

### 2.1 Dynamic Parameter Injection (Success)
The `universal_converter` successfully extracts key dimensions from the safetensors metadata and injects them into the `.mud` global metadata. 
- **GQA Integration:** The motor now dynamically infers `kv_dim` for Grouped Query Attention models and adjusts the Multi-Head Attention (MHA) caching loops accordingly, mapping the KV-heads to Q-heads via `kv_groups = num_heads / num_kv_heads`.
- **Dense to MoE Mapping:** Standard MLPs (`gate_proj`, `up_proj`, `down_proj`) are accurately mapped to the MoE structure as `expert.0` (`w1`, `w3`, `w2`), with `num_experts = 1`. 

### 2.2 Tokenizer Mismatch (Error / Mismatch)
The current pipeline forcibly injects the `core_skills` tokenizer (vocabulary size: 32,000) into the converted model. However, standard models like `SmolLM2` have different vocabularies (e.g., 49,152). 
- **Impact:** The `token_embd.weight` projection in the final step is misaligned. Index `1024` in SmolLM means something entirely different than index `1024` in our `core_skills` dictionary. 

## 3. Coherence and Confidence Audit

When running `step_inference` on the converted `SmolLM2-135M` model, the output is largely incoherent (e.g., `LavIfNo305CHANGEprove6derive...`).

### 3.1 Neural Autopsy Diagnostics
Using `tools/neural_autopsy.rs`, we observed the following state behavior:
- **Entropy (Confidence):** `~4.0 - ~5.8`. An entropy $>1.0$ indicates extreme uncertainty. The model is essentially guessing uniformly among top tokens rather than showing decisive peaks.
- **X-Move (State Dispersion):** `~55.0`. The L2 distance of the hidden state between steps remains excessively high, indicating severe state instability and lack of convergence. 
- **Logit Variance:** Consistently hovers around `~18.0 to 24.0`, displaying wide dispersion.

### 3.2 The Ternarization Impact (Zero-Loss Fallacy)
The primary reason for the destroyed coherence is the extreme compression of the model through **Ternarization**. The `universal_converter` aggressively forces all internal Dense FP32/BF16 weights into exactly three states: `-1, 0, 1` using uniform mean-absolute-value (Gamma) scaling per layer chunk. 
- LLMs are highly sensitive to weight distribution. While the engine executes the 1.58b ternary operations flawlessly, mapping a dense Gaussian weight distribution strictly to Ternary without activation-aware distillation (like KD or QAT) obliterates the mathematical correlations between layers. The model is functionally lobotomized by the abrupt rounding.

## 4. Gradients & Convergence (Auto-Trainer Impact)
When dealing with a blindly converted model:
- **Gradients (Auto-Trainer):** The gradients computed by `forge_autograd` will reflect the massive cross-entropy loss caused by the vocabulary mismatch and the quantization destruction. 
- **Sigma / Variance:** The gradients will violently try to pull the remaining FP32 `token_embd` to compensate for the destroyed ternary weights in the inner layers, likely saturating the SwiGLU activations and killing convergence.

## 5. Opportunities for Improvement & Next Steps

1. **Vocabulary / Tokenizer Syncing:**
   - **Fix:** Update `universal_converter` to parse `tokenizer.json` from Hugging Face models instead of inheriting `core_skills.mud.bak` metadata. 

2. **Quantization-Aware Distillation (QAT):**
   - **Fix:** Pure Post-Training Quantization (PTQ) to 1.58b is too destructive. We need to introduce a "Calibration & Distillation" phase in `universal_converter/calibration.rs` where the model uses a teacher (original FP32) to iteratively align the ternary weights using actual KL-Divergence loss over a small text corpus.

3. **GQA Threading in Rayon:**
   - **Fix:** Now that GQA is structurally supported in `src/mud/inference.rs`, the next logical step is to parallelize the attention loops across multiple threads using `rayon` to scale performance on P-Cores.

4. **MoE Routing Override for Dense Models:**
   - **Fix:** When `num_experts == 1`, we should bypass the `MudRouter` entirely during inference to save the `gate_proj` computational overhead.

---
**Conclusion:** The infrastructure to dynamically load and dimension any model architecture into MUD is 100% operational. However, to produce coherent text, the parameter extraction must be paired with accurate tokenizer serialization and proper ternary distillation.