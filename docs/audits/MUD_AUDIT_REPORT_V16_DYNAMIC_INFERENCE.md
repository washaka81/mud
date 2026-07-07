# MUD Audit Report V16: Dynamic Inference & Universal Agnosticism

## 1. Context & Motivation
Following the **Zero-Hardcoding Mandate** and **Universal Agnosticism**, it was identified that `mud.sh` and the main inference engine were manually defaulting to generic prompts or incorrectly assuming architectural formats (like LLaMA-3 vs ChatML). Additionally, the ternary format compilation failed due to pointer casting regressions during workspace separation.

## 2. Issues Identified
1. **Compilation Failures**: Resolving the inference loops (`src/mud/forward.rs` and `src/mud/sampling.rs`) caused `cargo` to lose type inference bounds, resulting in E0282 and E0689.
2. **Pointer Arithmetic Ambiguity**: Reading FP16 bytes with a `u32` pointer in the projection layers crashed the engine due to unaligned memory and byte formatting.
3. **Prompt Template Hardcoding**: The interactive CLI in `src/main.rs` sent raw text directly without configuring the chat interaction tokens (`<|start_header_id|>`, `<|im_start|>`, etc.), resulting in poor model reasoning because it didn't recognize it was in a chat context.

## 3. Resolution & Architectural Pivot
### 3.1 Strict Typing
Added strict type bounds (`f32::`) to all tensor math functions across the entire inference codebase to prevent any ambiguous assumptions by the Rust compiler. This permanently restored the **0-error, 0-warning** strict policy.

### 3.2 Dynamic Inference Vocabulary Interrogation
Instead of relying on statically baked templates (which were lost during Universal Conversion) or hardcoding fallback prompts, the MUD engine now natively auto-detects its environment:
- It interrogates `engine.tokenizer.vocab.contains_key()` directly.
- If it finds `<|start_header_id|>`, it injects LLaMA-3/BitNet structures.
- If it finds `<|im_start|>`, it applies ChatML (Qwen/Mistral) logic.
- If it finds `[INST]`, it applies LLaMA-2 format.
- Otherwise, it falls back to a base User/Assistant generic layout.
This firmly anchors the MUD engine as **Architecturally Agnostic**.

### 3.3 Deep QAT Autotraining
The latest conversion resulted in severe **Linguistic Aphasia** (Ternary Shock) because the fast L-QAT process (2 epochs) wasn't enough to reseat the BPE embeddings after 1.58-bit ternary mapping.
We have initiated a deep **5-Epoch Full QAT** on the corpus via `mud_corpus_trainer --full-qat` to permanently reseat the vocabulary and solve the semantic output issues.

## 4. Next Steps in the Roadmap
1. Validate output coherence after Deep QAT.
2. Ensure the Vulkan iGPU pipeline is accurately reading the dynamic prompt template.
3. Verify the `diffusion_mode` also utilizes the autodetected vocabulary template.
