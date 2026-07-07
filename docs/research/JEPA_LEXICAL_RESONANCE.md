# Research Note: Lexical Resonance (Semantic Attractor) for JEPA

**Date:** 2026-06-19
**Author:** Forge LLM Architecture Team
**Status:** PROPOSED (Roadmap Phase 7)

## The Problem: Ternary Shock and Semantic Aphasia
When updating the inference engine to the new **SlimeRegister Paradigm** (with multiplicative JEPA gating), the ternary weights (\(\{-1, 0, 1\}\)) lose their original alignment. Because the JEPA gate is purely statistical (based on numeric variance `VarH` of the outputs), it is "blind" to the actual semantic meaning of the token being processed. 

During the initial QAT (Quantization-Aware Training) epochs, this blindness results in **Semantic Aphasia**:
- `VarH` collapses to ~0.01.
- `Sat%` drops to 0.00%.
- Cross-Entropy Loss explodes to ~18.0 (worse than random uniform guessing, \(\ln(128256) \approx 11.76\)).

The model is forced to blindly rediscover the lexicon by slowly adjusting the weights over hundreds of epochs.

## The Proposal: Lexical Resonance (Lexical Prior)
Instead of starting the JEPA state (`mu_ctx` and `var_ema`) empty or purely based on layer-0 numeric output, we propose injecting the **Lexical Energy** (the semantic fingerprint of the token) directly into the JEPA gate at the very start of the forward pass.

### Mechanism
In `SlimeWorkspace`, before evaluating Layer 0, the `jepa_packed` state of each register would be initialized using the magnitude of the original token embedding:

```rust
// Current (Blind) Approach:
ws.registers[h].matmul_accum = (shadow_emb[emb_start + h] / iscale) as i16;
ws.registers[h].jepa_packed = 0; // Starts with zero semantic bias

// Proposed (Lexical Resonance) Approach:
let lexical_energy = shadow_emb[emb_start + h].abs(); 
ws.registers[h].jepa_packed = float_to_f16(lexical_energy);
```

### Theoretical Advantages
1. **Immunity to Aphasia:** The JEPA gate acts as a semantic "lighthouse". Even if the ternary layers produce noise, the multiplicative gate will suppress any activations that do not resonate with the original token's lexical energy.
2. **Hyper-Fast Convergence:** QAT would no longer need 200 epochs to blindly rediscover the vocabulary. The gradients would be guided immediately by the Lexical Prior, potentially dropping the loss below 11.76 in the first 5 epochs.
3. **Continuous Semantic Awareness:** Deep layers (e.g., Layer 29) often suffer from "context collapse" where they forget the specific input token. Lexical Resonance ensures the pure semantic intent is carried unchanged in the bits 16-31 of every `SlimeRegister` throughout the entire forward pass.

## Next Steps
This concept has been scheduled for the next architectural roadmap (Phase 7). Implementation will require:
1. Refactoring `SlimeWorkspace::clear_registers()` to accept a `lexical_prior` vector.
2. Updating `corpus_trainer.rs` and `slime_forward.rs` to compute and pack `lexical_energy` into the `f16` slot of the `SlimeRegister`.
3. Verifying that the Lexical Prior does not overly constrain the model from learning long-range dependencies (preventing the upper layers from attending to other tokens).
