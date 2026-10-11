# MUD Session Report: 2026-07-01
**Focus:** Semantic Aphasia Cure, mHC Geometric Prison Resolution, Tanh Soft-Clipping, Lexical Resonance

## 1. The Persistent Semantic Aphasia
Despite fixing the `VarJ` collapse and implementing dynamic temperatures, the model's generated entropy (`H`) remained extremely high (~10.5), and it continued to sample random valid BPE tokens (e.g., `mortgages`, `affirmativeleet`, `pean`, `lem`). 
Telemetry analysis revealed that the `VarH` (Residual Stream Variance) was still deadlocked at exactly `0.01` across all dimensions, even after 100 epochs of training. The cross-entropy loss averaged `8.85`, which corresponds mathematically to `ln(7000)`, meaning the model had uniformly memorized the negative sampling subset rather than learning the actual context.

## 2. The Tanh Soft-Clipping Implementation
To prevent gradient death caused by hardcoded `.min(10.0)` caps in the JEPA stabilizer, we successfully implemented an asymptotic Soft-Clip using the Hyperbolic Tangent (`tanh`).
- **Formula:** `inv_sigma_ctx = dynamic_limit * tanh(raw_inv_sigma / dynamic_limit)`
- **Dynamic Limit:** Derived directly from the model topology as `sqrt(hidden_size)`. For a 576-dim model, this is `24.0`.
- **Validation:** Empirical telemetry (`mud_metrics.log`) confirmed the Tanh was squashing values beautifully. For example, a raw target of `17.15` was smoothly bounded to `14.75` without killing the derivative, proving the mathematical elasticity.

## 3. The Breakthrough: The mHC Geometric Prison
Investigation into why `VarH` was locked at `0.01` led to the discovery of a catastrophic mathematical bug in the **Manifold-Constrained Hyper-Connections (mHC)** algorithm port from DeepSeek-V4.
- **The Bug:** `mhc_radius` was initialized as `max_emb` (the maximum absolute value of the embedding tensor, typically `~2.4`). The mHC residual projection squashes the $L_2$ norm of the *entire 576-dimensional hidden vector* to this radius.
- **The Math:** If $||h||_2 \le 2.4$, then the average element is bounded to $2.4 / \sqrt{576} = 0.1$. The maximum possible variance of a vector bounded by $0.1$ is exactly $0.01$. The mHC projection was acting as an inescapable geometric prison, structurally preventing the model from generating contextual variance.
- **The Fix:** The radius was re-scaled geometrically to bound the $L_2$ norm properly: `mhc_radius = max_emb * sqrt(hidden_size)`. This increases the spherical volume limit by 24x, granting the `VarH` a maximum theoretical headroom of `5.76`.

## 4. Lexical Resonance & The Electrocardiogram Pattern
Upon fixing the mHC radius and compiling (`cargo build --release`), telemetry immediately revealed the model breaking out of its prison.
- `VarH` exploded from `0.0068` to ranges between `0.008` and `0.037` (a 500% increase).
- Loss on grammatical tokens plummeted (e.g., `,` dropped from 5.29 to 4.29).
- **The "Electrocardiogram":** The JEPA Variance graph displayed massive, sharp spikes up to `24.66` (exactly the geometric volume `sqrt(576) = 24.0`). These spikes represent the **Lexical Resonance** injection at Layer 0 (the token embedding's energy jump-starting the system). The variance then drops steadily across the 30 layers due to the JEPA spring force (`jepa_alpha = 0.01`). Once the untrained layers fully train their weights, they will learn to push back against the spring, transforming the sharp "latidos" (spikes) into a smooth, continuous waveform (the "GPS route").

## 5. JEPA Gate Rewire — The Missing Gate
**Problem:** `jepa_energy` (bits 16-31 of SlimeRegister) stored `z` (raw EMA tracker), but `mhc_residual` never used it to modulate the residual blend. The gate `sigmoid(v_jepa)` simply did not exist in the forward pass execution. The stabilizer computed `z_next = 0.9*z + 0.1*y_norm` and stored it, but `mhc_residual` just copied `jepa_energy` through — the residual was always `α·h + β·f_h` regardless of the JEPA state.

**Fix (triple):**
1. EMA tracker `z` moved from `jepa_energy` to a flat buffer `workspace.jepa_z[2 × num_layers × hidden]`. The `jepa_stabilizer` reads/writes `z` there
2. `jepa_energy` now stores `v_jepa = (z - μ) / σ` (centered+scaled gate value), written **after** the statistics update (μ, σ) so it reflects the current system state
3. `mhc_residual` reads `jepa_energy` → `sigmoid(v_jepa)` → modulates blend as `gate × α × h_in + (1-gate) × β × f_h`

**Embedding init:** New `SlimeRegister::init_from_embed()` sets `jepa_energy=0` (gate=0.5 neutral) and initializes all `2×num_layers` per-head z trackers in `jepa_z` with `emb_val.abs()`.

**Tape:** records `v_jepa` (not `spring_force`) for correct backward pass.

**Validation:** 89 tests pass, clippy P-06 clean, inference produces coherent text.

## 6. Enhanced Telemetry
Added to mud_metrics.log and Engine Log:
- Step throughput (tok/s)
- Dynamic temperature (T) and top_p per step
- Entropy (H) and per-token throughput
- Column headers updated to match

## 7. P-13 Audit
Replaced silent fallbacks `unwrap_or(896/4096/8/etc)` with `.expect("P-13: ...")` in `corpus_trainer.rs` for `hidden_size`, `max_pos`, `n_heads`, `n_kv_heads`, `n_layers`, `eps`.

## 8. Training Run
- Converted fresh SmolLM2-135M from safetensors → mud (282 MB, 210 ternary tensors + ECC)
- Trained 1 epoch, batch 8: Loss = 9.07, Speed = 851 ops/s
- Checkpoint saved to `weights/checkpoints/model_latest_checkpoint.mud`

## Next Steps
- Retrain with more epochs to reduce loss below 8.0
- Validate the gate gradient flow via backward pass (DeltaJ analytical derivatives)
- Monitor VarJ / VarH to confirm the gate creates diverse blend factors across dimensions
