# Heterogeneous Multi-Processing (HMP): Asynchronous Vulkan Offload Plan

**Date:** 2026-06-30
**Target Architecture:** Intel Core i7-1260P (4 P-Cores, 8 E-Cores) + Intel Iris Xe Graphics + DDR4 2666MHz (Shared Memory).
**Core Problem:** In an integrated GPU environment, the GPU (Iris Xe) and CPU share the same memory bus (max ~42.6 GB/s). Executing memory-bound tasks (like standard GEMV QAT) simultaneously on both causes bus contention and slows down the system.
**Core Solution:** "Smooth but Powerful Harmony" — Isolate P-Cores for sequential, memory-bound Matrix Multiplications (AVX2), and offload strictly **compute-bound (O(N³))** or **asynchronous** isolated tasks to the Vulkan Shaders on Iris Xe.

---

## 1. Muon Optimizer Offload (Newton-Schulz)

### Rationale
The Muon optimizer performs 5 steps of Newton-Schulz orthogonalization on the gradient matrix: $X_{k+1} = 1.5 X_k - 0.5 X_k (X_k^T X_k)$.
This operation is incredibly dense mathematically (O(N³) FLOPs) but operates on relatively small, stationary matrices (e.g., 576x576 or 2560x2560). It is heavily *compute-bound* and perfectly suited for the Iris Xe execution units.

### Implementation Plan
- **Module:** `src/mud/muon.rs` and `src/mud/qat_dispatcher.rs`.
- **Shader:** Create `newton_schulz.comp`, utilizing `subgroupAdd` for the dot products and matrix transpositions.
- **Data Flow:** 
  1. CPU P-Cores compute the raw gradient matrix during the Backward pass.
  2. CPU hands off the gradient matrix buffer to `VulkanContext`.
  3. CPU proceeds with the Forward pass of the next token/batch.
  4. Iris Xe runs the 5 iterations of Newton-Schulz asynchronously in VRAM cache.
  5. Iris Xe returns the "purified" orthogonalized gradient for the final Adam/SGD update.

---

## 2. Real-Time Telemetry & Thermodynamic Reductions

### Rationale
Extracting metrics (`VarH`, `VarJ`, `Z_Entrop`, covariance matrices, and `Sat%`) requires reading the massive activation tensors (`SlimeWorkspace`) and performing variance/mean reductions. Doing this on the CPU interrupts the critical path and burns cycles that should be doing AVX2 GEMV.

### Implementation Plan
- **Module:** `src/mud/vulkan.rs` and `tools/train_telemetry.rs`.
- **Shader:** Create `tensor_thermodynamics.comp`.
- **Data Flow:**
  1. After every `evaluate_slime_block`, the CPU maps the `SlimeWorkspace` registers to a Vulkan Subbuffer.
  2. The Iris Xe Shader uses Subgroup reductions to calculate Mean, Variance, and Entropy in a single fast parallel sweep.
  3. Vulkan writes the scalar results (a few bytes) back to host memory.
  4. The Agentic UI (`crossterm`) reads these scalars to update the dashboard without slowing down the P-Cores.

---

## 3. DSpark: Asynchronous Speculative Drafter

### Rationale
Speculative Decoding (DSpark - Priority 39) relies on a smaller "drafter" model predicting K future tokens. Running the drafter on the CPU would steal AVX2 cycles from the main verification model.

### Implementation Plan
- **Module:** `src/mud/speculative.rs`.
- **Shader:** Implement a lightweight 2-layer `ternary_gemv.comp` strictly for the drafter model.
- **Data Flow:**
  1. The Iris Xe loads the mini-drafter weights (very small footprint, easily cached).
  2. Iris Xe autoregressively generates $K$ candidate tokens in the background.
  3. The P-Cores take the $K$ candidates and run a single parallel Forward pass to verify them.
  4. This decouples the generation bottleneck. The GPU "guesses" while the CPU "verifies", increasing throughput by ~80%.

---

## Phased Rollout Schedule
- **Phase 11.A:** Integrate `core_affinity` for P-Cores and disable Vulkan for QAT (Completed).
- **Phase 11.B:** Implement `newton_schulz.comp` and wire `muon.rs` to Vulkan (Completed).
- **Phase 11.C:** Implement `tensor_thermodynamics.comp` for asynchronous telemetry (Completed).
- **Phase 11.D:** Port DSpark drafter logic from CPU to Vulkan (Completed).
