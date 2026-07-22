# MUD Engineering Guidelines: Memory & Performance
**Last Update:** 29 de mayo de 2026

## 1. Zero-Allocation Mandate
No code in `src/mud/inference.rs` should call `Vec::new()`, `Box::new()`, or any allocating function during the `step()` call.
- **Solution:** Use `SlimeWorkspace` pre-allocated buffers (see `src/mud/slime.rs`). Historical `InferenceWorkspace` was removed (L-03 / P-08).
- **Validation:** Monitor RSS memory growth during long chat sessions. Memory footprint MUST remain constant (O(1)) during Mamba sequence generation.

## 2. Pointer Arithmetic Standards
Ternary weights are packed 16 per `u32`. This creates strict alignment requirements.
- **Row Step:** Always use `(n_in + 15) / 16` to calculate the number of blocks per row.
- **Safety:** Use `std::ptr::null()` for missing optional weights (like expert gates in dense models or convolution biases).
- **Checks:** Use `mud_diagnostics` to verify pointer offsets across deep hybrid stacks.

## 3. SIMD, ASM & Asynchronous GPU Heartbeat
Assembly kernels and GPU shaders are the heart of the engine.
- **Register Hygiene:** Always call `vzeroupper` before returning from an ASM function to avoid AVX-SSE transition penalties.
- **Mamba Parallel Scan:** Use the specialized `mamba_scan_avx2` kernel for processing the recurrent states. Do not attempt scalar recurrence in Rust.
- **Asynchronous Heartbeat:** To keep the Vulkan iGPU alive without blocking the CPU's 160 TPS sequence loop, delegate heavy projections (e.g. `Mamba out_proj`, `Attention out_proj`) to the GPU asynchronously (`is_async: true`). The CPU must immediately continue to the next layer's setup.

## 4. Quantization: PRQ (Per-Row Quantization)
Global scaling is deprecated for all architectures >12 layers.
- **Standard:** Every ternary tensor `.weight` must have a corresponding `.scale` tensor.
- **Dimensionality:** If weight is `[rows, cols]`, scale MUST be `[rows]`.
- **CPU Loop:** Apply scales after ASM accumulation but before writing back to the output buffer to maximize cache hits.

## 5. Universal Agnosticism & Justified Constants
The engine MUST remain agnostic to specific model architectures.
- **Dynamic Meta:** Derive all parameters (context length, hidden size, heads) from model metadata.
- **Justification:** Every constant used in the engine (e.g., dampening, sparsity ratios, neural kick) MUST be justified in the code and documentation.
- **Standard Values:** Use mandated values from `GEMINI.md` (`0.7071` for dampening, `0.7` for sparsity, `1e-5` for jitter) to ensure cross-model mathematical consistency.

## 6. Hot Ternary SGD & Autograd
The `mud_autotrainer` implements a specialized memory-mapped SGD.
- **Shape Matching:** When adding new layers to `forge_autograd`, ensure dimensions strictly match between forward pass projections and backward gradient accumulations (especially in non-square transformations like Mamba's `in_proj`).
- **Gradients Accumulation:** Use the `GradAccum` structs to prevent cloning vectors into the computational tape, preserving the zero-allocation philosophy even during training.

## 6. Documentation Discipline
- **Synchronization:** When a core engine parameter changes (e.g. alignment logic, hybrid ratios), update `MUD_ROADMAP.md` and `GEMINI.md` immediately.
- **Version Control:** Tag major architectural shifts in `docs/MUD_CRITICOS_MAXIMOS.md`.

---
*Precision is a Choice. Performance is a Mandate.*
