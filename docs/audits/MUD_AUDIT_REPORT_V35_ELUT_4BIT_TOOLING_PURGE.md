# MUD Audit Report V35: ELUT 4-bit Tooling Alignment and Dead Code Purge

**Date:** 2026-07-11
**Subject:** Tooling Synchronization with ELUT 4-bit Packing and SlimeRegister Dual-f16 Paradigm

## 1. Executive Summary

Following the discovery of the "Ternary Shock" pointer math bug during dequantization (where 4-bit packed weights were read using 2-bit offsets, causing memory overlaps), a system-wide audit of all tools, shaders, and diagnostic utilities was conducted. The audit revealed that several components were still assuming the legacy 2-bit packing format (`n / 16`) instead of the current ELUT 4-bit format (`n / 8`). Furthermore, strict compliance with **P-08 (Dead Code Elimination)** was enforced by removing over 20 obsolete, unregistered diagnostic binaries.

## 2. Critical Findings & Resolutions

### 2.1 Vulkan Inference Shader (Critical)
**File:** `assets/shaders/ternary_gemv_unified.comp`
- **Issue:** The compute shader used for GPU inference (`MUD_USE_VULKAN=1`) was iterating over blocks using `pcs.n_in / 16` and decoding via `(w_bits >> (i * 2)) & 3u`. This mismatch with the 4-bit ELUT format stored on disk generated garbage logits during hardware-accelerated inference.
- **Resolution:** The compute shader was updated to iterate using `pcs.n_in / 8`, decode 8 elements per `u32` block, and shift by 4 bits `(w_bits >> (i * 4)) & 0xFu`. 

### 2.2 Diagnostic Tooling Offsets
**File:** `tools/diagnose_model.rs`
- **Issue:** The offset logic used to inspect `token_embd.weight` manually evaluated the index as `token_id * (hidden_size / 16)`.
- **Resolution:** Corrected to `token_id * (hidden_size / 8)`.

### 2.3 FP32 Memory Transgressions in Telemetry
**File:** `tools/analyze_row_sums.rs`
- **Issue:** The tool assumed `output.weight` was an IEEE `f32` flat array and used `std::slice::from_raw_parts(*const f32)`. Reading packed Ternary2Bit data as floats generates entirely arbitrary statistics.
- **Resolution:** Updated to respect `MudTensorType::Ternary2Bit`. The tool now allocates an intermediate buffer and properly invokes `dequantize_ternary_row` to measure true activation magnitudes. *(Note: This file was subsequently deleted during the dead code purge, but its historical correction is noted here).*

### 2.4 SlimeBackward Documentation
**File:** `src/mud/slime_backward.rs`
- **Issue:** Safety documentation comments incorrectly stated the required buffer size as `(n_in / 16) * 4 * n_out`.
- **Resolution:** Updated comments to reflect the actual `(n_in / 8) * 4 * n_out` requirements of the ELUT format.

## 3. P-08 Dead Code Purge (Orphan Tools)

A massive accumulation of undocumented, unmaintained tools in the `tools/` directory was violating the **0-Warning, 0-Error** policy and contributing to architectural confusion. Specifically, tools that were not formally registered as `[[bin]]` entries in `Cargo.toml` were executing on outdated assumptions of the `SlimeRegister` (e.g., treating it as `i16` accumulators rather than the dual `f16` format).

**Deleted Files (21 files):**
- `analyze_row_sums.rs`
- `chat_telemetry.rs`
- `check_emb2.rs`, `check_emb3.rs`
- `cli_chat.rs`
- `doppler_radar.rs`
- `dump_norms.rs`
- `elut_bench.rs`, `lm_head_bench.rs`, `kernel_bench.rs`, `slime_backward_bench.rs`, `slime_bench.rs`, `galore_dora_benchmark.rs`
- `hub_api.rs`
- `list_tensors.rs`
- `mud_executable.rs`, `mud_os_forge.rs`
- `reset_weights.rs`
- `slime_parity_check.rs`
- `test_regex.rs`
- `wave_superposition_demo.rs`

## 4. Next Steps

With the architecture structurally sound and verified across `cargo test` and `cargo clippy`, the model's pipeline is fully repaired from the Ternary Shock regression. The engine is ready for the restart of the `smollm2.mud` training session using the UCP v2 procedure.
