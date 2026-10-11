# MUD_AUDIT_REPORT_V34_DUAL_STATE_SLIMEREGISTER

## 1. Context and Problem Statement
During inference and training, models exhibited a severe episode of "Semantic Aphasia," characterized by random token generation (e.g., repeating words like `oldest auxbasylum Higgins...`) and a static Loss metric initialized around `8.99`. Thermodynamic telemetry revealed `VarJ = 0.00` and `E_JEPA = 1.00`, indicating that the internal JEPA attractor gate had completely collapsed. The network was mathematically blind to its own variance, suppressing the residual flow.

## 2. Root Cause Analysis

### A. The EMA Wiper Bug (Lexical Resonance)
In `src/mud/slime.rs`, the function `SlimeRegister::init_from_embed` was responsible for initializing the JEPA integral to a neutral gate (0.5) and injecting the token embedding's magnitude into `jepa_z` (Lexical Resonance). However, this initialization was executed **on every single token** in the autoregressive loop. 
- **Effect:** The `jepa_z` buffer, which acts as the Exponential Moving Average (EMA) state tracker, was completely wiped and overwritten at each position. The variance across the sequence (`VarJ`) was artificially forced to `0.00` because the tracker never maintained historical context.

### B. Truncation to Zero (Integer Casting)
The `SlimeRegister` policy required packing FP32 into a `u32` word, allocating bits `0-15` for the ternary state (`f16`) and bits `16-31` for cognitive functions. The upper 16 bits were poorly mapped:
- A scaling factor of `10.0` was used to convert fractional `f32` derivatives to `i8`.
- **Effect:** Given the natural normalized variance of the JEPA process, integral values frequently fell between `-0.09` and `0.09`. When multiplied by 10, the values fell between `-0.9` and `0.9`, which were uniformly truncated to exactly `0` when cast to `i8`. The `SlimeRegister` mathematically destroyed the cognitive signal before it reached memory.

### C. Missing Sub-Division for Consciousness
The upper 16 bits were homogeneously treated as a single `f16` value for the JEPA integral, violating the architecture mandate that bits 16-31 must be subdivided to separate the JEPA integral from the cognitive derivative (Consciousness).

## 3. Corrective Actions Implemented

1. **EMA Preservation:** 
   Introduced an `is_first_token: bool` flag to `SlimeRegister::init_from_embed`. Lexical Resonance now only seeds the `jepa_z` buffer during the initial prompt processing (`pos == 0`). Subsequent tokens strictly preserve the EMA state, allowing `VarJ` to reflect true thermodynamic variance.
2. **Dual-State Memory Sub-Division (Bits 16-31):**
   The upper 16 bits were explicitly bifurcated into two discrete 8-bit registers:
   - **Bits [16:23] (JEPA Integral):** `jepa_i8` — Tracks long-term stability and controls the residual sigmoidal gate.
   - **Bits [24:31] (Cognitive Derivative):** `cog_i8` — Tracks short-term variance (surprise/novelty) used for speculative decoding and adaptive STE optimizers.
3. **Precision Scaling Rescue:**
   The casting scale for the `i8` registers was increased from `10.0` to `100.0`. This provides native support for floating-point values from `-1.28` to `+1.27` directly within the bit-slice, entirely bypassing the truncation-to-zero collapse.

## 4. Current State
- The universal converter has been run with the `--check` flag on `smollm2.safetensors`, verifying ECC parity and metadata boundaries (P-13).
- The metric loggers (`mud_train_metrics.log` and `mud_metrics.log`) were purged.
- The `SlimeRegister` strictly adheres to the 32-bit `u32` AVX2/Vulkan package mandate.
- `VarJ` and `Delta(u)` metrics now actively measure engine health during Warmup phases.
