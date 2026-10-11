# ASM Pointer/Register Calculus & SmartTokenizer Design

## 1. Problem Statement

The MUD engine runs three tightly-coupled computational regimes through a shared set of AVX2 assembly kernels:

| Regime | Nature | Calculus | Example Kernels |
|--------|--------|----------|-----------------|
| **GEMV (FP32/Ternary)** | Deterministic | Fixed outer product + linear map | `dot_product_avx2`, `ternary_gemv_avx2`, `f32_gemv_rowwise` |
| **STE (QAT)** | Probabilistic | Hard threshold + identity-pass gradient | `pack_elut_prq`, `pack_ternary_into`, `apply_optimizer_cpu_step_and_pack` |
| **JEPA / mHC** | Stochastic | OU process + EMA statistics + gate modulation | `jepa_stabilizer`, `mhc_scale_sgd_step` |
| **CMUD / Thinker** | Complex-valued | Phase-coherent attention + Hermitian projection | `think_step_phase_attn`, `cmud_backward` |

The challenge: **each regime has different data-layout, pointer-arithmetic, and register-pressure requirements.** An "intelligent tokenizer" must bridge the gap between the text→token interface and these low-level backends, producing data in the exact memory format each kernel expects — with zero-copy, zero-alloc, and cache-optimal alignment.

---

## 2. ASM Kernel Register Analysis

### 2.1 Full Register Map

Every ternary/FP32 kernel uses a consistent **ELUT decode pipeline** (branchless):
```
u32 word (8 nibbles) → vpsrlvd (per-lane shift) → vpand (mask 0x0F) → 
  vpcmpeqd (+1 check) / vpcmpeqd (-1 check) → vpand (weight gating) → 
  vaddps / vsubps (accumulator update)
```

**Register budget per kernel:**

| Kernel | YMM used | Scalar GPR | Accumulator strategy | Spill? |
|--------|----------|------------|---------------------|--------|
| `ternary_gemv_avx2` | 9 (0,8-15) | RBP,RSI,RDI,RDX,RCX | 4 parallel accumulators (0,9,13,14) | No |
| `ternary_gemv_4rows_avx2` | 11 | RBP,R12-R14 | 4 row-acc (0-3) + scratch (4-8) | No |
| `ternary_gemv_8rows_avx2` | **16 (all)** | RBP,RBX,R10-R15 | 8 row-acc (0-7) | **All YMM used** |
| `dot_product_avx2` | 4 (0-3) | RBP,RSI,RDI,RDX | Dual FMA acc (0,3) | No |
| `silu_vectorial_avx2` | **16 (all)** | RBP | Full polynomial pipeline | **All YMM used** |
| `adam_step_avx2` | **16 (all)** | RBP,RBX,R12 | 8 broadcasts + 4 loads + 4 scratch | **All YMM used** |
| `rms_norm_scale_asm` | 2 (1,2) | RBP | Dual-accum sum-of-sq | No |
| `lm_head_logits_avx2` | 4 (0-3) | RBP,R8,R9,R12-R15,RBX | Dual-accum FMA per row | No |

### 2.2 Register pressure hot spots

The 8-row kernel (`ternary_gemv_8rows_avx2`) uses **all 16 YMM registers** — there is zero spill room. This means:

1. **No prefetch-in-registers**: Only 3 NTA prefetches for 8 weight streams (would need 8 to be optimal)
2. **No NaN guard in vector**: The per-lane NaN guard requires 2 extra YMM (mask + allones) — impossible at 16/16
3. **Scalar NaN guard only**: Post-reduction `vucomiss` on each row accumulator

**Implication for the tokenizer**: When producing data for the 8-row kernel:
- Input activation arrays (`x_f32`) should be hot in L1 (prefetched 4 cache lines ahead)
- Weight data should be prefetched NTA (never re-read)
- Output rows should be written sequentially (no scatter)

---

## 3. Pointer Arithmetic Patterns

### 3.1 Three canonical pointer idioms

**Pattern A — usize-capture closure (PCorePool dispatch):**
```rust
let acts_p = acts_f32.as_ptr() as usize;
let w_p = w_u32 as usize;
pool.execute(move || {
    let acts = acts_p as *const f32;
    let w = w_p as *const u32;
    let row_ptr = w.add(row * row_u32s);
    // AVX2 kernel reads from row_ptr, acts
});
```

**Why**: `as_ptr() as usize` avoids lifetime constraints when moving into `FnOnce` closures sent to the thread pool. The usize is a plain integer that doesn't borrow.

**Pattern B — Workspace raw-pointer borrow:**
```rust
let regs = ws.registers.as_mut_ptr();
let z = ws.jepa_z.as_mut_ptr();
// SAFETY: jepa_stabilizer writes into regs[i].jepa_energy
```

**Why**: The `SlimeWorkspace` owns all buffers. The hot path accesses them through `*mut` pointers to avoid bounds checks and aliasing analysis in the compiler.

**Pattern C — Copy-free row dispatch:**
```rust
// In f32_gemv_rowwise:
let ws = workspace.as_mut_ptr();
let acts = ws.add(OFFSET_ACTS) as *const f32;
let out = ws.add(OFFSET_OUT) as *mut f32;
```

**Why**: Pre-allocated workspace means the GEMV output goes directly into the next layer's input buffer — no intermediate allocation.

### 3.2 Data layout requirements per regime

| Regime | Input layout | Weight layout | Output layout | Alignment |
|--------|-------------|---------------|---------------|-----------|
| **Ternary GEMV** | f32 contiguous `[n_in]` | u32 packed-ternary `[n_in/8]` per row | f32 contiguous `[n_out]` | 32-byte YMM |
| **F32 GEMV** | f32 contiguous `[n_in]` | f32 contiguous `[n_in * n_out]` | f32 contiguous `[n_out]` | 32-byte YMM |
| **JEPA** | f32 hidden `[hidden]` | z_buf f32 `[hidden]`, mu/sigma scalar | f32 gate per element | 8-byte register |
| **CMUD** | f32 hidden `[hidden]` | phase f32 `[hidden]` (phi array) | f32 complex `[2*hidden]` | 32-byte YMM |
| **STE pack** | f32 shadow `[n]` | packed u8 `[n/4]`, scale f32 `[n_row]` | u8 packed + f32 scales | None (byte) |

---

## 4. Probabilistic Calculus: STE Deadzone

The Straight-Through Estimator creates a **discontinuity surface** in the weight space:

### 4.1 Forward (quantization)

```
if |w_shadow| > PRQ_scale * THRESHOLD_RATIO (0.7) → ternary ±1
else → ternary 0 (deadzone)
```

The deadzone region `|w| ∈ [0, PRQ_scale × 0.7]` produces **zero gradient** in the forward pass — any weight movement inside this zone has no effect on the output.

### 4.2 Backward (identity-pass)

```
∂L/∂w_shadow = ∂L/∂w_ternary  (STE identity approximation)
```

The gradient flows **through** the quantization as if it didn't exist. This is the "straight-through" part — it's a deterministic approximation of a probabilistic process.

### 4.3 Optimizer interplay

```
Adam step on w_shadow → clamp ±MUD_TRAIN_WCLAMP_K → pack_elut_prq
```

The optimizer (Adam/Muon) updates `w_shadow` in full FP32 precision. The clamp prevents scale explosion (historical fix for PRQ-scale inflation). The pack step re-quantizes.

### 4.4 Tokenizer role

The SmartTokenizer contributes to this calculus by:

1. **Token frequency embedding scale**: Tokens that appear rarely have smaller gradients → their embeddings stay near zero longer → STE deadzone keeps them at 0 ternary value for more steps.
2. **Per-token LR modulation**: The tokenizer provides `vocab_size metadata` so the optimizer can scale learning rate by inverse token frequency.
3. **Embedding row pointer**: `encode_raw()` returns `(*const u32, len)` — the token ID array that indexes directly into `token_embd.weight` rows.

---

## 5. Deterministic Calculus: FMA Dot Product

The deterministic path through `dot_product_avx2`:

```asm
; Main loop: 16 floats per iteration (2 YMM loads)
vfmadd231ps  ymm0, ymm2, ymm3   ; acc0 += a[0..7] * b[0..7]
vfmadd231ps  ymm1, ymm2, ymm3   ; acc1 += a[8..15] * b[8..15]
```

**The dual-accumulator pattern** hides FMA latency (4 cycles on i7-1260P):
- ymm0 accumulates first 8 elements
- ymm1 accumulates second 8 elements
- Horizontal sum at end: `vextractf128 + vaddps + vhaddps × 2`

**Implication for tokenizer**: The embedding GEMV (one per token) is identical in both ternary and F32 modes:
1. Token ID → row offset into embedding matrix
2. Row GEMV produces hidden_size f32 outputs
3. Output goes directly into `SlimeRegister.matmul_accum`

---

## 6. JEPA Probabilistic Dynamics

### 6.1 OU Process with Adaptive Alpha

```rust
// y_norm = z-score of hidden state
// alpha = clamp(|y_norm - z| / 2.0, 0.01, 0.10)
// z_next = z * (1-alpha) + alpha * y_norm + jitter(1e-4)
```

The **adaptive alpha** makes the tracker respond faster to large jumps and slower to noise. This is a pure stochastic differential equation in discrete time.

### 6.2 mHC Gate

```
gate = sigmoid(v_jepa) where v_jepa = (z - mu) * inv_sigma
out_i = alpha_i * h_in_i + (1 - gate_i) * beta_i * f(h)_i
```

When `v_jepa` is large positive (hidden state far from its EMA), `gate → 1`, and the residual connection bypasses the block output. This is a **probabilistic gating mechanism** — the same input can produce different gate values depending on the EMA state.

### 6.3 Tokenizer JEPA integration

The SmartTokenizer provides:
- **Per-batch JEPA feature buffer** (pre-allocated `Vec<f32>`) when `caps.has_jepa`
- **Zero-alloc write** into workspace `jepa_z` via `pack_into_raw`
- **Embedding row pointer** so the JEPA stabilizer can index embeddings by token ID

---

## 7. CMUD Complex Manifold

### 7.1 Phase-Coherent Attention

```
φ_i = ω * i + q_phase[i]  (positional + learned bias)
Q_i = h_i * cos(φ_i)      (phase-modulated query)
K_j = h_j * cos(φ_j)      (phase-modulated key)
attn_ij = softmax(cos(φ_i - φ_j) * scale)  (phase-coherent attention)
```

The **phase array** `φ` is the core novelty — it's a hidden_size-length vector that rotates each dimension's contribution.

### 7.2 Hermitian Ball Clamp

```
radius = RMS(hidden) * CMUD_RADIUS_RMS_FACTOR
h = h / |h| * radius  (project onto hypersphere)
```

This is a **deterministic normalization** applied after each think step.

### 7.3 Wave-Function Collapse

```
h_final = Re(h) · (1 + tanh(|Im|)) · cos(θ)
```

The "imagination energy" `|Im|` modulates the output magnitude. This is **deterministic given the complex state**.

### 7.4 Tokenizer CMUD integration

CMUD operates **after** the transformer stack and **before** the LM head — it doesn't interact with the tokenizer directly. The SmartTokenizer's contribution is:
- Mode detection (`has_cmud` flag) to reserve CMUD phase buffers
- Token sequence metadata for sequence-aware phase initialization
- Same token ID → embedding path as non-CMUD mode

---

## 8. Pointer/Register Optimization Strategy

### 8.1 YMM alignment contract

| Data | Required alignment | Kernel |
|------|-------------------|--------|
| `x_f32` (activations) | 32 bytes (256-bit) | All `_mm256_load_ps` |
| `w_u32` (packed ternary) | 4 bytes (word) | `vpbroadcastd` (no alignment needed) |
| `w_f32` (F32 weights) | 32 bytes | `_mm256_load_ps` |
| `out_f32` (output) | 32 bytes | `_mm256_store_ps` |
| `scales_f32` (PRQ) | 4 bytes | Scalar `vmovss` |
| `token_ids` (u32) | 4 bytes | Scalar GEMV index |
| `jepa_z` (f32) | 32 bytes | `_mm256_load_ps` in stabilizer |
| `phi` (CMUD phase) | 32 bytes | `_mm256_load_ps` in phase attn |

### 8.2 Prefetch distances

Measured on i7-1260P (Alder Lake, 12 MB L3, 1.25 MB L2, 80 KB L1):
- **L1 prefetch**: `prefetcht0` at 3-4 cache lines ahead (192-256 bytes)
- **L2 prefetch**: `prefetcht1` at 6-8 cache lines ahead
- **NTA (non-temporal)**: `prefetchnta` at 2-3 cache lines ahead for weight streams

### 8.3 Cache line separation

Critical structures that must not share a cache line (64 bytes) with hot-mutable data:
- `SlimeRegister` (8 bytes) — 8× padding to fill 64 bytes
- `PackedTokenBatch.ids` — should be in its own allocation
- `jepa_z` buffer — should not share with `jepa_mu`/`jepa_inv_sigma`

---

## 9. SmartTokenizer Design Summary

### 9.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SmartTokenizer                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Tokenizer (core BPE)                                      │    │
│  │  - vocab: HashMap<String, u32>                             │    │
│  │  - merges: HashMap<(String,String), u32>                   │    │
│  │  - byte_encoder, space_char, special_tokens                │    │
│  └────────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  ModelCapabilities                                         │    │
│  │  - mode: Ternary | F32                                     │    │
│  │  - has_jepa, has_cmud, has_thinker                         │    │
│  │  - hidden_size, vocab_size                                 │    │
│  └────────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Zero-Alloc Buffers (reused Vecs)                          │    │
│  │  - encode_buf: Vec<u32>                                    │    │
│  │  - jepa_scratch: Vec<f32>                                  │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
         │                     │                     │
    ┌────▼────┐         ┌─────▼──────┐        ┌─────▼──────┐
    │ PACK    │         │ STREAM     │        │ RAW        │
    │ pack_   │         │ stream_    │        │ encode_    │
    │ text()  │         │ batches()  │        │ raw()      │
    └─────────┘         └────────────┘        └────────────┘
```

### 9.2 Data flow through the system

```
Text input
    │
    ▼
SmartTokenizer.encode_to_buf()  →  &[u32]  (internal buffer, zero-alloc on re-use)
    │
    ├──► corpus_trainer AOT cache  →  .bin (pre-tokenized u32 streams)
    │       │
    │       ├──► sequence_pack::read_chunk → (input_ids, target_ids)
    │       │       │
    │       │       └──► token_embd.weight GEMV (ternary or F32)
    │       │               │
    │       │               └──► SlimeRegister.matmul_accum (f32)
    │       │                       │
    │       │                       └──► Transformer layers
    │       │                               │
    │       │                               └──► LM head → logits
    │       │                                       │
    │       │                                       └──► Sampled softmax loss
    │       │                                               │
    │       │                                               └──► Backward (STE)
    │
    └──► main.rs inference
            │
            └──► token_embd.weight GEMV  →  autoregressive loop
```

### 9.3 Key design decisions

1. **Wrapping, not replacement**: SmartTokenizer wraps the existing `Tokenizer`. All original methods remain accessible through `self.tokenizer.*`.
2. **Buffer reuse**: `encode_to_buf()` reuses an internal `Vec<u32>` — the trainer can call it repeatedly without allocating.
3. **Raw pointers**: `encode_raw()` returns `(*const u32, len)` for zero-copy integration with workspace register setup.
4. **Mode detection**: `ModelCapabilities` is auto-detected from metadata — no manual configuration.
5. **Streaming**: `TokenBatchStream` provides corpus→batch iteration without intermediate allocations.

---

## 10. Integration Points

| Subsystem | File | Change | Status |
|-----------|------|--------|--------|
| Trainer | `src/mud/corpus_trainer.rs` | `Arc<Tokenizer>` → `Arc<SmartTokenizer>` | DONE |
| CLI inference | `src/main.rs` | `Tokenizer::from_mud_metadata` → `SmartTokenizer::from_mud_metadata` | DONE |
| Inference lib | `src/mud/inference.rs` | Keeps raw Tokenizer for internal analysis helpers | NO CHANGE NEEDED |
| Debate arena | `src/mud/debate_trainer.rs` | Uses bare `Tokenizer` (passes via `tokenizer` field) | NO CHANGE NEEDED |
| Arena judge | `src/mud/arena_judge.rs` | Borrows `&Tokenizer` | NO CHANGE NEEDED |
| CMUD train | `tools/cmud_train.rs` | Uses `Tokenizer::from_mud_metadata` for auto-tokenization | NO CHANGE NEEDED |

---

## 11. Future Optimisations

### 11.1 SIMD tokenizer encoding

The BPE `encode()` method currently processes one character at a time. The byte_encoder lookup could be vectorised with `_mm256_shuffle_epi8` (vpshufb) for the 16-byte-aligned path, mapping bytes to unicode codepoints with a single table lookup.

### 11.2 Embedding cache

For inference with repeated prompts (e.g., chatbot), cache the embedding lookup results for the most frequent token IDs. The SmartTokenizer can maintain a small LRU cache of `(token_id → embedding_f32[hidden_size])`.

### 11.3 JEPA feature pre-computation

The JEPA stabilizer's z-score calculation (mean + variance across hidden dimensions) could be pre-computed per token batch by the tokenizer, saving one reduction pass in the forward path.

### 11.4 Register-pressure-aware batching

The 8-row GEMV kernel (16 YMM, all used) constrains the number of parallel activation streams. The smart tokenizer could detect this and split batches into 4-row sub-batches (which leave 5 spare YMM for prefetch/NaN guards) when the model is in register-pressure-sensitive mode.

---

## 12. References

- AVX2 kernels: `src/asm/*.s`
- STE packing: `src/mud/ezop.rs` (pack_elut_prq, pack_ternary_into)
- JEPA stabilizer: `src/mud/slime_jepa.rs`
- CMUD thinker: `src/mud/cmud.rs`
- SmartTokenizer: `src/mud/smart_tokenizer.rs`
- Workspace: `src/mud/workspace.rs`, `src/mud/slime.rs`
- PCorePool: `src/mud/pcore_pool.rs`
- Training pipeline: `src/mud/corpus_trainer.rs`
