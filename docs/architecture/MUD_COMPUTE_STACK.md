# MUD Compute Stack — ELUT / FP32 / AVX2×8 / ash Vulkan

**SSOT for runtime behaviour:** `GEMINI.md` §0.  
**Last review:** 2026-07-16 (post streams A–E).

---

## 1. Data formats

### 1.1 On-disk weights (ELUT Ternary2Bit)

| Property | Value |
|----------|--------|
| Codes | nibble 4-bit: `0x1` → +1, `0xF` → −1, else 0 |
| Density | 8 weights / `u32` |
| Scale | Per-row PRQ `f32` (`*.prq_scale`) |
| Sparsity pack threshold (QAT) | `\|w\| > 0.7 · s` ≈ 26% zeros target |

```
byte stream:  [u32 packed][u32 packed]…
                └ nibble0 = w[0], nibble1 = w[1], … nibble7 = w[7]
```

### 1.2 Runtime tensors (FP32 path)

| Tensor | Dtype | Notes |
|--------|-------|--------|
| Activations into GEMV | `f32` | After RMSNorm; i8-act GEMV path retired (NaN history) |
| `SlimeRegister.matmul_accum` | `f32` | P-02 current contract |
| JEPA `jepa_z` | `f32` | Workspace buffer |
| Shadow weights (QAT) | `f32` | STE; pack back to ELUT after step |
| LM head weights (inference dequant) | `f32` flat | Unpacked once at load for `lm_head_logits_avx2` |

---

## 2. CPU path — AVX2 + PCorePool (8)

### 2.1 Pool

- `src/mud/pcore_pool.rs` — lock-free slot handoff, spin wait.
- Global: `get_pool()` → `default_pcore_threads()` (env `MUD_PCORE_THREADS`, else min(cores, 8)) — **L-07**.
- Pinning: `core_affinity` to first N reported cores (intent: P-cores + HT on i7-1260P).

### 2.2 Forward GEMV (`ternary_gemv_rowwise`)

```
for task in 0..8:
  assign row range [start, end)
  pool.execute:
    while row+4 <= end:
      ternary_gemv_4rows(n_in, x, W[row..], out[row..], scale=1, stride)
      out[r] *= prq_scale[r]   # finite clamp
    leftover rows: ternary_gemv(..., scale=prq)
pool.wait_all()
```

Hot kernels (polished 2026-07-15/16):  
`src/asm/ternary_gemv.s`, `ternary_gemv_4rows.s`.

### 2.3 Other CPU hot kernels

| Op | Kernel | Wired |
|----|--------|-------|
| SiLU | `silu_vectorial_avx2` | `slime_forward` SwiGLU |
| Dot (attention) | `dot_product_avx2` | `slime_forward` scores |
| LM logits | `lm_head_logits_avx2` | `main.rs` AR loop |
| LM argmax | `lm_head_avx2` | FFI ready |
| Batch-4 GEMM | `ternary_gemm_batch4_avx2` | drafter / batch paths |
| C = A Bᵀ | `sgemm_abt_avx2` | speculative |
| QAT step | Muon/GaLore/Chunked preprocess + STE pack | **L-01 LIVE** |
| Adam / Sparse | `adam_state` + `adam_step_avx2` (sparse skips zero rows) | **Stream A LIVE** |
| Pack after step | Rust + 8-way pool | `apply_optimizer_cpu_step_and_pack` |

---

## 3. GPU path — ash + Iris Xe

### 3.1 Stack

- Backend: `src/vulkan/ash_backend.rs` (not vulkano).
- QAT helper: `src/mud/ash_qat_dispatcher.rs` (host-visible UMA buffers).
- Env: `MUD_USE_VULKAN=1` enables context creation attempts; **does not alone imply GEMV/Muon live**.

### 3.2 Shaders (ELUT-aligned)

| Shader | Role | Dispatch status |
|--------|------|-----------------|
| `ternary_gemv_unified.comp` | Forward GEMV 4-bit | **B+/C:** auto policy; **F:** QKV = 3 dispatches / 1 CB |
| `shadow_optimizer.comp` | SGD-like pack/update | Partial QAT async path |
| `newton_schulz_step1/2.comp` | Muon | **L-02 LIVE** via `dispatch_newton_schulz_sync` |
| `mha.comp` / `rms_norm.comp` | Attention / norm | **L-06 LIVE:** `dispatch_*_sync`; output_norm GPU if hidden≥512; dense MHA helper |

### 3.3 When to use GPU (stream C — automated)

| Mode (`MUD_GPU_GEMV`) | Behaviour |
|----------------------|-----------|
| `auto` / unset | One-shot CPU vs GPU micro-bench; GPU only past break-even (contiguous high-end wins) |
| `1` / `on` | Force GPU when work ≥ `GEMV_GPU_MIN_WORK` or `MUD_GPU_GEMV_MIN` |
| `0` / `off` | Always AVX2 |

- Prefer **CPU** for SiLU, JEPA, small RMSNorm (dispatch overhead).
- Tool: `cargo run --release --bin gemv_auto_bench` (+ `MUD_GPU_GEMV_LOG=1`).
- On many Iris Xe + hot AVX2×8 setups, calib may report **NEVER** (correct — stay CPU).

---

## 4. End-to-end diagrams

### Inference (CLI `forge_llm`)

```
token emb (ELUT→f32 unpack at load)
  → optional MoE buses (moe_load; multi-expert or dense expert.0)
  → evaluate_slime_block[_moe] × L
       RMSNorm → GEMV (AVX2×8 or auto ash) → Attn
         HCA mean-pool + dense ring; CSA top-k on large HCA (inference)
       FFN / ExpertBus → silu_vectorial ASM → down
       JEPA + mHC
  → output_norm
  → lm_head_logits_avx2 (full vocab, prealloc buffers)
  → Top-P sample
```

### Training step (corpus trainer)

```
full-seq window (or pairs@pos0 if MUD_TRAIN_FULL_SEQ=0)
  → per-token forward (pos grows, KV retained in window)
  → Sampled Softmax loss; accumulate grads
  → backward STE (full HCA if tape — CSA off)
  → AshQat: flush_pending → step_async_deferred (L-05 DoubleFrame)
  → CPU fallback: apply_optimizer_cpu_step_and_pack (Muon/GaLore/Chunked/Adam + STE)
```

---

## 5. Compliance notes

| Policy | Stack note |
|--------|------------|
| P-00/P-01 | Pool + ASM + prealloc buffers; EZOP partial |
| P-02 | f32 accum |
| P-03 | ELUT 4-bit hot wire |
| P-07 | Rust + ASM + GLSL only |
| P-13 | Pool via `default_pcore_threads` / env — L-07 |
| P-27 | No Rayon on runtime GEMV path |

---

## 6. Open ledger (compute-related)

| ID | Item |
|----|------|
| L-01 | ~~Wire strategies into step~~ **DONE** |
| L-02 | ~~Newton-Schulz GPU dispatch~~ **DONE** (`MUD_USE_VULKAN=1`) |
| L-04 | ~~ASM orphans purge~~ **DONE** |
| L-05 | ~~True double-buffer~~ **DONE** (DoubleFrame + deferred readback) |
| L-06 | ~~mha/rms_norm~~ **DONE** (dispatch + output_norm GPU) |
| L-07 | ~~Pool/eps/P-13~~ **DONE** |
| L-08 | ~~NaN guards ASM~~ **DONE** |
| L-09 | ~~EZOP~~ **DONE** (`src/mud/ezop.rs`) |
| L-10 | ~~Sequence packing~~ **DONE** (`src/mud/sequence_pack.rs`) |
| L-11 | ~~Mini MoE~~ **DONE** (`slime_expert.rs`, `expert_bus.rs`) |
| Phase B | ~~GEMV tile + QKV parallel~~ **DONE** |
| Phase B+ | ~~GPU GEMV in forward~~ **DONE** |
| Stream C | ~~GEMV auto policy~~ **DONE** (`gemv_policy`, `gemv_auto_bench`) |
| Stream F | ~~QKV multi-matrix one CB~~ **DONE** (`dispatch_gemv_qkv_host_sync`) |
| L-12 | ~~P-13 props + CI~~ **DONE** (`mud::p13`, `./mud.sh ci`) |
| L-13 | ~~HCA/32k KV~~ **DONE** (`kv_context`, dense ring) |
| Stream E | ~~CSA top-k~~ **DONE** (`csa_indexer`) |
| L-14 | ~~C-MUD~~ **DONE** (`src/mud/cmud.rs`, opt-in think) |
| L-15 | ~~grad ckpt~~ **DONE** (`grad_checkpoint`, `MUD_GRAD_CKPT=1`) |
| Stream A–B–D | Adam · MoE load · full-seq | **DONE** |

---

## 7. Handoff

**Ledger L-01…L-15 + depth A–E closed.**  
**Next improvements:** `docs/research/MUD_IMPROVEMENTS_POST_AE.md` (F QKV one-CB, …).  
**Residuals:** `docs/research/MUD_GAP_ANALYSIS_POST_L15.md`.  
Validate: `./mud.sh audit-full`.

### ASM set after L-04 (11 files)
`ternary_gemv`, `ternary_gemv_4rows`, `ternary_gemm_batch4`, `lm_head`, `silu`, `math`, `rmsnorm`, `sgemm`, `q4_0_gemv`, `rope`, `adam_step`.  
Removed: elut_gemv, ternary_pext, ternary_lut, slime_rmsnorm, mamba, ternary_backward.

*Tools must print LIVE policy (`gemv_policy`, CSA, full-seq) explicitly.*
