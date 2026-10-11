# MUD Register Bit Packing Evolution

> **Canonical reference** for the SlimeRegister, SlimeXCellV5, and f32 accumulation architecture.  
> **Status:** Active — P-02 contract: `matmul_accum` is f32 in compute, f16 in storage.  
> **Last updated:** 2026-07-24

---

## 1. Current Architecture: SlimeRegister v3

### 1.1 Physical layout (u32)

```
┌─────────────────────┬─────────────────────┐
│  bits 0-15: re      │  bits 16-31: im     │
│  IEEE binary16      │  IEEE binary16      │
│  (ternary GEMV acc) │  (JEPA integral)    │
└─────────────────────┴─────────────────────┘
         4 bytes total (u32)
```

### 1.2 Conversion path

```
Compute (f32) ──vcvtps2ph──► Storage (f16 in u32) ──vcvtph2ps──► Compute (f32)
                1 cycle                      1 cycle
                on i7-1260P                  on i7-1260P
```

### 1.3 API (slime.rs)

| Method | Input | Output | HW |
|--------|-------|--------|-----|
| `pack(re, im)` | f32, f32 | SlimeRegister | vcvtps2ph ×2 |
| `real()` / `read_accum()` | — | f32 | vcvtph2ps |
| `imag()` / `read_integral()` | — | f32 | vcvtph2ps |
| `write_accum(val)` | f32 | — | vcvtps2ph |
| `write_integral(val)` | f32 | — | vcvtps2ph |
| `gate()` | — | f32 (sigmoid) | exp() |
| `as_complex_f32()` | — | (f32, f32) | vcvtph2ps ×2 |
| `write_complex(re, im)` | f32, f32 | — | vcvtps2ph ×2 |

### 1.4 CMUD Thinker mode

In CMUD mode (`MUD_CMUD_THINK=1`), the register is treated as `Complex<f16>`:
- `re` = ternary GEMV accumulator
- `im` = JEPA integral derivative
- Phase-coherent attention uses `as_complex_f32()` → (re, im) pair
- Hermitian ball clamp operates on the complex magnitude

---

## 2. Historical Evolution

### v1: i16 saturating accumulator (deprecated)

```
┌─────────────────────┬─────────────────────┐
│  bits 0-15: i16     │  bits 16-31: i16    │
│  (ternary accum)    │  (JEPA state)       │
└─────────────────────┴─────────────────────┘
```

**Problems:**
- **Saturation crisis:** Accumulators clamped at ±32767, destroying gradient signal
- **Dead zone:** Small STE gradients lost when truncating f32→i16
- **P-04 obsoleted:** `SLIME_RESEAT_STRIDE` / i16 mid-row reseat no longer needed

**Status:** DEPRECATED (P-04). All i16 accum code paths removed.

### v2: f16 with iscale parameter

```
pub fn read_accum(&self, iscale: f32) -> f32   // iscale was a per-layer scale
pub fn write_accum(&mut self, val: f32, iscale: f32)
```

**Problems:**
- `iscale` parameter threaded through every call site (~50 locations)
- Redundant with PRQ per-row scales already in workspace
- API cluttered, error-prone

**Status:** REMOVED (Session 77, 2026-07-02). `iscale` dropped from API.
**Code verification (2026-08-02):** `read_accum(&self) -> f32` and `write_accum(&mut self, val: f32)` confirmed in `src/mud/slime.rs:68,74` — signatures match v3 below, no `iscale` parameter.

### v3: f16 packed, f32 compute (current)

```
Storage: u32 = f16(re) | f16(im) << 16
Compute: read_accum() → f32 via vcvtph2ps
         write_accum(val) → f16 via vcvtps2ph
```

**Key insight:** The register stores f16 for compactness (4 bytes), but ALL math happens in f32. The f16↔f32 conversion is 1 cycle each way on modern Intel with F16C.

**JEPA state:** Moved out of the register into `SlimeWorkspace.jepa_z` (flat f32 array). The register's `im` component stores the JEPA integral derivative, not the full JEPA state.

---

## 3. SlimeXCellV5 (Transport Format)

### 3.1 Physical layout (u32, 8:8:8:8)

```
┌──────────┬──────────┬──────────────┬──────────────┐
│  bits 0-7│ 8-15     │ 16-23        │ 24-31        │
│  accum   │ address  │ jepa_vel     │ ste_parity   │
│  fp8 (u8)│ tag (u8) │ (i8)         │ (u8)         │
└──────────┴──────────┴──────────────┴──────────────┘
```

### 3.2 Field semantics

| Field | Bits | Type | Range | Description |
|-------|------|------|-------|-------------|
| `accum_fp8` | 0-7 | u8 | [0, 255] | Pseudo-FP8 magnitude of ternary accumulator |
| `address_tag` | 8-15 | u8 | [0, 255] | Register index (256-slot Memory Continuity Table) |
| `jepa_velocity` | 16-23 | i8 | [-128, 127] | JEPA integral mapped to i8 |
| `ste_parity` | 24-31 | u8 | [0, 1] | Sign of ternary accumulator (0=+, 1=−) |

### 3.3 Purpose

SlimeXCellV5 is a **transport/storage** format, NOT a compute format:
- Used in Memory Continuity Table (256 slots, 100% L1d)
- Used in AOT Boot Indexer for fast model loading
- Used in ShadowExpertBus for multi-expert routing
- **Never** used in the hot GEMV/attention compute path

### 3.4 Conversion: SlimeRegister → SlimeXCellV5

```rust
// snapshot_to_slimex() in slime.rs
let real_abs = reg.read_accum().abs().min(255.0);  // f32 → u8
let accum_fp8 = real_abs as u8;
let address_tag = (i & 0xFF) as u8;
let jepa_velocity = jepa_z.clamp(-128.0, 127.0) as i8;
let ste_parity = if reg.read_accum() >= 0.0 { 0 } else { 1 };
```

---

## 4. SlimeWorkspace f32 Buffers

All compute buffers in `SlimeWorkspace` are `AlignedArray<f32>` (64-byte aligned):

| Buffer | Size | Purpose |
|--------|------|---------|
| `registers` | hidden_size | SlimeRegister u32 array |
| `kv_cache` / `v_cache` | kv_dim × max_pos | KV cache (f32 or f16 packs) |
| `hca_kv_cache` / `hca_v_cache` | kv_dim × hca_cap | HCA compressed history |
| `jepa_z` | 2 × num_layers × hidden | JEPA EMA state (flat f32) |
| `jepa_mu` / `jepa_inv_sigma` | hidden | JEPA z-score params |
| `gemv_accum` | hidden | GEMV scratch |
| `q_f32` / `k_f32` / `v_f32` | dim | Attention QKV |
| `scores` | max_pos | Attention scores |
| `ffn_*_f32` | ffn_mid/hidden | FFN intermediate |

---

## 5. Compute Pipeline

```
Embedding (ELUT 4-bit) ──read_accum()──► f32
       │
       ▼
RMSNorm ──────────────────────────────────► f32
       │
       ▼
QKV GEMV (AVX2) ─────────────────────────► f32
       │
       ▼
SDPA + GQA (HCA + CSA top-k) ────────────► f32
       │
       ▼
Output GEMV ──────────────────────────────► f32
       │
       ▼
JEPA OU update ──── write_integral() ─────► f16 in register
       │
       ▼
mHC residual blend ──────────────────────► f32
       │
       ▼
FFN (Up/Gate/SiLU/Down) ────────────────► f32
       │
       ▼
write_accum() ────────────────────────────► f16 in register (next block input)
```

---

## 6. Policy Reference

| Policy | Rule | Status |
|--------|------|--------|
| **P-02** | `matmul_accum` is f32 in compute | ACTIVE — current truth |
| **P-03** | ELUT 4-bit nibble wire format | ACTIVE — 8 weights/u32 |
| **P-04** | i16 partial reseat ≤256 | DEPRECATED — obsoleted by f32 accum |
| **P-00** | Raw pointer mastery in hot path | ACTIVE — `*const u32` in GEMV |
| **P-01** | Zero allocation in hot loops | ACTIVE — SlimeWorkspace pre-alloc |

---

## 7. Benchmark Results (2026-07-24)

| Kernel | ASM/AVX2 | Scalar | Speedup |
|--------|----------|--------|---------|
| ternary_gemv (n=4096) | 480 ns | 2502 ns | 5.2× |
| dot_product (n=4096) | 285 ns | 2075 ns | 7.3× |
| rms_norm (n=4096) | 474 ns | 3749 ns | 7.9× |
| fenwick_prefix_sum (n=131072) | 451 ns | 3930453 ns | 8715× |
| **Geometric mean** | | | **8.8×** (16 kernels) |
