# MUD Session Report — 2026-07-16

**Focus:** Document reconciliation · ASM polish & wiring · Compute stack audit (ELUT/FP32 + PCorePool×8 + ash Vulkan) · Tooling truthfulness

---

## 1. Document hierarchy (carried from prior session)

| Document | Role |
|----------|------|
| `GEMINI.md` | SSOT policies P-#, status, ledger L-## |
| `AGENTS.md` | Agent context (derived) |
| `VISION_ROADMAP.md` / `PLAN_MAESTRO.md` | Product vision & phases |

Canonical open work remains **L-01…L-08** (Phase A). Do not treat banners that say “Muon+Adam” as runtime truth.

---

## 2. ASM polish (completed this arc)

### Kernels
| File | Change |
|------|--------|
| `ternary_gemv.s` | Wide prefetch (512/1024), `vhaddps` reduce, NaN/Inf → 0 on scalar out |
| `ternary_gemv_4rows.s` | Safe `cmp $8` loop, wider prefetch, vhaddps + NaN sanitize, drop unused r15 |
| `ternary_gemm_batch4.s` | Prefetch inner loop, vhaddps reduce |
| `lm_head.s` | Inlined FMA argmax + **new** `lm_head_logits_avx2` (full vocab logits) |
| `silu.s` | Clamp domain, `vrcpps` + 1 NR instead of `vdivps`, prefetch |
| `adam_step.s` | AT&T rewrite, prefetch 4 streams, NaN grad kill |
| `math.s` | Scalar leftover on dot, `vhaddps`, push/pop `%rbx` in hadamard |
| `elut_gemv.s` | `vzeroupper` |
| `sgemm.s` | Prefetch on `sgemm_abt` inner loop |

### Wiring
| Call site | Kernel |
|-----------|--------|
| `main.rs` generation | `lm_head_logits_avx2` + prealloc `logits`/`reg_f32` (P-01) |
| `slime_forward` SwiGLU | `silu_vectorial_avx2` |
| `slime_forward` attention scores | `asm::dot_product_avx2` |
| `speculative` / pool paths | `sgemm_abt` → ASM (no pure-Rust triple loop) |

### FFI (`src/asm/mod.rs`)
Live: `lm_head_avx2`, `lm_head_logits_avx2`, `adam_step_avx2`, `sgemm_abt_avx2`, ternary/silu/dot, etc.  
Dummy `ternary_gemv_backward_avx2` removed. `qat_step.s` build reference removed.

### Tests
- ~86 lib tests pass (incl. lm_head logits vs scalar, silu smoke, gemv NaN sanitize contract).
- Clippy clean on lib; release `forge_llm` builds.

### Ledger
- **L-04 PARTIAL** — lm_head logits live; adam FFI ready but not on QAT step; slime_rmsnorm still open.

---

## 3. Compute stack audit — packing FP32 · AVX2 · 8 threads · ash

### 3.1 What “empaquetado” means here (two layers)

```
STORAGE (disk / .mud tensors)
  Ternary2Bit ELUT 4-bit nibble packing
  8 weights per u32; codes: 0x1 → +1, 0xF → −1, else 0
  + PRQ per-row f32 scale (*.prq_scale)

RUNTIME ACCUMULATION (hot path)
  Activations & register matmul_accum: native FP32
  (dual-f16 / i16 accum is historical — P-02 current = f32)
```

**Unpack:** `unpack_ternary2bit_to_f32` / row dequant for shadows & embeddings.  
**Never:** cast packed ELUT bytes as `*const f32`.

### 3.2 Forward GEMV — CPU (primary, live)

```
ternary_gemv_rowwise (slime_forward.rs)
  → PCorePool global (8 workers, pinned via core_affinity)
  → split n_out rows across 8 tasks
  → ternary_gemv_4rows (×4 rows, shared x) or ternary_gemv (tail)
  → multiply by PRQ scale per row (finite clamp)
  → pool.wait_all()
```

| Property | Reality |
|----------|---------|
| Parallelism | **8 threads** hard-coded (`get_pool()` → `PCorePool::new(8)`) — P-13 debt (L-07) |
| ISA | AVX2 + FMA handwritten ASM |
| Activation dtype | **FP32** vector load |
| Weight dtype | ELUT u32 packed |
| Output | FP32 scalar per row |
| Vulkan GEMV in this path | **Not used** |

### 3.3 Forward GEMV — Vulkan ash (present, not on critical path)

| Component | Status |
|-----------|--------|
| Shader `ternary_gemv_unified.comp` | ELUT 4-bit, subgroup reduce — **aligned with CPU packing** |
| `AshContext::dispatch_gemv_sync` | Implemented |
| Callers | **None** outside ash_backend itself |
| Intent (docs) | Large mats (≥~1M elems) when `MUD_USE_VULKAN` — **not wired from slime_forward** |

### 3.4 Training / QAT

| Piece | Reality |
|-------|---------|
| Shadow weights | FP32 |
| Live optimizer step | **`sgd_step_avx2`** only (`apply_optimizer_cpu_step_and_pack`) |
| `select_optimizer()` | Stores Muon/GaLore/… **unread at step** → **L-01** |
| Pack-back | CPU: threshold `0.7 * s` → ELUT 4-bit (8-thread pack loop) |
| `AshQatDispatcher` | Zero-copy buffer scaffolding; optimizer GPU path partial |
| Newton-Schulz shaders | Compiled; **not** dispatched from step (**L-02**) |
| L-QAT init path | Still has `use_vulkan` flag with TODO + CPU fallback |

### 3.5 Heterogeneous picture (truth table)

```
                 CPU AVX2×8          Iris Xe (ash)
Forward GEMV     LIVE primary        API ready, unwired
SiLU / dots      LIVE ASM            n/a
LM head logits   LIVE ASM            n/a
QAT step         LIVE SGD AVX2       partial async scaffolding
Muon NS          CPU module only     shaders exist, no dispatch
```

**Conclusion of review:** Packing ELUT↔FP32 is consistent CPU↔shader. Parallelism is real on **CPU GEMV + pack**. Ash is infrastructure-ready but **must not be advertised as the live training/inference compute path** until dispatch is hooked and measured.

---

## 4. Tooling updates (same session)

See commit/diff for: `run_trainer`, `corpus_trainer` banners, `training_healthcheck`, `avx_math_validator`, `STATUS_REPORT.md`, `GEMINI.md` / `AGENTS.md` sync.

Rule for tools: report **live path**, then **planned** strategies separately.

---

## 5. Next recommended work (tomorrow / roadmap resume)

1. **L-01** — wire `OptimizerStrategy` into step (primary).  
2. **L-02** — Newton-Schulz when Muon.  
3. **L-03** — delete `InferenceWorkspace`.  
4. **L-07** — pool size / metadata, not hard 8.  
5. Optional later: GEMV ash router with measured threshold.

Handoff pointers updated in: `VISION_ROADMAP.md` §7, `GEMINI.md` §6.4, `AGENTS.md` Next Work, `docs/README.md`.

---

*Session end 2026-07-16. Policies: GEMINI.md. Vision: VISION_ROADMAP.md. Stack: MUD_COMPUTE_STACK.md.*
