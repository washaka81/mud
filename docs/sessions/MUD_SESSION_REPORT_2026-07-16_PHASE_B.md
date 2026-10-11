# Session Report — Phase B core (2026-07-16)

## Goals
Extreme perf foundations from `VULKAN_AVX2_THREADS_OPTIMIZATION` / vision Phase B:
1. Shared-memory tiling on GPU ternary GEMV
2. CPU QKV (and FFN up/gate) without serial `wait_all` barriers

## Delivered

### GPU: `ternary_gemv_unified.comp`
- Activation vector loaded in **shared tiles** (`SX_MAX=4096`, 16 KiB)
- Workgroup **tree reduce** (no subgroup-size assumption)
- Optional fused RMS (`do_norm`) via shared sum-of-squares
- SPIR-V rebuilt (`--target-env=vulkan1.1`)
- Host API: `AshContext::dispatch_gemv_host_sync`
- Threshold constant: `GEMV_GPU_MIN_WORK = 256²`
- GPU test: `test_phase_b_gemv_shared_tile` (weight[0]=+1 → y = x0·scale)

### CPU: pool overlap
- `ternary_gemv_rowwise_submit` — enqueue without wait
- `ternary_gemv_qkv_parallel` — Q+K+V then **one** `wait_all`
- Wired into attention Step 2
- Dense FFN up+gate use the same submit/wait pattern

## Not done (Phase B+)
| Item | Notes |
|------|--------|
| GPU GEMV as default forward | Still CPU ASM; profile before switching |
| Multi-matrix GPU dispatch (QKV one CB) | Future |
| Fused RMSNorm+GEMV full path | Shader flag exists; not engine-default |

## Validation
- `cargo test --lib` → **127 passed**
- `cargo clippy --lib -- -D warnings` → clean

## Next
**L-12** property tests + CI health battery.
