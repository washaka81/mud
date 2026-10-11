# Session Report — Phase B+ GPU GEMV in forward (2026-07-16)

## Goal
Wire tiled ash ternary GEMV into the live forward path without regressing default AVX2 performance.

## Design
| Gate | Rule |
|------|------|
| Env | `MUD_GPU_GEMV=1` (opt-in; default **off**) |
| Vulkan | `MUD_USE_VULKAN` not `0`/`false` |
| Size | `n_in * n_out ≥ GEMV_GPU_MIN_WORK` (256²) |
| Fallback | Any failure → AVX2 `ternary_gemv_rowwise_submit` |

### Weight cache
Host weight/scale pointers from `.mud` are stable. `GemvGpuCache` skips VRAM re-upload when `w_u8` / `scales` pointers match the previous call (`dispatch_gemv_host_sync_ex` flags). Activations always upload.

## Call site
`ternary_gemv_rowwise` (shared by dense FFN, MoE experts, O-proj, etc.):

```text
try_gpu_ternary_gemv → else pool AVX2
```

QKV remains CPU-parallel (`ternary_gemv_qkv_parallel`) for multi-matrix overlap; individual large GEMVs elsewhere use the GPU gate.

## Usage
```bash
export MUD_USE_VULKAN=1
export MUD_GPU_GEMV=1
# then forge_llm / trainer / chat
```

## Validation
- `test_gpu_gemv_opt_in_default_off`
- `test_phase_bplus_gpu_gemv_matches_cpu_if_available` (skip if no GPU)
- full lib suite + clippy clean

## Ledger
Phase B+ **DONE**. Active open items: deferred L-13/L-14/L-15 only.
