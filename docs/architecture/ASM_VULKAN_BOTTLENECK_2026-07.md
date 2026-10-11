# Vulkan iGPU vs AVX2 GEMV — bottleneck notes (2026-07-16)

## What nvtop shows (Iris Xe, no NVIDIA)

| Signal | Meaning |
|--------|---------|
| iGPU clock / % busy | Heartbeat shader, EZOP QAT buffer alloc, compositor, **not** necessarily ternary GEMV |
| No `nvidia-smi` | No discrete NVIDIA GPU on this host |
| `run_trainer` ~800% CPU | **Real hot path = AVX2 × 8 P-cores** |

## Bench (this machine)

```
./target/release/gemv_auto_bench   # MUD_GPU_GEMV=auto MUD_USE_VULKAN=1
→ min_work=NEVER  GPU never beat CPU
  512×512:  cpu≈24µs  gpu_hot≈2.8ms  (GPU ~100× slower)
```

Root cause: **UMA upload W + dispatch + readback Y** dominates.  
Even when policy said “GPU wins ≥16k work” mid-train, re-bench shows CPU always wins on Alder Lake iGPU.

## Train path reality (Bonsai 1.7B)

- Each STE **step**: full **FWD 28 layers** (7 GEMVs/layer) + sampled head + BWD last-N.
- TELEM `tok/s` = **steps/s**, not chat tokens/s.
- ~1.5 steps/s ⇒ ~11 s/chunk × 16 steps — memory + compute bound on CPU, not “Vulkan idle bug”.

With `MUD_TRAIN_SCALES_ONLY=1`: ash **optimizer pack is skipped** (trit-safe). Ash may still:
- allocate QAT VRAM (EZOP),
- run **heartbeat** once per step (keeps RC6 awake → clocks up in nvtop).

## ASM improvements (priority)

| # | File | Change | Expected |
|---|------|--------|----------|
| 1 | `ternary_gemv_4rows.s` | **16-col main loop** (2× unroll) — DONE 2026-07 | −loop overhead on hot path |
| 2 / **T10** | `ternary_gemv.s` (+4rows) | Prefetch tune for **i7-1260P** (T0/T1 x, NTA W @256/512/1024) — DONE 2026-07 | hide DDR latency |
| 3 / **T11** | `ternary_gemv_8rows.s` + submit | **8-row** kernel + tests; submit default **4-row**, opt-in `MUD_GEMV_ROWS=8` — DONE 2026-07 | +reuse of `x` (host-dependent) |

### Microbench note (this host, n=2048, 1024 rows ×32, release)

| Kernel | vs 1-row | vs 4-row |
|--------|----------|----------|
| 4-row | ~1.3–2.1× | 1.0× |
| 8-row | ~1.0–2.0× | ~0.8–1.3× (noisy; often ≤4-row when L2 warm) |

Default stays **4-row**. Enable 8-row when profiling a colder multi-thread FWD and it wins.
| 4 | Train topology | `FWD_LAST_N` skip frozen layers | **largest** win for seating |
| 5 | Emb | On-the-fly row unpack (no 1.2 GB FP32 emb) | RAM + BW |

## Recommended env for seating on this laptop

```bash
export MUD_PCORE_THREADS=8
export MUD_GPU_GEMV=0          # force CPU — matches bench; avoids GPU thrash
export MUD_USE_VULKAN=1        # optional: heartbeat / future NS GPU only
export MUD_TRAIN_SCALES_ONLY=1
```

Do **not** expect nvtop iGPU % to track train speed until GEMV stays resident on GPU without per-call full W upload (persistent weight SSBOs + only upload `x`/`y`).
