# SlimeRegister v2 — Integral JEPA Architecture

## Design: Single u32 (4 bytes) per hidden dimension

```
u32 bits:
  [31:16] = jepa_integral_f16   — running integral ∫v_jepa dt (f16, I-controller)
  [15:0]  = ternary_f16         — ternary matmul accumulation (f16, no iscale)
```

## Key Changes vs v1

| Aspect | v1 | v2 |
|--------|----|----|
| Ternary storage | `i16` fixed-point × iscale | `f16` half-float (direct) |
| JEPA storage | `u16` f16 instantaneous v_jepa | `u16` f16 integral I |
| iscale in hot loop | Required everywhere | Eliminated |
| Gate signal | sigmoid(v_jepa) proportional | sigmoid(I) integral |
| Register size | 4 bytes | 4 bytes (unchanged) |

## Integral Control Law (JEPA I-controller)

`I[t] = 0.99 * I[t-1] + 0.01 * v_jepa[t]`

- I→0 when v_jepa→0 (equilibrium guaranteed)
- Low-pass filter: rejects transient noise spikes
- Smoother convergence than proportional-only control
