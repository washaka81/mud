# MUD Engine - Audit & Tune-up Resolution Log

Este documento registra todas las correcciones estructurales aplicadas al motor MUD para garantizar estabilidad matemática, rendimiento y robustez.

| # | Hallazgo | Archivos Modificados | Fix Implementado |
|---|---|---|---|
| 1 | RoPE incompatible (base vs half) | `src/model/transformer.rs` | Unificado a LLaMA-style (adjacent-pair, base=10k). |
| 2 | Softmax produce NaN si sum_exp=0 | `src/mud/inference.rs`, `transformer.rs` | Agregado `+ 1e-30` al denominador y multiplicación por `inv_sum`. |
| 3 | PageRank pierde rango (dangling) | `src/mud/graph.rs` | Redistribución equitativa de rango para nodos sin salida. |
| 4 | Temperature=0 causa Inf | `src/mud/inference.rs` | `temperature.max(1e-8)` y tipos f32 explícitos. |
| 5 | RngExt error + falta de tests | `tools/ternary_audit.rs`, `src/asm/tests.rs` | Corregido uso de `rand` y agregados 9 tests AVX2. |
| 6 | hidden_size hardcodeado en facts | `src/mud/store.rs`, `web_search.rs` | Dinamización del parámetro `hidden_size` vía metadatos. |
| 7 | Damping 1/sqrt(N) arbitrario | `src/mud/inference.rs` | Normalización estándar `1/sqrt(2)` para pre-norm. |
| 8 | Sesgo en sampleo (r > cum_sum) | `src/mud/inference.rs` | Guard `cum >= 1.0 - 1e-7` y fallback a `UNK`. |
| 9 | Sandbox sin funciones avanzadas | `tools/math_sandbox.py` | Agregadas: sqrt, sin, cos, tan, log, exp, abs, round, pi, e. |
| 10 | output_norm_w null sin check | `src/model/inference.rs` | Validación explícita `if !is_null()`. |
| 11 | Alignment memory no verificado | `src/mud/inference.rs` | `assert!` para `hidden_size % 16` y `% 64`. |
| 12 | Vulkan .unwrap() en hot-path | `src/mud/inference.rs` | Implementado fallback silencioso a CPU si GPU falla. |
| 13 | vec![] allocation en hot-loop | `src/mud/inference.rs` | Migrado a `InferenceWorkspace` (buffers pre-asignados). |

*Status: Audit Resolution Phase COMPLETED.*
