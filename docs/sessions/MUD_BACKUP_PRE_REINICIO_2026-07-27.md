# MUD Backup Pre-Reinicio — 2026-07-27

## Estado del sistema

- **Tests:** 409 passed / 0 failed / 2 ignored
- **Clippy:** 0 errors / 0 warnings (`cargo clippy --all-targets`)
- **Build:** 0 warnings (`cargo build --release`)
- **Thread-local pattern:** `thread_local! { static FOO: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) }; }`

## Rondas de optimización completadas (12 rondas)

| Ronda | Área | Tipo | Impacto | Archivo |
|-------|------|------|---------|---------|
| 1 | C-MUD `cue_phase_repulsion` | P-01 (fused loop) | 1 alloc eliminada | `src/mud/cmud.rs` |
| 1 | C-MUD `wave_collapse_batch_avx2` | Perf (AVX2) | ~3× en wave_collapse | `src/mud/cmud.rs` |
| 2 | C-MUD `think_step_phase_attn` | P-01 (thread-local) | 1 alloc eliminada | `src/mud/cmud.rs` |
| 2 | C-MUD `cmud_spectral_health` | P-01 (inline Cauchy) | 1 alloc eliminada | `src/mud/cmud.rs` |
| 2 | C-MUD `maybe_think_collapse_report` | P-01 (thread-local) | 1 alloc eliminada | `src/mud/cmud.rs` |
| 3 | C-MUD `think_step_trainable_core` | P-01 (5 buffers) | 5 allocs eliminadas | `src/mud/cmud.rs` |
| 4 | C-MUD attention loop | Perf (trig tables) | ~3.7× speedup | `src/mud/cmud.rs` |
| 4 | C-MUD `v_flat` precompute | Perf (eliminar method chain) | ~1.5× en inner loop | `src/mud/cmud.rs` |
| 5 | C-MUD `cue_phase_repulsion` | Perf (half-angle identity) | ~2.9× speedup | `src/mud/cmud.rs` |
| 6 | JEPA `jepa_stabilizer` | Perf (loop merge + branchless) | ~2× en register update | `src/mud/slime_jepa.rs` |
| 7 | STP `stp_loss_and_grad` | Perf (combined GPU loop) | ~28% menos memory traffic | `src/mud/stp_loss.rs` |
| 8 | Herramienta `inference_coherence` | Nueva herramienta | 3 métricas de coherencia continua | `tools/inference_coherence.rs` |
| 9 | RopeCache `get_or_compute` | P-01 (borrowed slices) | 0 allocs en cache hit | `src/mud/slime_forward.rs` |
| 10 | C-MUD `wave_collapse` (scalar) | Perf (trig identity) | ~5.3× en scalar fallback | `src/mud/cmud.rs` |
| 11 | KV Cache `compress_block` | P-01 (thread-local buffer) | 0 allocs (era 1 por bloque) | `src/mud/kv_compressed.rs` |
| 12 | DSPark `confidence_spherical` | P-01 (thread-local buffers) | 0 allocs (era 2 Vec<f32>) | `src/mud/speculative.rs` |
| 12 | DSPark `project_hidden_to_d` | P-01 (thread-local buffer) | 0 allocs (era 1 Vec<f32>) | `src/mud/speculative.rs` |

**Total:** 13 allocations eliminadas + múltiples optimizaciones de perf + herramienta de evaluación continua.

## Archivos modificados

### C-MUD (`src/mud/cmud.rs`)
- Thread-local buffers (líneas 16-62): `THINK_ATTN_SCRATCH`, `SPECTRAL_MAG_SCRATCH`, `THINK_TRAINABLE_SCRATCH`, `CUE_REPULSION_SCRATCH`
- `cue_phase_repulsion` (líneas 1630-1700): half-angle identity + thread-local
- `think_step_phase_attn` (líneas 795-860): trig tables + borrowed slices
- `think_step_trainable_core` (líneas 865-960): thread-local buffers + trig tables + v_flat
- `wave_collapse` (líneas 390-405): trig identity (atan2+cos → sqrt+div)

### JEPA (`src/mud/slime_jepa.rs`)
- `jepa_stabilizer` (líneas 166-312): v_jepa loop merge + branchless jepa_alpha + single-pass mean/variance

### STP (`src/mud/stp_loss.rs`)
- `stp_loss_and_grad` (líneas 112-165): combined GPU loop

### RopeCache (`src/mud/slime_forward.rs`)
- `RopeCache::get_or_compute` (líneas 138-180): borrowed slices + raw pointer eviction
- `evaluate_slime_block_moe` (línea 1164): guard held for borrowed slices

### KV Cache (`src/mud/kv_compressed.rs`)
- Thread-local `KV_DELTA_BUF` (líneas 1-10)
- `compress_block`: thread-local buffer instead of `vec![]` allocation
- `decompress_block`: `unwrap_or_default()` + bounds check

### DSPark (`src/mud/speculative.rs`)
- Thread-local buffers: `SPHERICAL_NORM_A`, `SPHERICAL_NORM_B`, `HIDDEN_PROJ_BUF`
- `confidence_spherical`: inlined spherical_norm + thread-local buffers
- `project_hidden_to_d`: thread-local buffer

### Herramientas
- `tools/inference_coherence.rs`: nueva herramienta (3 métricas de coherencia)
- `Cargo.toml`: agregado `[[bin]] name = "inference_coherence"`
- `mud.sh`: agregado comando `coherence`

### Documentación
- `docs/sessions/MUD_SESSION_REPORT_2026-07-27_CMUD_OPTIMIZATIONS.md`: 12 rondas documentadas

## Pendientes (DEFERRED en AGENTS.md)

| ID | Tarea | Requiere |
|----|-------|----------|
| CellV5 Fase 2 | ASM kernels (`cmud_cell_v5_avx2.s`) | Bit-exact layout 8:8:8:8 |
| CellV5 Fase 3 | KV CellV5 pre-compresión | Integración lossless |
| phase_repulsion_batch_avx2 | AVX2 para CellV5 u32 | Exploración bit-exact |
| DSPark production wiring | Wire stubs into inference pipeline | Integration con `main.rs` |

## Comandos de validación

```bash
cargo test --lib -- --test-threads=1    # 409 passed / 0 failed / 2 ignored
cargo clippy --all-targets             # 0 errors / 0 warnings
cargo build --release                  # 0 warnings
```

## Patrón thread-local confirmado

```rust
thread_local! {
    static FOO: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

// Uso:
FOO.with(|cell| {
    let mut buf = cell.borrow_mut();
    buf.clear();
    buf.resize(n, 0.0f32);
    // ... use buf ...
});
```

## Estado del todo list al momento del backup

1. [in_progress] Wirear SlimeDrafter en main.rs generation loop (MUD_DSPARK=1)
2. [pending] Validar con tests + clippy
3. [pending] Documentar cambios
