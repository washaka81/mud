# MUD Session Report — 2026-07-29

## F32 Backward + Checkpoint Thread-Local Lifetimes

### Objetivo
Revisar la telemetría, identificar cuellos de botella en CMUX/SLIMEX training y optimizar.

### Hallazgos

#### CMUX / SLIMEX no son cuellos de botella
- **CMUX** (`MUD_CMUD_THINK`): OFF por defecto, no conectado al path de entrenamiento. Módulo inference-only.
- **SLIMEX** (`MUD_SLIMEX`): OFF por defecto, `ShadowExpertBus` + `SlimeXCellV5` pre-allocados pero nunca activados durante training.

#### Bottleneck real: F32 backward era un no-op
El modelo `models/smollm2.mud` usa F32 weights (conversión Q2_0→F32). El backward pass en `backward_slime_block` (`slime_backward.rs:524-527`) **saltaba completamente** las capas F32:

```rust
if layer.is_f32_weights {
    let n = grad_out.len().min(grad_in.len());
    grad_out[..n].copy_from_slice(&grad_in[..n]);
    return;  // ← Sin gradientes de pesos, sin actualizaciones del optimizer
}
```

Esto significaba que **el optimizer no tenía nada que actualizar** para modelos F32 — el entrenamiento no estaba ocurriendo.

#### Bottleneck secundario: allocations en `sync_shadow_to_mud`
Aunque se habían agregado thread-local buffers (`SYNC_PACKED_BUF`/`SYNC_SCALES_BUF`), el código no compilaba debido a 4 errores de lifetime en el patrón `LocalKey::with` que retornaba un `RefMut`.

### Cambios aplicados

#### 1. `src/mud/checkpoint.rs` — Thread-local lifetime fixes
- Reestructurado `sync_shadow_to_mud` (emb tensor) para usar `with_borrow_mut`/`with_borrow` con todo el trabajo dentro del closure
- Reestructurado `update_layer_tensor` (layer tensors) — primero lee metadata sin borrow mutable, luego usa thread-locals, luego copia resultados y escribe a `core.tensors`
- Eliminado `Vec::with_capacity` + `.to_vec()` en el path `scales_only` de `update_layer_tensor`
- F32 embedding path: eliminado `.to_vec()`, usa `&[u8]` view directo

#### 2. `src/mud/slime_forward.rs` — `f32_gemv_backward`
Agregada función `f32_gemv_backward` (líneas 448-481):
- **Weight gradient**: `dL/dW[row, col] += grad_y[row] * x[col]` (outer product)
- **Input gradient**: `dL/dx[col] += sum_r grad_y[r] * W[r, col]` (GEMV)
- Zero-alloc, raw pointer writes (P-00/P-01)
- 2 tests: `test_f32_gemv_backward_matches_reference`, `test_f32_gemv_backward_accumulates`

#### 3. `src/mud/slime_backward.rs` — F32 backward completo
Reemplazado el no-op por backward completo para F32 weights (líneas 524-705), cubriendo todos los 8 pasos:
1. FFN Down Projection Backward (`f32_gemv_backward`)
2. FFN ReLU² Activation Backward
3. FFN Up & Gate Projection Backward (`f32_gemv_backward`)
4. FFN RMSNorm Backward
5. Attention O Projection Backward (`f32_gemv_backward`)
6. SDP Attention Backward (STE Proxy)
7. Attention Q, K, V Projection Backward (`f32_gemv_backward`)
8. Attention RMSNorm Backward

### Verificación
- `cargo check --release` ✅
- `cargo clippy --all-targets` ✅ (0 warnings)
- `cargo test --lib -- --test-threads=2` ✅ (413 passed, 0 failed, 2 ignored)

### Archivos modificados
- `src/mud/checkpoint.rs` — thread-local lifetime fixes, zero-alloc scales_only path
- `src/mud/slime_forward.rs` — `f32_gemv_backward` + 2 tests
- `src/mud/slime_backward.rs` — F32 backward completo (524-705)
- `AGENTS.md` — actualizado Runtime Truth, Kernels, Historical Fix Log, nueva sync session
