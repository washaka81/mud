# MUD Session Report — 2026-07-31 (mHC sync FIX + Checkpoint speed)

## Resumen

| Área | Estado | Tests | Clippy |
|------|--------|-------|--------|
| mHC sync fix | ✅ COMPLETADO | 423 ✅ | 0 ✅ |
| Checkpoint packing speed | ✅ COMPLETADO | 423 ✅ | 0 ✅ |

## 1. BUG CRÍTICO: mHC alpha/beta no persistidos en checkpoint save (FIX #47)

### Síntoma

El entrenamiento mHC (`mhc_scale_sgd_step` en `optim.rs:221`) actualizaba los valores
de `mhc_alpha` / `mhc_beta` in-place en los buffers shadow (`mhc_alpha_shadow` /
`mhc_beta_shadow` en `SlimeLayer`), pero `sync_shadow_to_mud` (checkpoint.rs:135)
**nunca** escribía estos valores de vuelta a `core.tensors["blk.N.mhc_alpha"]` /
`["blk.N.mhc_beta"]` en el archivo `.mud`.

→ Los valores mHC entrenados se perdían en cada `save_checkpoint` / final sync.
El entrenamiento mHC era **silent corruption** (los valores se zeroaban/reseteaban
a los valores originales del .mud base en cada checkpoint).

### Causa raíz

Dos problemas:

1. **sync_shadow_to_mud no tiene acceso a `layers`**: la función recibía `shadow_layers`
   (que contiene weight/PRQ scale shadows) pero no `layers: &[SlimeLayer]` (que
   contiene `mhc_alpha_shadow` / `mhc_beta_shadow`). Nunca se pasaba la información
   de mHC.

2. **hash_trained_weights no detecta mHC**: la función solo hasheaba
   `t.owned_data.as_deref()`. Los tensores mHC son Float32 (mmap-backed,
   `owned_data = None` antes del sync), por lo que eran **invisibles** al no-op
   check. El hash de entrada y salida eran idénticos aunque mHC se entrenara.

### Fix

**`sync_shadow_to_mud`** (checkpoint.rs):
- Firma extendida: agregado `layers: &[crate::mud::slime_forward::SlimeLayer]`
- Nueva función `sync_mhc_shadows()` itera layers → para cada layer con
  `mhc_alpha_shadow.is_some()`, llama `sync_f32_shadow_tensor()` que:
  - Lee los bytes del shadow buffer (`&[f32]` → `&[u8]` via `from_raw_parts`)
  - Compara old-vs-new para ΔW tracking (bits_changed / bytes_total)
  - Llama `write_owned_bytes` para persistir en `core.tensors["blk.N.mhc_alpha"]`

**`hash_trained_weights`** (checkpoint.rs):
- Extendido: cuando `owned_data` es None, intenta leer desde `data_ptr` (mmap-backed)
  con tamaño calculado desde `shape × element_size`. Así el no-op check ahora
  detecta cambios en mHC (mmap-backed antes del sync) y cambios post-sync
  (owned_data después).

**Call sites actualizados:**
- `mod.rs` shim: `sync_shadow_to_mud` y `save_checkpoint` extended con `layers` param
- `session.rs`: 4 call sites (2x `save_checkpoint` periódico, 1x `sync_shadow_to_mud`
  final, 1x en `run_debate_session` via `debate.rs`)
- `debate.rs:497`: agregado `&layers`

### Equivalencia verificada

- mHC alpha/beta son Float32 (shape `[hidden_size]`), no Ternary2Bit → no requiere
  packing ternario. El sync es un simple `copy_from_slice` de 4 bytes × hidden_size.
- `pack_ternary_into` (usado para weights) no se aplica a mHC.
- Tests de packing `test_pack_ternary_basic` + `test_pack_ternary_all_zero` +
  `test_direct_pack_matches_reference` siguen verdes (validan que el reemplazo
  de packed loop produce idéntico output byte-for-byte).

## 2. Speed Optimizations en checkpoint sync

### Cambio 1: Batch packing (1 store/8 elementos → 8× menos stores)

**Antes** (checkpoint.rs:190-213, 611-656): loop scalar per-element con
read-modify-write:
```rust
for c in 0..cols {
    let val = (w * s_inv).round().clamp(-1.0, 1.0);
    let bit = if val > 0.5 { 0x1 } else if val < -0.5 { 0xF } else { 0x0 };
    let u32_val = ptr::read_unaligned(packed.add(u32_idx));  // READ
    let u32_new = u32_val | (bit << shift);
    ptr::write_unaligned(packed.add(u32_idx), u32_new);      // WRITE
}
```
8 RMW por u32 word → 8 reads + 8 writes = 16 operaciones.

**Después**: `pack_ternary_into` batch (ezop.rs:371):
```rust
// 1 pass: zero all words (sequential, cache-friendly)
for i in 0..u32_count { *out.add(i) = 0; }
// 1 store per word (no RMW)
while i < full {
    let mut word = 0u32;
    for j in 0..8 { /* OR bit */ }
    *out.add(i/8) = word;  // SINGLE STORE
    i += 8;
}
```
8 writes → 1 store + 8 ORs. Aproximadamente **8× menos stores** en la packing phase.

**Equivalencia ternaria verificada:**
- Old: `round(w · s⁻¹) > 0.5` ⟺ `w > 0.5 · s` (para w positivo)
- New: `pack_ternary_into(w, cols, delta=0.5·s)` → `w.abs() > 0.5·s`
- Ambas clasifican los mismos valores como {+1, 0, -1}.

### Cambio 2: Eliminado double-clone (48 MB → 0 bytes copiados)

**Antes** (checkpoint.rs:251):
```rust
let packed_copy = SYNC_PACKED_BUF.with_borrow(|p| p.clone());  // ~48MB clone
write_owned_bytes(emb_tensor, &packed_copy);                     // ~48MB copy
```
Doble copia de ~48 MB para el embedding de smollm2 (151643×128).

**Después**:
```rust
let packed_move = SYNC_PACKED_BUF.with_borrow_mut(std::mem::take);
write_owned_bytes(emb_tensor, &packed_move);
```
`mem::take` extrae el Vec de la thread-local (reemplazándolo con Vec::new()) y se
usa directamente como `&[u8]` via Deref. **Cero clone, cero doble-copy.**

### Cambio 3: Scales buffer — mismo patrón `mem::take`

```rust
// Antes: SYNC_SCALES_BUF.with_borrow(|s| s.clone()) + copy
// Después: SYNC_SCALES_BUF.with_borrow_mut(std::mem::take)
```

### Cambio 4: Path `scales_only` en `update_layer_tensor`

Mismo patrón aplicado: `mem::take` en vez de clone + with_borrow closure.

### Impact estimado

Para el embedding de smollm2 (151643 × 128 = 19.6M elementos):
- Packing: ~19.6M reads + 19.6M writes (RMW) → ~2.5M stores (batch) = **~8× menos stores**
- Clone elimination: **~48 MB de copia eliminada** por sync de embedding
- Scales: **~768 KB de copia eliminada** por sync de scales (151643 × 4 bytes)

Velocidad de training: ~7.6 tk/s → se espera mejora moderada en checkpoint sync
(cuellos de batalla de I/O siguen dominando, pero CPU overhead reducido ~8×).

## 3. Files modified

| File | Change |
|------|--------|
| `src/mud/checkpoint.rs` | mHC sync (new signature + `sync_mhc_shadows` + `sync_f32_shadow_tensor`); `hash_trained_weights` mmap-backed path; batch `pack_ternary_into` in both packing loops; `mem::take` for packed/scales buffers |
| `src/mud/corpus_trainer/mod.rs` | `sync_shadow_to_mud` + `save_checkpoint` signatures extended with `layers` param |
| `src/mud/corpus_trainer/session.rs` | 4 call sites updated to pass `&layers` |
| `src/mud/corpus_trainer/debate.rs` | Call site updated to pass `&layers` |
| `tools/train_telemetry.rs` | Removed Model cursor; dynamic loss profile title (last, dir, delta, rate, band, steps); band legend overlay; 5-label y-axis; dynamic last-value on all charts; status bar entropy + DW |
| `AGENTS.md` | New 2026-07-31 session entry |
| `ROADMAP.md` | Batch D/E statuses → DONE; mHC sync FIX entry |
| `docs/sessions/MUD_SESSION_REPORT_2026-07-31_MHC_SYNC_FIX.md` | Session report created (this file) |

## 4. TUI Telemetry Improvements

### Loss Profile: cursor removed + dynamic window
- **Removed** the "Model" scatter ball (single-point scatter at last data point) — redundant visual cursor.
- **Dynamic title**: `Loss Profile | last=6.02 ↓ descending Δ-0.034 rate -0.001/step | Plateau | zoom 1.0x | steps=247`
- **Band legend overlay** at chart bottom: `[Deep Valley] │ [Valley] │ [Plateau] │ [Hill] │ [Peak]` with current band bracketed.
- **5-label y-axis** (0%, 25%, 50%, 75%, 100%) instead of 3.

### All charts: dynamic last-value
- `make_chart` appends `| last=X.XXX ↓` to title automatically.

### Status bar: enriched
- Added `entropy=X.XXX` and `dw=X.X%`.

## 5. Verification

```bash
cargo clippy --all-targets      # 0 warnings ✅
cargo test --lib -- --test-threads=4   # 423 passed ✅
```
