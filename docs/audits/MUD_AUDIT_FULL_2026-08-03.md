# Auditoría Integral — Forge LLM / MUD (2026-08-03)

**Fecha:** 2026-08-03
**Alcance:** Código fuente completo (`src/`, `tools/`), hot paths (forward/backward/optimizer), telemetría y logging, documentación vs código (AGENTS.md/GEMINI.md), configuración.
**Baseline verificado:** `cargo clippy --all-targets` → 0 errores / 0 warnings. `cargo test --lib -- --test-threads=2` → **460 passed, 0 failed, 2 ignored**.
**Método:** Revisión manual de archivos clave + análisis forense del log real `mud_train_metrics.log` (1104 líneas `[TELEM]` de una sesión de entrenamiento en vivo, 23 columnas, 0 líneas malformadas) + verificación cruzada de cada hallazgo contra el código (no solo grep).

---

## Resumen ejecutivo

El sistema está sano en línea base (tests + clippy). Sin embargo la auditoría encontró:

- **6 bugs correctos** (2 con riesgo de panic/UB en datos corruptos, 4 de telemetría falsa/engañosa)
- **4 riesgos** (crecimiento de logs sin límite, race de lectura TUI, alloc en hot path KV, falla silenciosa)
- **4 discrepancias documentación↔código** — features presentadas como COMPLETADAS en AGENTS.md que no existen o están a medio hacer
- **~6 mejoras** de calidad/código muerto

Ninguno de los hallazgos es de seguridad de alta severidad (SEC-H backlog sigue cerrado).

---

## 1. BUGS

### BUG-1 — `decompress_block` puede panicear con datos corruptos (P-17 violado)

**Archivo:** `src/mud/kv_compressed.rs:240-268`
**Severidad:** BUG (panic en runtime sobre datos de entrada)

El path de decompresión LZ4 del KV cache usa `unwrap_or_default()` sobre un `Result` de `lz4_flex::decompress`:

```rust
let decompressed = lz4_flex::decompress(&block.compressed, block.original_size)
    .unwrap_or_default();               // ← &[] en fallo
let decompressed_f32: &[f32] = if decompressed.len() == actual_elems * 4 {
    unsafe { std::slice::from_raw_parts(decompressed.as_ptr() as *const f32, actual_elems) }
} else {
    &[]                                 // ← slice vacío
};
```

Las consecuencias con un fallo de decompress (datos corruptos, versión distinta de lz4_flex, block truncado):

| Línea | Código | Efecto |
|-------|--------|--------|
| `:256` | `out[..first_token_elems].copy_from_slice(&block.base)` | panic si `block.base.len() < first_token_elems` — **nunca se valida** |
| `:263` | `out[curr_start + i] = out[prev_start + i] + decompressed_f32[curr_start + i]` | index panic (slice vacío) en path delta |
| `:268` | `out[..actual_elems].copy_from_slice(decompressed_f32)` | panic (empty vs non-empty) en path sin delta |

**Nota:** el guard `out.len() < actual_elems` de `:234` valida el *output*, no la *entrada* descomprimida. El patrón actual es un fix a medias: el `else → &[]` "arregla" el caso largo pero empeora el path delta (`:263` itera `valid_tokens` veces sobre un slice vacío).

**Fix sugerido:** validar `decompressed.len() == actual_elems * 4` y `block.base.len() >= first_token_elems` ANTES de cualquier slice, devolviendo `0` (P-17 fail-fast/skip+log, nunca panic). Usar `decompress_into` con el buffer pre-allocado (ver RIESGO-3).

---

### BUG-2 — Deref no-alineado de `*const u32` en `unpack_ternary2bit_to_f32` (UB formal)

**Archivo:** `src/mud/slime_backward.rs:204` (deref en `:213` y `:221`)
**Severidad:** BUG (UB formal, reprochable con miri)

```rust
let packed_u32 = packed.as_ptr() as *const u32;   // ← cast de &[u8] (alignment 1)
...
let val = unsafe { *packed_u32.offset(u32_base + g as isize) };   // ← deref no-alineado
```

Si `packed` (un `&[u8]`) no está alineado a 4 bytes, el deref es UB según el modelo de Rust. **Callers en producción:**

- `src/main.rs:258` — `emb_packed` desde `owned_data.as_ptr()` de un `Vec<u8>` (alignment 1) o `data_ptr` mmap (offset arbitrario)
- `src/mud/inference.rs:386` y `:434`
- `src/mud/pointer_audit.rs:145`

Funciona en x86 (deref tolerante a alineación), pero es exactamente la clase de bug que miri ya detectó en `kv_compressed.rs` (2 bugs de alineación corregidos el 2026-07-28) y que luego se corrigió con `copy_nonoverlapping` + buffers alineados.

**Fix sugerido:** `std::ptr::read_unaligned` en los 2 derefs, o verificar `packed.as_ptr().align_offset(4) == 0` con fallback. Run `cargo +nightly miri test` sobre este path tras el fix.

---

### BUG-3 — Panel ΔW del TUI: flag binario mostrado como porcentaje

**Archivos:** `src/mud/checkpoint.rs:133-137` (emisor) / `tools/train_telemetry.rs:279-287` + `:1348-1353` (parser + render)
**Severidad:** BUG (telemetría engañosa)

El emisor escribe `[DW] bytes=1404 total=743424 prq=0.0000 moved=1` donde `moved` es un bool serializado como 0/1. El TUI:

```rust
fn absorb_dw(&mut self, kv: &HashMap<String, String>) {
    let moved = App::parse_val(kv, "moved").unwrap_or(0.0);   // ← 0.0 o 1.0
    ...
    self.dw_moved.push(x, moved);                              // ← el % real se ignora
}
```

`render_status_bar` formatea `dw={:.1}%` → el panel muestra siempre **"0.0%" o "1.0%"**, nunca el porcentaje real de bytes movidos (que en el ejemplo es 1404/743424 = 0.19% y está en la misma línea `[DW]`).

**Fix sugerido:** en `absorb_dw` computar `bytes/total*100` (las claves `bytes` y `total` ya se emiten), o emitir el pct directamente en `checkpoint.rs`.

---

### BUG-4 — Métricas C-MUD hardcodeadas a 0 en el emisor de telemetría

**Archivo:** `src/mud/corpus_trainer/session.rs:1625`
**Severidad:** BUG (feature de telemetría muerta / engañosa)

La línea `[TELEM]` incrusta literalmente en el format string:

```rust
" cmud_steps=0.0000 cmud_lock=0.0000 cmud_herm=0.0000 cmud_spread=0.0000 cmud_cauchy=0.0000 n_act={}"
```

**Verificado:** 100% de las 1104 líneas del log traen esos 5 valores en cero. El TUI tiene un `Mode::Cmud` completo (`tools/train_telemetry.rs:1187-1283`, 4 charts + leyenda) que muestra líneas planas en cero siempre. El trainer jamás puebla las métricas reales del `ThinkingState` de C-MUD.

**Fix sugerido:** emitir métricas reales del `ThinkingState` cuando `MUD_CMUD_THINK=1` (y los ceros cuando esté apagado, o directamente omitir el campo), o eliminar el modo C-MUD del TUI (P-08).

---

### BUG-5 — Punto LR fantasma en el TUI por parsing de la cabecera del log

**Archivo:** `tools/train_telemetry.rs:399-401`
**Severidad:** BUG menor (dato falso en chart)

`run_app` parsea cualquier línea que contenga la subcadena `TELEM`:

```rust
if line_buf.contains("TELEM") {
    let kv = App::parse_kv(&line_buf);
    app.absorb_kv(&kv);
}
```

La cabecera de sesión `# MUD_TELEMETRY v2 unix_ts=... lr=0.000500 opt=adam` **contiene "TELEM"** → `absorb_kv` inserta un punto `lr` en x=0.0 (verificado con el parser real: `{'lr': '0.000500', ...}`). El chart de LR arranca con un punto que no corresponde a ningún step.

**Fix sugerido:** skip de líneas que empiecen con `#` en `run_app` (las cabeceras son comentarios).

---

### BUG-6 — `checkpoint.rs:448` imprime con `println!` sin gate del TUI circuito

**Archivo:** `src/mud/checkpoint.rs:448-455`
**Severidad:** BUG (contaminación de TUI)

El reporte ΔW post-sync usa `println!` directo:

```rust
println!("{}", crate::mud::trainer_ui::note(
    if moved { "ok" } else { "warn" }, ...));
```

El módulo `corpus_trainer/` gatea sus `println!` con la macro de `mod.rs:4-15` (env `MUD_CIRCUIT_TUI`), pero `checkpoint.rs` **está fuera** de ese módulo → cuando corre el circuito con TUI (`./mud.sh circuit`), este `println!` contamina la pantalla crossterm en cada checkpoint sync.

**Fix sugerido:** mover el gate a una utilidad compartida (p. ej. `trainer_ui::note` ya existe — hacer que respete `MUD_CIRCUIT_TUI`), o gatear localmente.

---

## 2. RIESGOS

### RIESGO-1 — `mud_train_metrics.log` crece sin límite, multi-sesión concatenada

**Archivo:** `src/mud/corpus_trainer/session.rs:919-923`
**Severidad:** RIESGO medio

```rust
let mut telemetry_file = OpenOptions::new()
    .create(true)
    .append(true)    // ← nunca trunca
```

- Medido: ~2 líneas/chunk ≈ 520 B → 574 KB en 1104 chunks. A millones de chunks son GBs.
- Cada sesión añade su propia cabecera `# MUD_TELEMETRY` y el archivo mezcla sesiones (el doc forense `TRAIN_TELEMETRY_FORENSIC_2026-07-20.md` ya documentó sesiones antiguas con formato de 15 campos mezcladas).
- **Consecuencia colateral importante:** `src/mud/loss_cert.rs:262-283` (`parse_metrics_log`) y `tools/autofinetune.rs` parsean el archivo **entero** — un salto de loss en el borde entre sesiones puede disparar falsamente el gate loss_cert.

**Fix sugerido:** `truncate(true)` al inicio de sesión (con backup del anterior), o rotación por tamaño. Las cabeceras ya dicen "v2" sugiriendo que se planeó el versionado.

### RIESGO-2 — Race de lectura a mitad de línea en el TUI

**Archivo:** `tools/train_telemetry.rs:393-411`
**Severidad:** RIESGO bajo (no observado, sin protección)

`run_app` hace `seek(last_pos)` + `read_line` + parse inmediato. Si un poll cae entre el `writeln!` y el `flush` del trainer, la línea parcial se parsea como completa (valores cortados) y `last_pos` avanza más allá → el fragmento se pierde sin re-intento. No se observó en el log (write ≈ 1 syscall), pero no hay protección.

**Fix sugerido:** no parsear si `read_line` no terminó en `\n` (devolver el excedente al buffer).

### RIESGO-3 — `decompress_buf` pre-allocado NUNCA usado + 1 alloc `Vec` por decompress

**Archivo:** `src/mud/kv_compressed.rs:119, 131, 238-241`
**Severidad:** RIESGO (P-01 violado en hot path KV)

- `decompress_buf: Vec<f32>` se asigna en `new()` y jamás se toca (campo muerto).
- El comentario `:238` dice *"use pre-allocated buffer to avoid allocation (P-01)"* pero la llamada real `lz4_flex::decompress(...)` aloca un `Vec` nuevo **por decompress**.
- Además `:242` hace `from_raw_parts` del `Vec<u8>` como `&[f32]` — el mismo patrón de alineación del BUG-2 (ya corregido en los paths CellV5 previos, no aquí).

**Fix sugerido:** `lz4_flex::decompress_into(&[u8], &mut [u8])` con `decompress_buf` (casteado con alineación garantizada via buffer u8 + `copy_nonoverlapping`, como los fixes miri 2026-07-28), o borrar el campo.

### RIESGO-4 — `logs/circuit.log`: falla silenciosa + append infinito sin rotación

**Archivo:** `src/mud/trainer_ui.rs:102-109`
**Severidad:** RIESGO bajo

`OpenOptions::create(true).append(true).open("logs/circuit.log")` sin `create_dir_all`: si `logs/` no existe el open falla y la telemetría del circuito se pierde en silencio (verificado: `logs/` existe hoy pero vacío — el único que lo crea es `mud.sh`). Sin rotación ni cap (solo `MUD_CIRCUIT_MAX_CYCLES` limita las iteraciones).

**Fix sugerido:** `std::fs::create_dir_all("logs")` antes del open + log de error si falla.

---

## 3. Discrepancias DOCUMENTACIÓN ↔ CÓDIGO

Features que AGENTS.md/GEMINI.md presentan como COMPLETADAS y el código contradice (verificado, no inferido):

| # | Claim en AGENTS.md | Realidad en código | Verificación |
|---|---|---|---|
| D-1 | "Fase 1.1 ASM Fenwick batch prefix sum COMPLETADO (2026-07-28: `src/asm/fenwick_avx2.s` + FFI en `asm/mod.rs` + wrapper en `fenwick.rs`; ~4× speedup)" | `fenwick_avx2.s` **no está en build.rs** (`.file(...)`), **sin FFI** en `asm/mod.rs` (los 33 externs no lo incluyen), **sin wrapper** en `fenwick.rs`. El código usa loop escalar (intrinsics fake-SIMD). | `grep fenwick_avx2` → 0 hits fuera de `src/asm/` |
| D-2 | "CellV5 Fase 2 ASM kernels COMPLETADO (3 batch kernels: gauss_mul, wave_collapse, phase_repulsion; wireados en `asm/mod.rs` + `slimex_asm.rs`)" | `cmud_cell_v5_avx2.s` no compilado, sin FFI. El único kernel ASM C-MUD real es `cmud_gauss_mul_batch_avx2` (`cmud_batch.s` ✓, usado en `cmud_thinking.rs:52`). | `grep cellv5_gauss_mul` → 0 hits |
| D-3 | "`MUD_KV_COMPRESS_CELLV5` env IMPLEMENTADO (Fase-3 CellV5 KV pre-compression: F32→i8→Delta→LZ4, 16-24× lossy, delta-encoding per-i8)" | `src/mud/kv_compressed.rs:55-61`: flag `cellv5_reserved` solo loguea *"MUD_KV_COMPRESS_CELLV5 not yet implemented. Falling back to F32 delta+LZ4"* | `cellv5_reserved` → 0 uso funcional |
| D-4 | "Professional Training Banner agregado (`trainer_ui.rs:print_training_banner` + `TrainingBannerInfo` struct; llamado desde `session.rs`)" | **0 matches** de `print_training_banner` / `TrainingBannerInfo` en todo el repo | `grep` global |

**Extra verificado:**

| D-5 | "`slimex_cell_v5_avx2.s` wireado (SX-02)" | Se compila (build.rs:16) pero tiene **0 callers** en Rust desde que se borró `slimex_asm.rs` (2026-07-29). Peso muerto en el binario + tiempo de assembler por build. | `grep slimex_unpack_cells_v5` → 0 hits |
|---|---|---|---|
| D-6 | `adam_step.s` "reserved for full Adam moments path" | Compilado con FFI (`adam_step_avx2` en `asm/mod.rs:143`) pero sin callers de producción — solo tests? | verificar con grep antes de tocar |

**Impacto:** la documentación (AGENTS.md, GEMINI.md, session reports) sobreestima el estado real del sistema en 4-6 items. Recomendación: corregir las secciones "COMPLETADO" correspondientes o wirear el código (P-08 dice *borrar*, no *archivar*).

---

## 4. MEJORAS / CÓDIGO MUERTO / RUPTURAS MENORES

### M-1 — 5 claves de telemetría parseadas que el emisor NUNCA emite
`absorb_kv` parsea `gate_mean`, `gate_std`, `stp_λ_eff`, `L_STP`, `loss_cv` (`train_telemetry.rs:235-249`) pero ningún emisor las escribe (grep global: 0 hits). Consecuencia: el chart **"Gate Mean"** del modo JEPA (`:1057-1068`) muestra siempre "NO DATA"; `stp_lambda_eff`/`loss_cv` son dead code. Caso inverso: `n_act` **se emite** (`session.rs:1648`) pero ningún chart la muestra.

### M-2 — `load_shadow_bf16` write-only
`adam_state.rs:150` — `store_shadow_bf16` (`optim.rs:434/453`) escribe `shadow_bf16` en cada step con `MUD_BF16_SHADOW=1`, pero nada en producción lo lee (solo tests). Costo CPU + memoria por step sin beneficio. Wirear un reload real (checkpoint bf16) o eliminar (P-08) y ajustar la claim "D2 bf16 shadows COMPLETADO".

### M-3 — `AdamWOptimizer` muerto con bias-correction rota
`src/mud/optim.rs` (decl ~700) — sin callers fuera de tests; `bias_correction = 1.0` en `:641` invalida la corrección AdamW que la rama Adam sí aplica. Borrar o corregir (P-08).

### M-4 — `v_tmp: [f32; 256]` + copy por token en loop de atención
`src/mud/slime_forward.rs:1207-1228` — array en stack + `copy_from_slice` por iteración sobre el rango de atención. No es heap (P-01 OK) pero copia `head_d` elementos por token; hoisteable a `SlimeWorkspace`.

### M-5 — NaN en el primer frame del TUI
`train_telemetry.rs:789-793` — con `loss_data` vacío, `last_val=NaN` → `(NaN*5).floor().clamp() as usize` = 0 (DeepV) pero `height_color(NaN)` = rojo Peak; label y color contradictorios y título "Loss | NaN ↓". No panics (casts saturan) pero es feo en el primer frame.

### M-6 — Panics en paths de archivo en librería
`src/mud/checkpoint.rs:173/:177` — `skills.get_mut("core").unwrap()` / `expect("Missing token_embd.weight")` sobre un `.mud` malformado → panic en library en lugar de `Result` (input del usuario).

### M-7 — `varj ≈ 1/π` constante (a investigar)
En las 1104 líneas del log, `varj` oscila 0.318010–0.319673 (banda 0.5%) alrededor de 1/π = 0.3183099; `σ` casi constante (49.1–50.2%). O las activaciones están tan normalizadas que es legítimo, o `jepa_z`/`jepa_var_ema` están degenerados y la métrica da falsa sensación de salud. Verificar que `jepa_z` se actualiza realmente.

### M-8 — `loss_certification_bench.rs:106` trunca el log del trainer en vivo
`fs::write(metrics_path, "")` — si el trainer está corriendo hay dos escritores con políticas opuestas (append vs truncate) sobre el mismo path. El TUI tolera el truncate (`len < last_pos`, `:387-390`) pero borra el historial de la sesión activa.

---

## 5. Verificaciones OK (sin problema)

- **ASM en uso (20 `.s` compilados + 33 externs):** consistente — `ternary_gemv/4/8rows`, `math`, `pack`, `lm_head`, `jepa_ou`, `backward_rmsnorm/rope/ternary_gemv(8rows)`, `mhc_gate`, `cmud_batch`, `apply_scales` ✓
- **AVX2 en `ezop.rs`:** todos con `loadu/storeu` + tail escalar + guardas NaN; `exp_avx2` clampa ±127 antes de `cvtps_epi32` y blend NaN ✓
- **Fenwick:** 1-indexado consistente entre build/query/find_kth; `build_from_linear_raw` gated por `is_x86_feature_detected!("avx2")` ✓
- **Métricas reales en el log:** `grad_norm=0.0946 entropy=8.152 stp_loss=0.51360` — columnas 8/9/10 ya no son `0.0` (fix 2026-07-31 funciona) ✓
- **TUI resiliente:** archivo faltante → corre vacío "NO DATA"; rotación por inode y truncate detectados ✓
- **`store_dense_k/v`, `load_dense_k/v`:** slicing safe (OOB → panic, no UB) ✓

---

## 6. Orden de trabajo recomendado

| Prioridad | Item | Tipo | Esfuerzo |
|---|---|---|---|
| 1 | BUG-1: validar decompress antes de slices | bug + P-17 | ~10 min |
| 1 | BUG-2: `read_unaligned` en unpack_ternary2bit | bug + miri | ~10 min |
| 2 | BUG-3: `[DW]` % real en el TUI | bug telemetría | ~5 min |
| 2 | BUG-4: métricas C-MUD reales o eliminar modo | bug telemetría | ~30 min |
| 2 | BUG-5: skip de líneas `#` en TUI | bug telemetría | ~2 min |
| 2 | BUG-6: gate `MUD_CIRCUIT_TUI` en checkpoint.rs:448 | bug TUI | ~5 min |
| 3 | RIESGO-1: truncate/rotación de `mud_train_metrics.log` | riesgo | ~15 min |
| 3 | RIESGO-3: `decompress_into` con buffer pre-allocado | P-01 | ~20 min |
| 4 | D-1..D-6 + P-08: borrar 4 ASM muertos + corregir AGENTS.md | doc/código | ~30 min |
| 5 | RIESGO-2/4, M-1..M-8 | mejoras | variable |

---

## 7. Hallazgo contextual adicional

Durante la auditoría, `mud_train_metrics.log` (566 KB, presente al inicio) **desapareció del filesystem** sin que hubiera un proceso trainer activo en ese momento (verificado con `ps`). Coherente con RIESGO-1 (sin protección de rotación), pero no se encontró código que lo borre — probablemente limpieza manual/gestión externa. Vale confirmar que ninguna herramienta (p. ej. `loss_certification_bench`, ver M-8) lo trunca/borra en el flujo normal.
