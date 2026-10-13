# MUD Session Report 2026-10-13 — Ola 2 completa (F2) + Ola 3 iniciada (R-40, R-48)

**Date:** 2026-10-13
**Objective:** Ejecutar el plan maestro `docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md` (PRIORIDAD 0): cerrar la **Ola 2 (F2 — seguridad de entrada)** e iniciar la **Ola 3 (F3 — eficiencia)**.
**Status:** ✅ Ola 2 **COMPLETA** (R-30…R-38); Ola 3 en curso (R-40 ✅, R-48 ✅).

**Regla R-73 aplicada:** toda afirmación de este informe lleva `(comando, salida, fecha)`. No se incluyen claims por inspección.

---

## 1. Resumen ejecutivo

| ID | Ítem | Evidencia principal | Estado |
|---|---|---|---|
| R-32 | Parser GGUF con cotas | `cargo test --lib --release --features proptest` → 500/0/2 (12 tests R-32 + 2 proptest fuzz) | ✅ |
| R-33 | ECC valida paridad contra extent base | Tests R-33 ×2; suite 507/0/2 | ✅ |
| R-34 | `worm::get_embedding_row` exige Float32 rank-2 | Tests R-34 ×4 | ✅ |
| R-35 | GEMV exacto para `n%8∈1..7` + stride `div_ceil(8)` | Test R-35 falla antes (git stash) y pasa después | ✅ |
| R-36 | `unsafe` justificado + gate | `./mud.sh unsafe-audit` → `OK - 0 sin comentario (R-36)` (532 productivos) | ✅ |
| R-37 | RCE feature-gated documentado + confinamiento | `SECURITY.md` + 2 tests de gate por build | ✅ |
| R-38 | Symlink escape | Tests R-38 ×4 (dangling/symlink-out/ancestro/interno) | ✅ |
| R-40 | LM head ternario batcheado (sin f32) | A/B `ale.mud`: greedy **4.6×**; RSS **201→127 MB**; logits idénticos | ✅ |
| R-48 | `MUD_GPU_GEMV` doc==código | `./mud.sh env-audit` → 0 divergencias; decisión escrita | ✅ |

**Gates finales (2026-10-13):**

```text
cargo test --lib --release --features proptest -- --test-threads=2
  → test result: ok. 515 passed; 0 failed; 2 ignored
cargo clippy --all-targets --all-features -- -D warnings
  → Finished (0 warnings)
./mud.sh unsafe-audit
  → [UNSAFE-AUDIT] unsafe: 532 · sin justificar: 0  /  OK - 0 sin comentario (R-36)
./mud.sh env-audit
  → OK — 0 divergencias (R-05)
```

**Bloqueo pre-existente (no introducido en esta sesión):** `cargo test --all-targets --release` no compila por `[profile.release] panic="abort"` vs `tests/test_tokenizer.rs` (integración), la colisión de nombre `[lib] forge_llm` + `[[bin]] forge_llm` y carreras de link (`cc` OOM esporádico). Anotado en el plan; alcance de limpieza F5.

---

## 2. Ola 2 (F2) — detalle

### R-32 — Cotas del parser GGUF (`src/gguf/mod.rs`) — commit `8269222`

**Antes:** parser sin cotas — un `.gguf` truncado/hostil podía hacer panic por slicing, `Vec::with_capacity(n)` multi-GB, recursión ilimitada de arrays anidados, `alignment` hostil (0 → underflow en `data_start`), offsets sin validar.

**Ahora:**
- `GgufCursor` con `need()`/`read_bytes()` bounds-checked: fichero truncado ⇒ `Err`, nunca panic.
- Caps SEC-H1 (reutilizados de `src/mud/mod.rs`): `MAX_META_ENTRIES`, `MAX_TENSOR_ENTRIES`, `MAX_STRING_LEN`, `MAX_SHAPE_DIMS`, `MAX_TENSOR_DIM` (1<<28) + producto de shape con `checked_mul`.
- Arrays de metadata: cap `GGUF_MAX_ARRAY_LEN=4M`, reserva incremental (`try_reserve(n.min(1024))`), profundidad ≤ `GGUF_MAX_ARRAY_DEPTH=8`.
- `general.alignment` validado: ≠0, potencia de 2, ≤ `GGUF_MAX_ALIGNMENT=1M`.
- Offsets de tensores: `checked_add` + tamaño exacto por tipo (`gguf_tensor_byte_size`: F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1); K-quants validan al menos el offset.

**Tests:** `r32_bounds_tests` ×12 (truncado, counts, key/string len, array len, nesting, alignment, rank, dim, offset, end, happy-path) + `r32_fuzz` ×2 proptest (`--features proptest`). Ajuste fino: el check `data_start > EOF` se eliminó porque un GGUF vacío legítimo tiene `data_start` en EOF; el gate real son los checks por tensor.

**Fix de build:** el módulo fuzz requiere `#[cfg(all(test, feature = "proptest"))]` (si no, `dead_code` bajo `clippy --all-features`).

### R-33 — ECC parity bounds (`src/mud/mod.rs::ecc_verify_all`) — commit `a4a6c3a`

**Antes:** la paridad se leía con el conteo del tensor **base**: owned truncaba silencioso (`.take()`), mmap podía panic (slice) u over-leer. Ahora:
- Firma `-> anyhow::Result<(u32, u32)>`; `load_verified` propaga con `?`.
- `need = ceil(prod(base.shape)/8)` con `checked_mul`; owned exige `len >= need`; mmap usa `.get()` (offset + end validados).
- El segundo loop usa aritmética checked (sin `product()` con wrap).

**Tests:** `test_r33_short_ecc_is_err` (ECC diminuta ⇒ `Err` R-33), `test_r33_exact_ecc_is_ok`.

### R-34 — `WormSegment::get_embedding_row` (`src/mud/worm.rs`) — commit `a4a6c3a`

**Antes:** un tensor ternario (nibbles) se reinterpretaba como `*const f32` con extent `vocab*hidden` → over-read ~8× de basura como floats. Ahora: exige `Float32` y shape rank-2 `[vocab, hidden]`; dims del **shape** (nunca metadata); el `hidden` del llamante debe coincidir; `checked_mul` + `.get()` + guard de `owned.len()`; devuelve `None` en cualquier discrepancia.

**Tests:** ×4 (f32 ok, ternario ⇒ None, hidden mismatch ⇒ None, token OOB ⇒ None).

### R-35 — GEMV exacto `n%8∈1..7` + stride canónico — commit `a4a6c3a`

**Hallazgo:** los kernels ASM solo cubren múltiplos de 8 — `ternary_gemv.s::.leftover8` procesa un bloque completo de 8 (over-read de `x` + acumula basura), y `ternary_gemv_4rows/8rows` descartan la cola. Además los call-sites usaban `n_in/8` (floor) como stride, cuando el layout canónico R-15 es `div_ceil(8)`.

**Fix (en wrappers Rust, ASM intacto):** `asm::ternary_gemv_tail` (escalar 1..7, nibbles LSB-first idénticos al ASM) cableado en `ternary_gemv`, `ternary_gemv_4rows`, `ternary_gemv_8rows` (`n_main = n & !7` al ASM + tail). Strides: `gemv_gpu.rs` (3 sitios) y `slime_backward.rs` (`w_u32_stride`) → `div_ceil(8)`. GPU intacta (Ash hace bail si `n_in % 8 != 0`).

**Test:** `test_r35_gemv_wrappers_exact_on_remainder` (15 tamaños `n ∈ 1..103`, 1/4/8 filas vs referencia escalar). Verificado que **falla antes** del fix (`git stash` → `FAILED. 0 passed; 1 failed`) y pasa después.

### R-36 — SAFETY + `unsafe-audit` — commit `83536e1`

**Problemática real:** la auditoría estimó "19 `unsafe` sin SAFETY"; la medición mecánica encontró **532 sitios `unsafe` productivos** (excluyendo tests, `#[cfg(test)]` y `src/bin`).

**Entregables:**
- `./mud.sh unsafe-audit`: implementación **awk embebida** en `mud.sh` (sin Python — P-07). Regla: `SAFETY`/`# Safety` en la misma línea o ≤8 previas; operaciones de **alto riesgo** (`from_raw_parts*`, `Mmap::map`, `CStr::from_ptr`, `transmute`, `alloc/dealloc`, `from_ptr`, `as_mut()`, `.offset(`) exigen comentario **adyacente**; el resto admite contrato `//! SAFETY:` de módulo (no decorativo, enumera patrones). Salida: `[UNSAFE-AUDIT] unsafe: 532 · sin justificar: 0`.
- ~40 ficheros documentados: contratos de módulo en kernels (`ezop`, `adam_state`, `slime_forward/backward`, `ash_backend`, `ash_qat_dispatcher`, `sandbox`, `rlvr`) + comentarios per-site en fronteras (`checkpoint`, `inference`, `main`, `train_session/step`, `optim`, `workspace_agent`…).
- Gate en `mud.sh ci` y en `.github/workflows/ci.yml`.
- Diff **solo comentarios** (910+ / 2−; las 2 eliminaciones son comentarios reemplazados).

**Hallazgos reportados por la revisión (posibles bugs, NO corregidos — fuera de alcance R-36, no se taparon):**
1. `fenwick.rs::build_from_linear(&[])` lee 1 elemento OOB vía API **segura** (`n = max(1)` con `data.len() == 0`). Sin llamadores productivos.
2. `adam_state.rs::sparse_adam_step` valida `w/g/state.m` pero no `state.v` (fields `pub`; los constructores mantienen lockstep).
3. `slime_jepa.rs` llama el kernel JEPA con `out_z == z_ptr` (viola la precondición documentada del kernel; funcional por read-before-write).
4. `session/debate/train_step` reinterpretan `prq_scale`/`output_norm.weight` como f32 sin verificar `t_type == Float32` (un `.mud` malformado sería OOB; R-30 valida shape, no tipo en esos sitios).
5. `debate_trainer.rs` chequea AVX2 pero no `bmi1` antes de `find_kth_avx2` (todos los CPU AVX2 reales tienen BMI1).

### R-37 — RCE feature-gated documentado + confinamiento — commit `318063c`

- `SECURITY.md` reescrito: el RCE de `unsafe-rce` (`rustc`) y `unsafe-cli` (`bash -c`) es **estructural** (feature-gated, `default = []`), no "0 RCE"; filas del loader/GGUF/ASM actualizadas a su estado real post-R-30/31/32/36.
- **Tests de gate por build** (se compilan solo con la feature OFF): `test_terminal_sandbox_disabled_by_default` (⇒ `PermissionDenied`), `test_rlvr_critic_disabled_by_default` (⇒ `-1.0` + mensaje).
- `skills/data_analysis.rs`: ya no usa `fs::read_to_string`; lee vía `AgentWorkspace::read_file` (confinado al root; inerte sin workspace montado). Nota: `skills/` no se compila hoy (dead code, decisión R-60).

### R-38 — Symlink escape (`src/mud/workspace_agent.rs`) — commit `318063c`

**Antes (SEC-WS-01):** `Path::exists()` sigue el symlink; un **enlace colgante** cuyo destino está fuera del root se trataba como "fichero nuevo" y `fs::write` seguía el enlace → escritura fuera del workspace.

**Ahora:**
- Resolución por el **ancestro existente más profundo**: `symlink_metadata` (no sigue) + `canonicalize` + `starts_with(root_canon)`; enlace colgante ⇒ `Err(PermissionDenied)` (fail-closed);
- Apertura de hoja con **`O_NOFOLLOW`** (Linux `0o400000` / macOS `0x0100`) como defensa TOCTOU;
- Bug de borde corregido: `resolved_prefix.join("")` añadía slash final ⇒ `ENOTDIR` al abrir un fichero regular (fix: `probe == candidate` ⇒ devolver el prefijo).
- Symlinks que resuelven **dentro** del root siguen permitidos.

**Tests (7 en el módulo, 4 nuevos):** `test_dangling_symlink_rejected`, `test_symlink_to_outside_rejected`, `test_symlink_dir_ancestor_escape_rejected`, `test_symlink_inside_root_ok` (+ 3 preexistentes).

---

## 3. Ola 3 (F3) — detalle

### R-40 — LM head ternario batcheado — commit `988f6a1`

**Antes:** el head se materializaba a f32 denso (`vocab × hidden × 4 B`; 77.6 MB **re-leídos por token** en `ale.mud`) y se ejecutaba con `lm_head_logits_avx2` **single-thread**.

**Ahora:**
- `gemv_gpu::ternary_head_logits` (público): GEMV ELUT por filas 4/8 (`ternary_gemv_4rows/8rows` + `ternary_gemv_tail`) sobre `PCorePool`, con escala PRQ por fila; intenta GPU primero (`try_gpu_ternary_gemv`, política `MUD_GPU_GEMV`) y cae a AVX2.
- `main.rs`: head ternario usa los **packed** del tensor (tied: `token_embd`; untied: `output.weight`) + `prq_scale`; **no se construye la matriz f32** (`out_f32_owned` ya no existe para heads ternarios). El factor de re-escalado (`scale_up` / `scale_up·logit_scale`) se aplica a los logits, idéntico al pre-escalado de las matrices densas.
- Heads Float32 (modelos Q2_0→F32) conservan el camino denso; tipos de head no soportados ⇒ `Err` (P-17).

**Mediciones A/B (mismo protocolo, `models/ale.mud`, 256 tokens, 2 runs):**

| Protocolo | Antes | Después | Mejora |
|---|---|---|---|
| Greedy (`MUD_INFER_GREEDY=1`; cómputo de modelo) | 3.33 s (11.4 ms/tok) | 0.93 s (2.5 ms/tok) | **4.6×** |
| Default nucleus (extremo a extremo) | 4.33 s (15.3 ms/tok) | 1.90 s (6.3 ms/tok) | **2.4×** |
| Baseline **congelado** del plan (22 ms/tok) | 22 ms/tok | 6.3 ms/tok | **3.5×** |
| RSS pico (VmHWM) | 201 MB | **127 MB** | **−74 MB** |

**Paridad numérica:** logits del modelo real con `MUD_DEBUG=1` (min/max/mean/std a 4 decimales + top-10) **idénticos** antes/después (`diff` → `LOGITS IDENTICOS`).

**Tests:** `test_r40_ternary_head_matches_scalar` (40 filas × 24 cols, varias escalas) y `test_r40_ternary_head_non_multiple_of_8` (n=19, tail R-35).

**Cuello restante medido:** el sampling nucleus (softmax + Fenwick sobre ~152k vocab) cuesta ≈3.8 ms/token (60% del tiempo restante) → candidato F3 (R-44 fusión de pases / R-68).

### R-48 — `MUD_GPU_GEMV` doc == código — commit `988f6a1`

Decisión escrita: **producción en CPU** (`mud.sh:48` fija `0`; Iris Xe SIGSEGV en `submit_and_wait`); `auto` queda como opt-in explícito del operador en hosts sin ese bug. `mud.sh` cita R-48 + §1.2; `MUD_ENV_REFERENCE.md §1.2` actualizado; `./mud.sh env-audit` → 0 divergencias.

---

## 4. Commits de la sesión

```text
8269222  R-32: GGUF parser bounds (cursor checked, SEC-H1 caps, depth limit, offset validation); 500/0/2 proptest
a4a6c3a  R-33/R-34/R-35: ECC bounds, worm emb validation, GEMV tail+stride; 507/0/2 proptest
83536e1  R-36: unsafe-audit gate (awk) + SAFETY justification for all 532 production unsafe sites (0 unjustified); CI wired
318063c  R-37/R-38: structural RCE docs + build-gate tests; workspace symlink escape fix (symlink_metadata + O_NOFOLLOW); Ola 2 complete; 513/0/2
988f6a1  R-40+R-48: ternary batched LM head (no f32 materialization; 2.4x default / 4.6x greedy on ale.mud, RSS -74MB, logits parity) + GPU_GEMV doc decision
```

## 5. Estado del plan tras la sesión

- **Ola 0 (F0)** ✅ cerrada · **Ola 1 (F1)** ✅ cerrada · **Ola 2 (F2)** ✅ **cerrada en esta sesión** (R-30…R-38).
- **Ola 3 (F3)** 🟠 en curso: R-40 ✅, R-48 ✅; restan R-41 (RSS emb), R-42 (1.6 bpp), R-43 (round-trips i8), R-44 (hilos/fusión), R-45 (residual f16), R-46 (KV sub-8-bit), R-47 (LUT mpGEMM).
- Progreso: **28/55** (`./mud.sh plan`).

## 6. Próximos pasos

1. **R-41** — eliminar la materialización de `emb_f32` (77.6 MB restantes en RSS): lookup por fila (`dequantize_ternary_row`) + `emb_rms`/`max_emb` en streaming; objetivo RSS < 40 MB (medir VmHWM; `/usr/bin/time -v` no disponible en este host).
2. **R-43** — quitar los 4 round-trips `f32→i8→f32` por capa del forward (`slime_forward.rs`).
3. **R-44** — política de hilos P+E y fusión de pases: el sampling nucleus es ahora el 60% del tiempo por token.
4. **R-42** — formato `TernaryTQ` (5 trits/byte, 1.6 bpw) con lectura del formato antiguo.
5. Hallazgos R-36 (§2) → evaluar como ítems nuevos del backlog (especialmente `fenwick build_from_linear(&[])` y el check de tipo en `prq_scale`).
