# AGENTS.md — Project Context for AI Agents

> **Do not contradict `GEMINI.md`.**  
> Policies (P-#), open ledger (L-##), and status snapshot: **`GEMINI.md`**.  
> Product vision & phases: **`ROADMAP.md`** (consolidado 2026-07-26).  
> This file = identity + architecture map + historical fixes + agent commands.

**Last sync:** 2026-10-11. **AUDITORÍA COMPLETA + PLAN MAESTRO.** Auditoría: [`docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`](docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md). Backlog activo: [`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) (`R-##`, 7 olas). **Verificado en vivo (no inspección) — post Ola 0 + R-10/R-11/R-12/R-13 (2026-10-11):** `cargo test --lib --release` **PASA** (466/0/2; UB fenwick purgado R-03, flaky `grad_checkpoint` env-race arreglado via `CheckpointPolicy::from_env`); `cargo clippy --all-targets --all-features -D warnings` **0 warnings**; `cargo test --all-targets --release` **exit 0**; CI workflow **válido** (`actionlint` exit 0; `on:` raíz + schedule + workflow_dispatch; slug `washaka81/mud` — R-01); `./mud.sh ci` usa release + all-targets + all-features (R-02); `git init` + `.gitignore` limpios, primer commit (R-04); `MUD_ENV_REFERENCE.md` + `env-audit` 0 divergencias (R-05); `sync_tied_output_weight` espeja entrenamiento→cabeza evaluada (R-10); **`silu_backward_avx2` (SiLU vs ReLU²) ✅ (R-11)**; **`mhc_activation_backward` con alpha/(1-gate)·beta ✅ (R-12)**; **`TERNARY_DEADZONE` SSOT (conversor↔trainer↔checkpoint 0.7) ✅ (R-13)**; `MUD_AUDIT_LATEST.md` actualizado (R-06). **Restan activos:** **deriva de escala PRQ** ×0.52 por guardado (R-14); wire format ternario = **4 bits/peso** (2.52× la entropía, R-42); **LM head en f32 denso** (77.6 MB/token en `ale.mud`), single-thread (R-40); inferencia medida **~46 tok/s** y salida **incoherente** (F3); sin validación `metadata↔shape` (OOB/OOM, R-30); parser GGUF sin cotas (R-32). Los claims "460 tests ✅", "0 OOB", "Rayon removed", "`MUD_GPU_GEMV` default auto en mud.sh", "Fenwick batch prefix sum COMPLETADO", "wormholes −50 % host compute" quedan **retirados** (detalle en `GEMINI §0.3`). **Regla:** ningún `COMPLETADO` sin `(comando, salida, fecha)` — R-73.

**Last sync:** 2026-07-26. **P0 (AVX2 batch register ops) + P1 (eliminar `registers_tmp`) + P2 (raw pointer workspace `AlignedArray<T>`) + P3 (JEPA SIMD batch 8) + P4 (zero-alloc LM head batch) + P5 (stack alloc ≤ 384) + Q2_0→F32 conversion fix + SlimeX CellV5 (u32 8:8:8:8) + ASM AVX2 unpacker + Memory Continuity Table (256 slots) + AOT Boot Indexer + MudFileMerger + SmartTokenizer + CMUD Thinker (ThinkingState + trainable sidecar) + Sprint 1 (security: 0 RCE **feature-gated**; ⚠️ claim "0 OOB" **retirado 2026-10-11** — ver `GEMINI §0.3`) + Sprint 2 (SlimeX end-to-end wired) + Sprint 3 (CMUD production-trainable) + Sprint 4 cerrado (`MudError` + `optim.rs` leaf + **`corpus_trainer.rs` split 7-módulos ✅**) + **Sprint 5 cerrado (5.1 zero-alloc + 5.2 Mutex→AtomicBool + 5.3 LUT backward + 5.4 branchless pack)** + **Sprints 6-7 cerrados (security docs)** + **F-K streams cerrados (QKV one-CB, multi-expert STE, long full-seq, KV bf16, CSA LSH, loss cert, converter P-13)** + **Fenwick trees (802 LOC, 25 tests, 19.5× speedup)** + **Chat refactor (Fenwick sampling, conversation history, system prompt, temperature)** + **Converter audit (6 findings, 3 fixes, 6 optimizaciones)** + **smollm2→.mud→C-MUD (conversión + entrenamiento + auditoría)** + **MUD-WORM (WormPropagator + JEPARouter + MhCoupler + WormChain, 18 tests, auto-conexión de modelos .mud como modelo único) + Wormhole Router/Network (⚠️ **claim "peer-to-peer bypass host, 50% host compute reduction" RETIRADO 2026-10-11**: `WormholeNetwork::forward` hace el mismo trabajo en ambas ramas con un multiplicador local; no hay red — ver `GEMINI §0.3`, R-67)** + **CellV5 + C-MUD State Packing (4× compresión, 16 KB en L1, 5 tests)** + **KV Cache Compression (Delta + LZ4 + CellV5 Fase-3, 16×, 11 tests)** completados.**  
**Sesión 2026-07-26 (Batch A/B/C/D-partial):** F1 Fenwick nucleus en `debate_trainer.rs` (elimina `vec1..vocab_size` per-token + sort O(n log n) → FenwickF32, patrón `main.rs:438`); F5b Percentile gate (`CertMode::Percentile` + `percentile_loss` via FenwickU32 histograma); Muon/GaLore/Chunked — eliminado `grad.to_vec()` copy y hoistea scratch fuera de loops (P-01 стены ~4 alloc/NS-call → 0/call); SEC-H1 caps en `MudFile::load` (`MAX_META_ENTRIES=4096`, `MAX_TENSOR_ENTRIES=65536`, `MAX_STRING_LEN`, `MAX_SHAPE_DIMS`); SEC-H-H8 `MUD_CIRCUIT_MAX_CYCLES` cap; SEC-H7 `COLLECT_FILES_MAX_DEPTH=32` recursion cap; PERF-7 `forward_last_hidden_and_head` tied-output path clone → `std::mem::take` (zero copy, ~295 MB saved en smollm2); run_trainer absluta + `MUD_TRAIN_CKPT_DIR`/`MUD_TRAIN_CORPUS_DIR` env override; `MUD_KV_COMPRESS_CELLV5` env **NOT IMPLEMENTED (deferred)** — el código (`kv_compressed.rs:68-72`) loguea `RECOGNIZED but not yet implemented`, fallback a Delta+LZ4 sobre i8 (Fase-2, vivo ~16×). La "Fase-3 CellV5 KV pre-compression (F32→CellV5→Delta→LZ4)" nunca se materializó: `MUD_KV_COMPRESS_CELLV5=1` no comprime con CellV5. *(D-3: corr. 2026-08-03 — AGENTS decía IMPLEMENTADO; era el claim de Fase-3 CellV5 que nunca existió).* 407 tests (debug) | clippy 0 ✅ — ⚠️ histórico (ver `GEMINI §0.3`). Documentación completa en `docs/sessions/MUD_SESSION_REPORT_2026-07-26_PENDIENTES.md`.
**Próximas decisiones:** ✅ **TODAS RESUELTAS/DOCUMENTADAS** — F4 Fenwick trees (`src/mud/fenwick.rs`, 947 LOC, 25 tests) **y** Sprints 6-7 (docs/seguridad) ambos completados (2026-07-24). Ver `docs/sessions/MUD_SESSION_REPORT_2026-07-24_SPRINT_5.md`. **Fase 1.1 Fenwick batch prefix sum COMPLETADO** (VERAZ 2026-08-03: `src/asm/fenwick_avx2.s` **no existe** — nunca fue añadido a `cc::Build` ni a `rerun-if-changed`; no hay FFI de Fenwick en `asm/mod.rs`). El kernel real es **intrinsics inline** en `src/mud/fenwick.rs:prefix_sum_batch4_avx2` (4 lanes, AVX2 dispatch con fallback escalar). `test_fenwick_f32_prefix_sum_batch4_avx2` **verde en debug, FALLA en release** (UB: store por puntero sobre binding inmutable) y el kernel **no tiene llamadores** ⇒ retirado (R-03, ver `GEMINI §0.3`). *(D-1: AGENTS decía ".s + FFI" — era intrinsics; el .s fue purgado P-08 en 2026-08-03).* **Fase 3 ezop AVX2 COMPLETADO** (2026-07-28: `axpy_avx2`, `axpby_avx2`, `sum_sq_avx2`, `scale_avx2` en `ezop.rs` con `_mm256_fmadd_ps` para 8 elementos/iteración; 4 tests de equivalencia). **Fase D2 bf16 AVX2 COMPLETADO** (2026-07-28: `f32_to_bf16_batch_avx2` en `adam_state.rs` con `_mm_shuffle_epi8` para 8 elementos/iteración; integrado en `store_shadow_bf16`). **Trainer P-01 fix COMPLETADO** (2026-07-28: eliminado `x_data[..hidden_size].to_vec()` en `train_step.rs:342` → buffer pre-allocado `backward_ws.emb_ckpt` con raw pointer para evitar borrow conflict; ~100 KB/chunk saved). **KV delta AVX2 COMPLETADO** (2026-07-28: `vadd_avx2`/`vsub_avx2` en `ezop.rs` para delta reconstruction/computation en `kv_compressed.rs`; 2 tests de equivalencia; ~8× speedup sobre scalar en loops de delta). **VicReg AVX2 COMPLETADO** (2026-07-28: `compute_vicreg_variance_gradients_avx2` con intrinsics reales (`_mm256_fmadd_ps` para varianza, `_mm256_add_ps`/`_mm256_mul_ps` para gradientes); wrapper de dispatch con fallback scalar; 4 tests incluyendo equivalencia AVX2 vs scalar). **Self-play P-01 + AVX2 COMPLETADO** (2026-07-28: pre-allocación de `regs_f32` y `logits` fuera del loop de generación; GEMV de logits con `f32_gemv_rowwise` (AVX2 + PCorePool) en lugar de loop escalar; ~vocab_size×hidden/8 speedup en logits). **slime_forward AVX2 COMPLETADO** (2026-07-28: RMSNorm sum-of-squares con `sum_sq_avx2`; peak-finding con `_mm256_max_ps` para i8 quantization; ambos con fallback scalar). **slime_backward AVX2 COMPLETADO** (2026-07-28: branch gradient con `axpy_avx2` (2 sitios: FFN + Attention); peak-finding con `_mm256_max_ps`; vector addition con `vadd_avx2` para FFN norm out). **arena_judge AVX2 COMPLETADO** (2026-07-28: `cosine()` con `_mm256_fmadd_ps` para dot/na/nb en paralelo; 8 elementos/iter; fallback scalar). **worm AVX2 COMPLETADO** (2026-07-28: `axpby_avx2` (b=0) para scaling loops en `WormChain::forward`, `route_and_forward`, `WormholeNetwork::forward` (5 sitios); ~8× speedup sobre scalar en loops de scaling). **STP loss AVX2 COMPLETADO** (2026-07-28: `vsub_avx2` para d1/d2 computation; `_mm256_fmadd_ps` para g_d1/g_d2 gradient computation (8 elementos/iter); accumulación en grad_h_r/t/s con `_mm256_add_ps`; fallback scalar; `test_stp_grad_matches_finite_diff` verde). **Topographic Telemetry COMPLETADO** (2026-07-28: `grad_norm` (L2) agregado a `TrainChunkMetrics` + `[TELEM]` log; **topografía escalada a 70% de pantalla** (full-width); **mapa topográfico real** (no scatter) — curva de loss dividida en **segmentos contiguos coloreados** (`GraphType::Line` + `Marker::Block`), cada segmento con el color del banda de elevación: Deep Valley (🔵 fg=RGB(0,0,255) bg=RGB(0,0,64)) → Valley (🔵 fg=RGB(0,128,255) bg=RGB(0,0,128)) → Plateau (🟡 fg=RGB(255,255,0) bg=RGB(128,128,0)) → Hill (🔴 fg=RGB(255,128,0) bg=RGB(128,64,0)) → Peak (🟠 fg=RGB(255,0,0) bg=RGB(128,0,0)); **RGB del hill actual** calculado dinámicamente y mostrado en el título del chart + label de la bola; **dirección de procesamiento** (↓ descending / ↑ ascending / → flat) mostrada en el label de la corrección; layout 70/30: Topo (70%) + Loss + GradNorm (30%). **Training UI banner** — **NOT IMPLEMENTED**: `print_training_banner`/`TrainingBannerInfo` → **0 matches** en el codebase tras purga P-08 (2026-08-03). AGENTS decía "agregado en trainer_ui.rs" — era un placeholder que se eliminó. `circuit_event` (trainer_ui.rs) sigue vivo y es el punto de entrada de logs/circuit.log. **Guardado de pesos verificado**: `sync_shadow_to_mud` → save `.tmp` → atomic `rename` → hash verify (no-op detection). ✅ D1 4-pillar efficiency (4/4 DONE: JEPA resonance, Muon, sampled softmax, **self-play COMPLETADO** — `self_play_training_step` en `corpus_trainer/session.rs:1644`, `MUD_SELF_PLAY=1`, `generate_synthetic_sequence` + `apply_gradient_jitter` + `is_sequence_confident` + `SelfPlayCurriculum` + `should_continue` + `self_play_reward`, **22 tests** en `self_play.rs`); D2 bf16 shadows COMPLETADO — `store_shadow_bf16` en `optim.rs:434/453` (Adam + SparseAdam arms), `MUD_BF16_SHADOW=1`, 3 tests en `adam_state.rs` (`test_bf16_roundtrip`, `test_bf16_zero_and_sign`, `test_shadow_bf16_store_load`). **Vulkan/ash delegation (2026-07-26):** Análisis profundo de funciones del trainer para identificar cálculos que pueden delegarse a GPU sin activar el SIGSEGV de Iris Xe (específico del shader `qkv_pipe`, no de `submit_and_wait` general). STP loss + mHC SGD + JEPA dynamics → `dispatch_optimizer_batch_async` (safe, no `submit_and_wait`); NS/RMSNorm/MHA/GEMV-regular → probar shaders no-QKV. Ver `docs/research/MUD_VULKAN_DELEGATION_ANALYSIS_2026-07-26.md`. **D5 GPU signal keep-alive (2026-07-26):** Plan para mantener Iris Xe en boost clock mediante batching de backward pass (14M ops) + mHC (49K) + STP (2K) + JEPA (25K) en un solo `dispatch_optimizer_batch_async`. Ver `docs/research/MUD_GPU_SIGNAL_KEEPALIVE_PLAN_2026-07-26.md`. **Trainer perf fixes (2026-07-26):** ΔW tracking para embedding (fix hash≠ΔW contradiction), pre-allocación de 15 buffers en `SlimeBackwardWorkspace` (P-01), eliminación de `ternary_data` Vec intermedio (~1.2 GB/embedding sync), zero-copy scale comparison, `mhc_stress_bench` tool (stress test + precision + speed benchmarks). **Pendientes session 2026-07-26:** CellV5 Fase 2: sólo **`cmud_gauss_mul_batch_avx2`** (1 de 3 kernels) en `cmud_batch.s` + FFI `asm/mod.rs:181`; `wave_collapse` y `phase_repulsion` **no existen**. `cmud_cell_v5_avx2.s` **no existía** (no se compilaba — purgado P-08 2026-08-03); `slimex_asm.rs` era huérfano no-declarado en `mod.rs` (borrado P-08 2026-08-03). *(D-5 corregido: AGENTS decía "compilado build.rs:16" — era dead weight.)* Fase 3 KV CellV5 pre-compression **NOT IMPLEMENTED (deferred)**; código loguea "not yet implemented"; Vulkan D5 elementwise integration **✅ COMPLETADO** (BATCH E, 2026-07-27); DSPark/stochastic drafter — **(D-4b) VERDADERO ESTADO CORREGIDO (2026-08-03):** AGENTS decía "`DSparkConfig` + `DSparkDrafter` en `dspy.rs`, 13 tests, `MUD_DSPARK=1` ✅ COMPLETADO" — **FALSO**. El código real es `SlimeDrafter<'a>` en `src/mud/speculative.rs` (7 tests); `dspy.rs` contiene el trait `DeclarativeSignature` (DECL-01), no drafter stubs; no existe el env `MUD_DSPARK`. **Pendientes 2026-07-28:** Vulkan backward shader (4.1) **✅ IMPLEMENTADO** (`ternary_backward_dx.comp`, `gemv_pipe` family, safe on Iris Xe; validación HW en curso); D1/D2 scale training **✅ COMPLETADO** (D1 4/4 DONE: self-play con `SelfPlayCurriculum` + `should_continue` + `self_play_reward`, 22 tests; D2 bf16 shadow training con `store_shadow_bf16` en `optim.rs:434/453`, 3 tests); C-MUD log-gas (E1-E5) **📝 DOCUMENTADO** (DEFERRED to reliability gate, `CMUD_LOGGAS_FEASIBILITY.md` §4); 1M LSH **📝 DOCUMENTADO** (research item, `csa_indexer.rs` tiene LSH prefilter pero 1M-scale no implementado); miri **✅ COMPLETADO** (2026-07-28: 11/11 tests, 0 UB, dos bugs de alineación corregidos en `kv_compressed.rs`). 464 tests (debug) | clippy 0 ✅ | proptest 6/6 (histórico; ver `GEMINI §0.3`). **Sesión 2026-07-31 (ASM backward batch 8):** `backward_ternary_gemv_8rows.s` — 8-row batched grad_x accumulation (8× reduction en grad_x load/store vs single-row); `vpsub`→`vaddps` fix; `lea` row pointer precomputation (scale 1,2,4 + `add`); thread-local scratch para `jepa_ou` OPT-3. **433 tests (debug) | clippy 0 ✅** (histórico; ver `GEMINI §0.3`) (perdió 1 test por slimex_asm eliminado en cleanup previo). Ver `docs/sessions/MUD_SESSION_REPORT_2026-07-31_ASM_BACKWARD.md`.

> **Last sync:** 2026-07-27. **BATCH E (Vulkan D5 elementwise integration) COMPLETADO** — `step_elementwise_sync` (elementwise.comp) integrado en `jepa_stabilizer` (modos 4/5: normalize + ema) y `stp_loss_and_grad` (modo 6: axpy). Fallback CPU automático cuando `AshQatDispatcher` es `None` (Iris Xe blacklisteado por `probe_gpu()`). **D5 signal keep-alive fase 2 COMPLETADA** — `dispatch_combined_batch_async` ahora despacha optimizer + mHC en un solo `vkQueueSubmit` via `step_combined_async_deferred` en `train_step.rs`. Ver `docs/sessions/MUD_SESSION_REPORT_2026-07-27_BATCH_E_GPU_INTEGRATION.md`.
> 
> **Last sync:** 2026-07-27. **SEC-H BACKLOG COMPLETAMENTE CERRADO** — SEC-H2 (proptest fuzzing: 6 property tests + 1 fallback, `--features proptest`); SEC-H3 (3 SAFETY comments añadidos a `AshBuffer`/`SlimeLayer`/`AlignedArray`, 9 ya auditados); **SEC-H6 completo (17 eprintln! migrados a `log::warn!`/`log::info!`** en library code: GGUF + NaN guards + cmud + GEMV + ECC + RLVR + boot_indexer + kv_compressed + WORM; sólo 4 quedan intencionalmente CLI/TUI telemetry; `env_logger` init en `main.rs`); SEC-H9/H10/H11 verificados ya-implementados en código. **P-02 DISCREPANCIA CRÍTICA RESUELTA** — docs (GEMINI.md/AGENTS.md/README/ROADMAP + 4 architecture docs) alineadas con código real (f16-packed dual `(re,im)` u32 register, commit `9bc665d1f` 2026-07-22); `registers_to_f32` dead-code gating `#[cfg(test)]` + `#[deprecated]` (P-01/P-08). **464 tests (debug) | clippy 0 ✅ | proptest 6/6 (histórico; ver `GEMINI §0.3`).** Items explícitamente DIFERIDOS con rationale (D1/D2, C-MUD log-gas, miri) documentados en `docs/audits/MUD_SEC_H_RESOLUTION_2026-07-27.md` §8. *(Nota: CellV5 Fase 2/3 y DSPark wiring fueron posteriormente COMPLETADOS en sesión 2026-07-28 — ver §Pendientes abajo.)*
>
> **Last sync:** 2026-07-28. **CellV5 Fase 3 KV pre-compression** — **NOT IMPLEMENTED (deferred)**; el código loguea `MUD_KV_COMPRESS_CELLV5=1 is RECOGNIZED but ... not yet implemented` (kv_compressed.rs:68-72), fallback Delta+LZ4 sobre i8 (Fase-2, vivo). *(D-3 corr. 2026-08-03: AGENTS decía "COMPLETADO" con `KV_CELLV5_I8_BUF`, `cellv5_reserved`→`cellv5_enabled`, 11 tests — todos falsos. El thread_local real es `KV_DECOMPRESS_F32_BUF`; el field sigue siendo `cellv5_reserved`; hay 8 tests de kv_compressed, no 11. El delta-encoding per-i8 bug fix pertenece al Fase-2 (BUG-1 del 2026-08-03), no a Fase-3.)* **D1/D2 scale training** — D1 4-pillar efficiency (4/4 DONE: JEPA resonance, Muon, sampled softmax, **self-play COMPLETADO** — `self_play_training_step` en `corpus_trainer/session.rs:1644`, `MUD_SELF_PLAY=1`, `generate_synthetic_sequence` + `apply_gradient_jitter` + `is_sequence_confident` + `SelfPlayCurriculum` + `should_continue` + `self_play_reward`, **22 tests** en `self_play.rs`); D2 bf16 shadows COMPLETADO — `store_shadow_bf16` en `optim.rs:434/453` (Adam + SparseAdam arms), `MUD_BF16_SHADOW=1`, 3 tests en `adam_state.rs` (`test_bf16_roundtrip`, `test_bf16_zero_and_sign`, `test_shadow_bf16_store_load`). **C-MUD log-gas (E1-E5)** — DEFERRED to reliability gate (documentado en `docs/research/CMUD_LOGGAS_FEASIBILITY.md` §4: requiere 25-epoch stable train + circuit honors + mHC stability antes de experimentar). **1M LSH** — research item (DeepSpec DSpark, track DSP-5); `csa_indexer.rs` tiene LSH prefilter (`MUD_CSA_LSH=1`, 16 bits, Hamming radius 4) pero 1M-scale LSH no implementado. **miri verification** — **COMPLETADO** (2026-07-28): nightly + miri instalados; `cargo +nightly miri test --lib -- kv_compressed --test-threads=1` → 11 passed, 0 UB. Dos bugs de alineación corregidos: (1) LZ4 decompress path (`from_raw_parts` de `Vec<u8>` como `&[f32]` → `KV_LZ4_F32_BUF` thread-local `Vec<f32>` con `copy_nonoverlapping`); (2) CellV5 decompress path (`from_raw_parts` de `Vec<u8>` como `&[u32]` → `KV_CELLV5_BUF` thread-local `Vec<u32>` con `copy_nonoverlapping`). Audit estático con SAFETY comments completado (SEC-H3). 464 tests (debug) | clippy 0 ✅ | proptest 6/6 (histórico; ver `GEMINI §0.3`).

**Sesión 2026-07-29 (F32 Backward + Checkpoint Thread-Locals):** **F32 backward COMPLETADO** — `backward_slime_block` en `slime_backward.rs:524-705` ahora implementa backward completo para F32 weights (Q2_0→F32 converted models) usando `f32_gemv_backward` (dL/dW = grad_y ⊗ x^T, dL/dx = grad_y^T @ W) en lugar del no-op anterior (que solo copiaba grad_in→grad_out). Cubre todos los 8 pasos: FFN Down/Up/Gate Projection Backward, FFN ReLU² Activation Backward, FFN/Attn RMSNorm Backward, Attention O/Q/K/V Projection Backward, SDP Attention Backward (STE Proxy). `f32_gemv_backward` agregado en `slime_forward.rs:448-481` con 2 tests (`test_f32_gemv_backward_matches_reference`, `test_f32_gemv_backward_accumulates`). **Checkpoint thread-local lifetimes COMPLETADO** — 4 errores de lifetime en `checkpoint.rs` (170, 179, 507, 513) corregidos reestructurando `SYNC_PACKED_BUF`/`SYNC_SCALES_BUF` para usar `with_borrow_mut`/`with_borrow` con todo el trabajo dentro del closure (el patrón `LocalKey::with` no puede retornar un `RefMut`). También se eliminó `Vec::with_capacity` + `.to_vec()` en el path `scales_only` de `update_layer_tensor`. 413 tests ✅ | clippy 0 ✅. Ver `docs/sessions/MUD_SESSION_REPORT_2026-07-29_F32_BACKWARD.md`.

**Sesión 2026-07-29 (Cleanup + Refactorización):** **Código muerto eliminado (P-08)** — `src/mud/slimex_asm.rs` (3104 LOC, solo usado en tests internos) + `tools/slimex_asm_bench.rs` + referencia en `Cargo.toml` y `mod.rs` eliminados. **RMSNorm AVX2 optimizado** — `slime_rmsnorm_i8` ahora extrae accum values a buffer f32 y usa `sum_sq` con AVX2 (`_mm256_fmadd_ps`, 8 f32/iter) para sum-of-squares, eliminando el loop escalar de f64. **f32_gemv_backward AVX2 optimizado** — outer product (dL/dW) usa `_mm256_fmadd_ps` con broadcast de `grad_y` (8 elementos/iteración); GEMV (dL/dx) usa row-iteration cache-friendly con AVX2. **f32_gemv_backward_batch AVX2 optimizado** — outer product Q/K/V con helper closure `outer_avx2`; GEMV dL/dx con row-iteration AVX2. **FFN backward AVX2 optimizado** — peak-finding con `peak_abs_avx2` (`_mm256_max_ps`); ReLU² activation backward branchless con `relu2_backward_avx2` (`_mm256_and_ps` mask); i8→f32 reconstruction con `scale_i8_to_f32_avx2` (`vpmovsxbd` + `vcvtpi256ps`). **Fenwick tree optimizado** — `find_kth_avx2` y `find_kth_walk` usan `lzcnt` (BMI1) para bitmask inicial O(1) en lugar de O(log n) loop. **412 tests ✅ | clippy 0 ✅** (perdió 1 test por slimex_asm eliminado). Ver `docs/sessions/MUD_CLEANUP_PLAN_2026-07-29.md`.

**TODO para mañana (2026-07-30):** (1) Entrenar modelo LLM solo con ASM — forward/backward/optimizer/JEPA/mHC en ASM puro (`backward_ternary_gemv.s`, `backward_rmsnorm.s`, `backward_silu.s`, `backward_rope.s`, `jepa_ou.s`, `mhc_gate.s`); (2) Extracción de archivos grandes (`slime_forward.rs` → `gemv_gpu.rs` + `rope_cache.rs`; `cmud.rs` → `cmud_thinking.rs` + `cmud_params.rs`; `session.rs` → `self_play.rs` + `stp.rs` + `telemetry.rs`); (3) Optimización ASM `ternary_gemv_8rows.s` con `vpmovzxbd` + `vpermd`; (4) Código muerto adicional (`slime_x.rs`, `debate_trainer.rs`, `arena_judge.rs`). Ver `docs/sessions/MUD_TODO_2026-07-30.md`.

**Sesión 2026-07-31 (Telemetry + Corpus):** **Telemetry telemetry enrichment COMPLETADO** — `grad_norm`, `entropy`, `stp_loss` agregados a `TrainChunkMetrics` + `ActStatsAccum` (metrics.rs), computados en `train_on_sequence` (grad_norm from pre-clip L2 norm, entropy from softmax `−Σ p·log₂p`, stp_loss del acumulador existente), logueados en `[TELEM]` stderr line + legacy file format (cols 8/9/10, antes `0.0` placeholders). TUI `train_telemetry` actualizado: Efficiency mode ahora muestra **Entropy (bits)** chart (reemplaza never-populated `loss_cv`) + **STP Loss** chart (reemplaza `stp_lambda_eff` que never recibía datos reales); `grad_norm` Series already existed, now populated via `grad_norm=` key. Verificado end-to-end: `grad_norm≈0.0013` (stable), `entropy≈0.109` (no collapse), `stp_loss=0.0` (STP off). **Code corpus extraction COMPLETADO** — `tools/extract_code_docs.rs` (nueva herramienta) extrae doc comments (`///` `//!`) + explanatory `//` comments de `src/mud/**/*.rs` + markdown de `docs/architecture/` y `docs/sessions/` → produce `training/corpus/mud_docs.txt` (6165 lines, 80k words de documentación del sistema: Adam, mHC, JEPA, AVX2 kernels, VRAM, ternary quantization). Auto-incluido en `build_project_corpus` via glob `training/corpus/*.txt`. `./mud.sh extract-docs` + catalogado en `./mud.sh tools`. 423 tests ✅ | clippy 0 ✅.

> **Sesión 2026-07-31 (mHC sync FIX + Checkpoint speed):** **mHC sync BUG CRÍTICO arreglado (FIX #47)** — `sync_shadow_to_mud` (checkpoint.rs:135) sincronizaba weight tensors + PRQ scales pero NUNCA sincronizaba `blk.N.mhc_alpha` / `blk.N.mhc_beta`. Los valores mHC se entrenaban in-place via `mhc_scale_sgd_step` (escribiendo a `mhc_alpha_shadow`), pero nunca se persistían al `.mud` → el entrenamiento mHC era silent corruption (los valores se perdían en cada checkpoint save). **Fix:** (1) `sync_shadow_to_mud` ahora acepta `&[SlimeLayer]`; (2) `sync_mhc_shadows()` escribe `mhc_alpha_shadow`→`blk.N.mhc_alpha` / `mhc_beta_shadow`→`blk.N.mhc_beta` via `sync_f32_shadow_tensor`; (3) `hash_trained_weights` extendido para hashear tensors mmap-backed (via `data_ptr`) además de `owned_data` — previamente solo hasheaba `owned_data`, por lo que mHC (mmap-backed) era invisible al no-op check. **Call sites actualizados:** `mod.rs` shim + `save_checkpoint` + `session.rs` (2 checkpoints + final sync) + `debate.rs`. **Speed optimizations en checkpoint sync:** (1) Packing loop reemplazado de scalar per-element read-modify-write (1 RMW por elemento) → `pack_ternary_into` batch (1 store por 8 elementos, cero RMW) en ambos paths: embedding sync (checkpoint.rs:190-213) y `update_layer_tensor` (checkpoint.rs:611-656); (2) Eliminado double-clone de `SYNC_PACKED_BUF` (~48MB) usando `with_borrow_mut(std::mem::take)` — move en vez de clone; (3) Eliminado double-clone de `SYNC_SCALES_BUF` de igual forma; (4) Path `scales_only` también optimizado con `mem::take`. Equivalencia ternaria verificada: `round(w·s⁻¹) > 0.5` ⟺ `|w| > 0.5·s` (delta en `pack_ternary_into`). 423 tests ✅ | clippy 0 ✅. Ver `docs/sessions/MUD_SESSION_REPORT_2026-07-31_MHC_SYNC_FIX.md`.

**TODO para mañana (2026-07-31):** (1) **Wiring ShadowExpertBus en backward** — `mount()`/`evict()`/`unmount()` pre-allocated in `SlimeBackwardWorkspace` (slime_backward.rs:170) pero NUNCA llamados desde training path; wire `mount()` en `backward_slime_block` FFN section (reemplaza inline dequant por slot buffer), `unmount()` post-backward, verify con `slimex-audit` (`mounted_count > 0`); (2) **OPT-1** JEPA `y_norm_tmp` Vec → thread-local buffer (slime_jepa.rs:254 — P-01 violation); (3) **OPT-2** Vectoriza adaptive alpha correction en JEPA (slime_jepa.rs:295) con `fmsub_avx2` + `axpy_avx2`; (4) **OPT-3** Vectoriza mHC alpha/beta scalar backward loop (slime_backward.rs:750-757) con `_mm256_mul_ps`/`_mm256_fmadd_ps`; (5) **OPT-4** Batch grad_w AXPY across 8 rows en `ternary_gemv_backward` (actualmente per-row incluso en 8-row path); (6) **Código muerto** (`slime_x.rs`, `debate_trainer.rs`, `arena_judge.rs` — P-08). Ver `docs/sessions/MUD_TODO_2026-07-31.md`.

---

**Last sync:** 2026-08-04. **GRASSHOPPER FIX (SGDR cosine restarts) COMPLETADO.** La causa raiz del "LR tiende a caer, loss se mantiene" era que el schedule cosenoidal single-cosine causaba una coincidencia de LR en step 2 y step 1059 (ambos 1.07e-4, 21% de base). La LR caia monoticamente desde su pico hasta el minimo, dejando el modelo en zona de LR insuficiente por el 73% del entrenamiento. Fix: SGDR en apply_lr_schedule() con ciclos que hacen saltar la LR. MUD_LR_RESTART=0 para deshabilitar, MUD_LR_CYCLES=N para setear conteo. Step 1059: LR 1.07e-4 a 4.81e-4 (4.5x mayor). 6 tests nuevos. Ver docs/sessions/MUD_SESSION_REPORT_2026-08-03.md. 460 tests (debug) | clippy 0 — ⚠️ **release falla 1/461** (ver `GEMINI §0.3`).

---

## Project Identity

| Field | Value |
|-------|--------|
| **Name** | Forge LLM (MUD — Modular Understanding Dynamics) |
| **Language** | Rust only in production (P-07) |
| **Goal** | 1.58-bit ternary LLM inference + STE QAT on commodity Intel CPU (+ optional Iris Xe Vulkan) |
| **Primary target HW** | Intel i7-1260P (P-cores + HT, Iris Xe) |
| **Format** | `.mud` (`MUD\x01`), ELUT 4-bit + PRQ (ternary) **or** Float32 (dequantized Q2_0) |
| **Example model class** | BitNet-class / SmolLM-class dims **inferred from metadata** (e.g. layers/hidden/ffn/heads — never hardcode) |

---

## Core Constraints (subset — full index in GEMINI §9)

| ID | Rule |
|----|------|
| P-00 | Raw pointer mastery in hot path |
| P-01 | Zero allocation in hot loops — `SlimeWorkspace` / pre-alloc trainer buffers |
| P-02 | `SlimeRegister` is a **f16-packed dual (re, im) u32**; `read_accum()` returns f32 via F16C (current truth) |
| P-03 | ELUT 4-bit nibble wire format (ternary path); F32 weight format for Q2_0→F32 converted models |
| P-05 | JEPA + mHC every block |
| P-06 | `cargo clippy --all-targets` 0 errors / 0 warnings |
| P-07 | No Python / No PyTorch |
| P-08 | Delete dead code (do not archive) |
| P-13 | No hardcoded model dimensions — metadata or fail |
| P-17 | Fail-fast / skip+log nulls — no silent corruption |
| P-27 | No Rayon in `forge_llm` runtime hot path (`PCorePool` only) |

---

## Runtime Truth (read before coding)

1. **Optimizer (L-01+L-02 + stream A):** shape-dispatch; Muon NS CPU or ash if `MUD_USE_VULKAN=1`; **Adam/Sparse moments LIVE**.
2. **Registers:** `SlimeRegister` is a **f16-packed dual (re, im) u32**; `read_accum()` returns f32 via F16C (current truth). No i16 / legacy `iscale` on native registers. GEMV computes in f32 scratch (`gemv_accum`) then commits to f16 storage via `write_accum()`.
3. **Weights:** ELUT 4-bit; unpack via `unpack_ternary2bit_to_f32` (+ PRQ). Never cast packed bytes as `f32`. **F32 weight format** (Q2_0→F32 converted models) uses dense f32 dot-product via `f32_gemv_rowwise` — `is_f32_weights` flag on `SlimeLayer` dispatches between ternary and F32 GEMV. **F32 backward** uses `f32_gemv_backward` (dL/dW = grad_y ⊗ x^T, dL/dx = grad_y^T @ W) in `backward_slime_block` — full backward support for Q2_0→F32 converted models (no longer a no-op).
4. **Forward GEMV:** AVX2×8 (PCorePool 8 hilos, Rayon OFF). **GPU dispatch = `MUD_GPU_GEMV=auto`** ⚠️ (**`mud.sh:48` fuerza `0`**; el default `auto` sólo aplica si la env está sin definir — corregido 2026-10-11, R-48): one-shot micro-bench CPU vs GPU(ash) → usa GPU sólo pasado el break-even. **En i7-1260P + Iris Xe (ADL GT2) el hot-path real elige AVX2** (dispatch por-GEMV en UMA no amortiza para 147M; bench sintético sí gana 2.5–14×). Force `0`=CPU / `1`=GPU. Backend Vulkan = **ash 0.38** (`src/vulkan/ash_backend.rs`; probe real vía `probe_gpu()`); usado además en RMSNorm output-norm / Muon NS / heartbeat. F32 GEMV path (`f32_gemv_rowwise`) uses `dot_product_avx2` per row, parallelized over PCorePool.
5. **Inference LM head:** `lm_head_logits_avx2` in `main.rs` (prealloc buffers).
6. **JEPA:** `jepa_z` OU; gate `v_jepa`; no spring/repulsion; mHC radius ~ `√hidden`.
7. **Attention context:** L-13 HCA + dense ring; **CSA top-k** on large HCA (`MUD_CSA*`, inference only).
8. **Train:** full-seq windows default (`MUD_TRAIN_FULL_SEQ` / `SEQ_LEN`); pairs@pos0 if `FULL_SEQ=0`.
9. **MoE:** `moe_load` buses; `MUD_TRAIN_EXPERT=N` for dense expert FFN (w3=up, w1=gate, w2=down).
10. **Workspace:** `SlimeWorkspace` only. **Stack doc:** `docs/architecture/MUD_COMPUTE_STACK.md`. **Register doc:** `docs/architecture/MUD_REGISTER_BIT_PACKING.md`.

---

## Key Architecture

- **SlimeWorkspace:** Pre-allocated registers, KV cache, norm/QKV/FFN buffers, `jepa_z[2 × num_layers × hidden]`, etc.
- **Forward:** RMSNorm → QKV GEMV → SDPA + GQA → O → Residual+JEPA/mHC → FFN RMSNorm → Up/Gate GEMV → SiLU → Down → Residual+JEPA/mHC → (output norm) → LM head
- **Tensors (typical names):** `blk.N.attn_q.weight` (Ternary2Bit), `blk.N.attn_q.prq_scale`, `blk.N.attn_norm.weight`, `blk.N.expert.0.w{1,2,3}.weight`, `token_embd.weight`, `output.weight` / tied. **F32 converted models** use `Float32` type for all weights (no `.prq_scale` tensors).
- **Kernels (active path):** `ternary_gemv` / `ternary_gemv_4rows` FP32 AVX2; `f32_gemv_rowwise` for F32 weights; `f32_gemv_backward` for F32 weight gradient computation; Vulkan GEMV optional by size/env
- **Vulkan:** `src/vulkan/ash_backend.rs` — zero-copy UMA intent; training may run CPU-only depending on flags/path
- **⚠️ Vulkan SIGSEGV on Intel Iris Xe (2026-07-18):** el driver Intel (`libvulkan_intel.so`, ADL GT2 UMA) hace **SIGSEGV en `submit_and_wait`** al dispatchar QKV GEMV (backtrace `dispatch_gemv_qkv_host_sync` → `evaluate_slime_block_moe` → `train_on_sequence`). Determinista en block 11/64. Fix: `mud.sh train` fuerza `MUD_USE_VULKAN=0`/`MUD_GPU_GEMV=0` (override `MUD_TRAIN_FORCE_VULKAN=1`) **y `probe_gpu()` ahora blacklistea Iris Xe in-binary** (`src/vulkan/ash_backend.rs`) → inferencia directa también degrada a AVX2. Un fallo de driver NO es capturable como `Result`. Detalle: `docs/sessions/MUD_SESSION_REPORT_2026-07-18.md` §2/§7.2.

### JEPA dynamics (current)

```
y_norm = (y - μ_y) / σ_y
z_next = 0.9·z + 0.1·y_norm   # pure OU / EMA
y_final = y                   # pass-through; mHC bounds
v_jepa = (z - μ_z) · inv_σ_z
gate = sigmoid(v_jepa)        # mHC blend modulation
```

---

## Ledger (closed) + Next Work

| Priority | ID | Task | Entry points |
|----------|-----|------|----------------|
| ✓ | **L-01** | ~~Wire strategy into step~~ | DONE |
| ✓ | **L-02** | ~~Vulkan Newton-Schulz~~ | DONE |
| ✓ | **L-03** | ~~Delete InferenceWorkspace~~ | DONE |
| ✓ | **L-07** | ~~P-13 cleanup~~ | DONE |
| ✓ | **L-04** | ~~ASM orphans purge~~ | DONE |
| ✓ | **L-08** | ~~NaN guards ASM~~ | DONE (non-finite → 0) |
| ✓ | **L-05** | ~~True double-buffer QAT~~ | DONE (DoubleFrame + deferred readback) |
| ✓ | **L-06** | ~~mha/rms_norm dispatch~~ | DONE (output_norm GPU; dense MHA helper) |
| ✓ | **L-09** | ~~EZOP raw-pointer pass~~ | DONE (`mud::ezop` + TLS scratch + zero-alloc backward) |
| ✓ | **L-10** | ~~Sequence packing (no pad)~~ | DONE (`sequence_pack`; full-chunk pairs) |
| ✓ | **L-11** | ~~Mini MoE ExpertBus~~ | DONE (`slime_expert`+`expert_bus`; FFN hook) |
| ✓ | **Phase B** | ~~GEMV shared-tile + QKV parallel~~ | DONE (shader + CPU barrier) |
| ✓ | **L-12** | ~~P-13 props + CI battery~~ | DONE (`p13`, `mud.sh ci`, GHA) |
| ✓ | **Phase B+** | ~~GPU GEMV opt-in in forward~~ | DONE (`MUD_GPU_GEMV=1`) |
| ✓ | **L-13** | ~~CSA/HCA 32k-ready KV~~ | DONE (dense ring + HCA policy) |
| ✓ | **L-14** | ~~C-MUD complex manifold~~ | DONE (`cmud`; `MUD_CMUD_THINK=1`) |
| ✓ | **L-15** | ~~Gradient checkpointing~~ | DONE (`MUD_GRAD_CKPT=1`) |
| ✓ | **F1** | ~~Trainable mHC α/β~~ | DONE (`mhc_scale_sgd_step`; CPU+ash paths; clamp [0,4]; finite-diff test) |
| ✓ | **F2** | ~~STP trajectory loss~~ | DONE (`stp_loss`; `MUD_TRAIN_STP=1` default OFF; `λ`=`MUD_TRAIN_STP_LAMBDA`) |
| ✓ | **F3** | ~~RLVR debate: juez + reward/penalty + aprendizaje~~ | DONE (`arena_judge` Verifiable/Rust/Text/Professor no-API; `run_game` infinito hasta basta; `MUD_DEBATE_LEARN` default OFF) |
| ✓ | **F3+** | ~~Seed-driven Training Circuit~~ | DONE (`corpus_trainer::run_training_circuit`; baterías por semilla vía LCG; align/debate/games/professor; time-box por fase; telemetría unificada + `logs/circuit.log`; guarda al `quit`). **Honors eval (2026-07-18):** `circuit_eval_integrity` (gate estructural: tensor presente/non-null tras `materialize_writable`; NO flag de nibble-skew) **+ gate de colapso de logits** (2026-07-20: `model_logits_collapsed` rechaza base con dominancia de token-0) + `circuit_benchmark_games` win-rate vs baseline; rollback a `.bak_circuit` si falla. CPU caveat: benchmark puede dar 0 matches → solo integridad guarda. **⚠️ REQUISITO:** el `.mud` de partida debe estar SANO (`attn_norm.weight` ≠ 0, `scale_audit` dentro de tolerancia, y no colapsado en logits). **ESTADO 2026-07-20 (resuelto):** el base colapsado fue reconvertido sano (`models/smollm2.mud`; colapsado en `models/smollm2.collapsed.bak`). `models/ternary_bonsai_1.7b.mud` es alternativa sana. **Retrain de verificación (2026-07-20):** CONFIRMADO que el trainer persiste pesos (checkpoint hash `0x872c…→0x24e7…`, MD5 base cambia) — `conf>20%` NO es requisito (2 chunks no alcanzan; el gate real es movimiento de pesos). Ver `MUD_SESSION_REPORT_2026-07-20.md` §6/§7/§8. **🔧 FIX 2026-08-03 (integrity gate pre-training + universal layout resolution):** `run_alignment_session` corre `circuit_eval_integrity` AL INICIO (antes de `materialize_for_ste_train`); si falla, aborta antes del chunk 0 con mensaje claro + recomendación `./mud.sh forge`. `MUD_SKIP_INTEGRITY=1` override (debug). **Universalidad de layout (2026-08-03):** el gate (`circuit_eval_integrity`), el CLI `diagnose_model`/`mud.sh health`, el forward/inference y el checkpoint **todos reusan el mismo resolver** `moe_load::dense_ffn_names_for_train` — el modelo detecta su propio naming (Dense SmolLM/forge `ffn_up/gate/down` vs MoE DeepSeek `expert.E.w{1,2,3}`) en lugar de hardcodearlo. Así health ⟷ gate ⟷ training están en lockstep: si el forward puede wiredar una capa, el gate la considera completa. `models/pico_sano.mud` 🟢 0 missing. **Tip 2026-08-03:** `universal_converter` embebe los mismos metadatos canónicos (`mud:layout`/`mud:ffn_naming`/`mud:weight_type`/`mud:quant`) detectados universalmente del naming emitido (Dense `ffn_up/gate/down` vs MoE `expert.E.w*` vs dense-Llama `gate/up/down`; F32 vs Ternary2Bit). Caveza: `cmud_audit_runs_and_reports` (test) early-returns si `models/smollm2.mud` no existe; el forward C-MUD thinking (`MUD_CMUD_THINK=1`) sobre modelos grandes puede `munmap_chunk` (race env-setter preexistente, NO del gate — este usa `forward_last_logits`, safe sobre grandes modelos).
| ✓ | **TLM** | ~~Live training telemetry TUI~~ | DONE (`[TELEM]`→log + `kv_f64` key parser + ΔW panel + `grad_norm`/`entropy`/`stp_loss` panels en Efficiency mode; pointer hot-loops P-00/P-01; debate writeback hash check) |
| ✓ | **SX-01** | ~~SlimeX CellV5 u32 (8:8:8:8)~~ | DONE (`SlimeXCellV5` con pack/unpack bit-exacto; tests roundtrip + equi_byte_partition). Doc: `docs/research/MUD_ULTRA_COMPRESSED_REGISTER_AND_MEMORY_CONTINUITY_MASTER.md` |
| ✓ | **SX-02** | ~~ASM AVX2 unpacker para CellV5~~ | DONE (`src/asm/slimex_cell_v5_avx2.s` + `src/mud/slimex_asm.rs`; `test_asm_unpack_cells_v5_matches_scalar` verde; 1 ciclo `vpmovzxbd`). Doc: `docs/architecture/SLIMEX_PHASE1_ASM_POINTERS_LOG.md` |
| ✓ | **SX-03** | ~~Workspace `slimex_cells` + `snapshot_to_slimex()`~~ | DONE (`AlignedArray<u32>` 64-byte aligned; pre-alloc; zero-size cuando `MUD_SLIMEX ≠ 1`; raw-ptr hot path). `src/mud/slime.rs:236-413` |
| ✓ | **SX-04** | ~~Memory Continuity Table (256 slots)~~ | DONE (`src/mud/memory_continuity.rs`; `MemoryPointerEntry` 64-byte aligned; 16 KiB total → 100% L1d; O(1) lookup 0.82 ns) |
| ✓ | **SX-05** | ~~AOT Boot Indexer~~ | DONE (`src/mud/boot_indexer.rs`; scan `models/*.mud` en t₀ + seat en Continuity Table; 0 ms cold-start) |
| ✓ | **SX-06** | ~~MudFileMerger (file fusion)~~ | DONE (`src/mud/file_fusion.rs`; merge N .mud → 1 binary atómico) |
| ✓ | **SX-07** | ~~ShadowExpertBus + `slimex-train` comando~~ | DONE (`src/mud/slime_x.rs:ShadowExpertBus` con mount/unmount/eviction; `mud.sh slimex-train` con `MUD_SLIMEX=1`) |
| ✓ | **ST-01** | ~~SmartTokenizer (mode-aware + zero-alloc streaming)~~ | DONE (`src/mud/smart_tokenizer.rs`; `TokenBatchStream` con buffers reusados; `encode_raw()` → `(*const u32, len)` para zero-copy con GEMV) |
| ✓ | **CMUD-01** | ~~C-MUD Thinker (Complex manifold + gates)~~ | DONE (`src/mud/cmud.rs` 1478 LOC; `ThinkingState`, `CmudLayerParams` con sidecar JSON; `maybe_think_collapse_report` gated por `MUD_CMUD_THINK=1`; Hermitian ball + phase-lock + CUE repulsion). Doc: `docs/research/CMUD_THINKER_INTEGRATION_MASTER_EQUATION.md` |

**Ledger L-01…L-15 + F1/F2/F3 + F3+ + TLM + streams A–E + P0/P1/P2/P3/P4/P5 + Q2_0→F32 fix + SlimeX (SX-01…SX-07) + SmartTokenizer (ST-01) + CMUD-01 = CERRADOS como histórico.**
**⚠️ 2026-10-11: el backlog ACTIVO es `R-##`** → [`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) (7 olas). Varias filas "cerradas" de arriba quedaron retiradas por la auditoría (ver `GEMINI §0.3`): Fenwick batch prefix sum, "0 OOB", wormholes −50 %, `MUD_GPU_GEMV` default, conteo de tests.  
**F1/F2 verification:** `docs/architecture/MUD_TRAINER_TERNARY_JEPA_MHC.md` §9; plan `docs/research/MUD_PLAN_MHC_STP_TRAINABLE.md`.  
**F3 verification:** `docs/research/MUD_PLAN_RLVR_DEBATE_SEED.md` (juez no-API, run_game infinito, MUD_DEBATE_LEARN default OFF).  
**F3+ circuit:** `docs/research/MUD_PLAN_RLVR_DEBATE_SEED.md` §10 + notas 2026-07-18 (`./mud.sh circuit`, baterías por semilla; honores = integridad estructural + win-rate benchmark vs baseline).  
**Orbit:** `docs/research/MUD_IMPROVEMENTS_POST_AE.md` (F · K · G · H…).  
**MUD-WORM:** `src/mud/worm.rs` (WormPropagator + JEPARouter + MhCoupler + WormChain, 18 tests). Auto-conexión de modelos `.mud` en `models/` como modelo único. ⚠️ **Los "wormholes" NO son red ni reducen cómputo** (mismo trabajo + multiplicador local `strength`; ver `GEMINI §0.3`, R-67). Ver `docs/research/MUD_WORM_IMPLEMENTATION.md`.  

**🔧 FIX APLICADO (2026-07-20):** el checkpoint de 25 épocas era MD5-idéntico al base y el base `models/smollm2.mud` tenía **PRQ-scale inflada** en las últimas capas (~14–28×) — el "vocabulary collapse" era de escalas, **NO** de prefix-espacio `Ġ` (el vocab siempre tuvo `Ġ`; `strings` no renderiza UTF-8 y dio falsa negativa). **Resuelto:**
- **P1.1/P1.2:** `models/smollm2.mud` **reconvertido sano** (ratio dequant/BF16 ≈ 0.37 uniforme); el colapsado quedó en `models/smollm2.collapsed.bak`. Nuevo tool `scale_audit` / `./mud.sh scale-audit <model.mud> [source_dir]`.
- **P0.1/P2:** `MUD_TRAIN_DEBUG_DW=1` reporta ΔW ternario + ΔPRQ scale por checkpoint en `sync_shadow_to_mud`. Confirmó que el trainer SÍ mueve pesos cuando están lejos del óptimo (STE deadzone en modelos ya convergidos → ΔW≈0 es esperado, no bug).
- **P0.3/P1:** `MUD_TRAIN_RESET_EPOCH=1` (reinicia contador de época para evitar el resume no-op), warn en `epochs=0`, y clamp de shadow `MUD_TRAIN_WCLAMP_K` (default 8, 0=off) que previene la inflación de escala.
- Verificado: clippy 0 warnings, 233 tests (nuevos: `forward_sanity`, `model_logits_not_collapsed`, `test_smollm2_space_char`), `./mud.sh health` 🟢 CERTIFIED sobre el base fresco.
- **Retrain de verificación (2026-07-20):** CONFIRMADO que el trainer persiste pesos (checkpoint hash `0x872c…→0x24e7…`, MD5 base cambia). `conf>20%` NO es requisito (gate real = movimiento de pesos; `MUD_QAT_LR=0.03` lo fuerza). **P3 gates automáticos:** `circuit_eval_integrity` ahora rechaza base con dominancia de token-0 (`model_logits_collapsed`). **Banner LR:** `corpus_trainer.rs:1719` usa `qat_learning_rate()` (era `3e-4` hardcoded). **Circuit (F3+) ya corregido:** `run_debate_session` reusa `sync_shadow_to_mud`. Ver `MUD_SESSION_REPORT_2026-07-20.md` §5/§6/§7/§8.
**Countdown log:** `docs/manuals/LAUNCH_COUNTDOWN.md`.

**🔧 FIX TELEMETRY/TUI + POINTER OPT (2026-07-20):** el TUI `train_telemetry` mostraba paneles vacíos aunque el trainer corría. **Causa raíz:** el trainer emitía `[TELEM]` sólo a **stderr** (`eprintln!`), no al archivo `mud_train_metrics.log` que lee el TUI. **Resuelto:**
- `corpus_trainer.rs`: `[TELEM]` ahora se escribe a **stderr Y** `mud_train_metrics.log` (via `telemetry_file`); `[DW]` (bytes movidos / sync) emitido en cada `sync_shadow_to_mud`.
- `train_telemetry.rs`: parser por **clave** (`kv_f64`) en vez de posición; nuevo panel **Weight Δ (bytes moved / sync)**; escalas corregidas (VarJ sin piso `1.0`, JEPA-integral ±0.1); layout 3 columnas. Verificado con `tmux capture-pane`: paneles poblados (Loss 5.00→2.86, VarJ, JEPA, VarH, Cognitive, ΔW).
- **Hot loops P-00/P-01:** `apply_optimizer_cpu_step_and_pack` (clamp raw-ptr), `dequantize_ternary_row` (LUT `TERNARY_LUT` branchless, `pub(crate)`), `unpack_ternary2bit_to_f32` (raw ptr + LUT, 8/u32), `pack_ternary_into` (8/u32 word), `pack_elut_prq` (2/byte) — zero-alloc. clippy 0, 233 tests ok.
- **Debate writeback hash check:** `run_debate_session` compara `hash_trained_weights` in/out y muestra ✓/⚠ NO-OP (resuelve warning sin usar). Verificado `WCLAMP_K=0`+64 steps → hash `0x939f…→0x8d52…` persiste.
- **Hallazgo STE deadzone:** en el base sano convergido con LR por defecto `0.0005` (umbral `s*0.7`) ΔW≈0 es esperado (no bug). Para movimiento visible: `MUD_QAT_LR=0.03`.

**🔧 FIX CIRCUIT TUI & RPG (2026-07-21):**
- **TUI Scrolling Glitch:** Macros `println!` y `eprintln!` en `corpus_trainer.rs` y `mod.rs` envueltos en `if std::env::var("MUD_CIRCUIT_TUI").is_err()` para prevenir que contaminen la pantalla ncurses/crossterm cuando el TUI corre.
- **RPG Lineage & Titles:** `CircuitRpgStats` extendido con nombres de guerreros generados por generación. El modelo campeón actual defiende el título; si es superado, su nombre pasa a ser el del Baseline.
- **Arena Battle HP:** Se implementó una barra `battle_hp` paralela e independiente en la UI para la Arena, que reacciona descontando HP dinámicamente frente a las penalizaciones del `JUEZ`, resolviendo la queja "ninguno se hace daño".
- **Fase de Currículum:** `build_battery` ahora usa un orden estático de aprendizaje: `align -> professor -> debate -> games`.
- **C-MUD Token Bounds & Tokenizer Auto-load (`cmud_train`):** `inference.rs` previene panics fuera de rango acotando `tid.min(vocab_size - 1)`. `cmud_train.rs` auto-tokeniza archivos de texto plano (`project_corpus.txt`) con `Tokenizer::from_mud_metadata`. `mud.sh` restaura `MUD_CMUD_THINK=0` por defecto para evitar distorsiones en chats no entrenados para la variedad compleja.

**🔧 SPRINT 4 SPLIT CULMINADO (2026-07-24):** el split de `corpus_trainer.rs` (4534 LOC, item 4.1) que se congeló inicialmente, se re-ejecutó con éxito tras el pivot a Fenwick trees. **Estructura final** (`src/mud/corpus_trainer/`):
  - `mod.rs` (380 LOC) — entry point: macros, struct, shims, re-exports, tests
  - `metrics.rs` (120 LOC) — TrainChunkMetrics + ActStatsAccum (+ grad_norm, entropy, stp_loss)
  - `emb.rs` (131 LOC) — load_emb_row_into + slice_stddev
  - `integrity.rs` (270 LOC) — model_norms_alive + circuit_eval_integrity + benchmark
  - `alignment.rs` (507 LOC) — validate_metadata + deep_local_alignment + audit_tokenization
  - `debate.rs` (516 LOC) — run_debate_session
  - `circuit.rs` (318 LOC) — run_training_circuit
  - `session.rs` (1556 LOC) — run_alignment_session (training loop principal)
  - `train_step.rs` (1065 LOC) — train_on_sequence (forward+backward+optimizer)
- **APIs externas intactas** (`forge_llm::mud::corpus_trainer::*`): `MudCorpusTrainer`, `SHOULD_TERMINATE`, `CHUNK_SIZE`, `run_alignment_session`, `run_debate_session`, `run_training_circuit`. **344 tests pasando** (post race-condition fix), clippy 0 errors, build verde. Ver `MUD_SESSION_REPORT_2026-07-24.md`.
- **🔧 SPRINT 4 PIVOT (2026-07-24):** Fenwick trees (`docs/research/MUD_FENWICK_TREE_OPTIMIZATION_PLAN.md`) — F1 nucleus sample + F4 histogram threshold. **~198× speedup** para n=131 072 weights; **3.9×** para F1 nucleus sample.
- **F2/F3/F5 diferidos** (no son hotspots actuales: bins ≤ 16, HCA ≤ 1024, certify_trajectory ya O(n) acceptable). Documentados como backlog en el plan.
- **API objetivo:** `src/mud/fenwick.rs` con `FenwickF32` / `FenwickU32` / `FenwickI32`, `pub(crate)`, zero-alloc `with_capacity` + `reset_zeroed`, env `MUD_FENWICK=1` (default on) con fallback lineal si 0.
- **Sprint 4b (opcional, post-validación):** ASM AVX2 `prefix_sum_batch4` (4 queries paralelas) + `find_kth_walk` (`_pdep_u32`+`tzcnt`) para F1 batched.
- **Patrón establecido** (para futuros splits): cada sub-módulo = un `impl MudCorpusTrainer` extension, mod.rs thin, free functions en sub-módulos con `pub(crate)`.

---

**🔧 TELEMETRY ENRICHMENT + CORPUS EXTRACTION (2026-07-31):**
- **grad_norm telemetry:** `norm_sq.sqrt()` se computaba en `train_step.rs` para gradient clipping pero no se almacenaba ni logueaba. Agregado `grad_norm: f32` a `TrainChunkMetrics` + `ActStatsAccum` (metrics.rs); acumulado via `act_accum.push_grad()`; logueado en `[TELEM]` line (`grad_norm={:.4}`) + legacy file format (col 8, antes `0.0` placeholder). TUI Efficiency mode: `grad_norm` Series already existed, ahora populada via `grad_norm=` key.
- **entropy telemetry:** Shannon entropy (`-Σ p·log₂p`) de la distribución softmax, nunca computada. Agregado `entropy: f32` a metrics; computado post-softmax en `train_on_sequence`; logueado en `[TELEM]` (`entropy={:.3}`) + file format (col 10). TUI: nueva `entropy` Series + chart en Efficiency mode (reemplaza never-populated `loss_cv`).
- **stp_loss telemetry:** `stp_loss_accum` se acumulaba en `train_step.rs` pero no se exponía. Agregado `stp_loss: f32` a `TrainChunkMetrics`; populado desde `stp_loss_accum / stp_triples`; logueado en `[TELEM]` (`stp_loss={:.5}`) + file format (col 9, antes `0.0`). TUI: `stp_loss` Series already existed, ahora populada.
- **Corpus extraction tool (`tools/extract_code_docs.rs`):** nueva herramienta que extrae doc comments (`///`, `//!`) + explanatory `//` comments de `src/mud/**/*.rs` + markdown de `docs/architecture/` y `docs/sessions/` → `training/corpus/mud_docs.txt` (6165 lines, 80k words). Strips code syntax, boilerplate noise (TODO/FIXME), normaliza whitespace, agrupa en párrafos coherentes. Auto-incluido en `build_project_corpus` via glob `training/corpus/*.txt`. `./mud.sh extract-docs` + listado en `./mud.sh tools`.

---

## Training Pipeline

- Trainer entry: `tools/run_trainer.rs`, TUI `tools/warp_aligner.rs`, core `src/mud/corpus_trainer.rs`
- Trainer console UI (unified box/tag formatting, P-06-clean): `src/mud/trainer_ui.rs`
- Telemetry: `tools/train_telemetry.rs` (live TUI reading `[TELEM]` lines from `mud_train_metrics.log`; series: loss, loss_vel, lr, varh, varj, e_jepa, cog, sigma, grad_norm, entropy, stp_loss, conf, toks, prog, n_act, cmud_*, dw_moved)
- Corpus extraction: `tools/extract_code_docs.rs` (extracts doc comments + markdown docs → `training/corpus/mud_docs.txt`; auto-included in `build_project_corpus`)
- RLVR debate judge (no-API, local cosine): `src/mud/arena_judge.rs`
- Create model from scratch: `tools/mud_forge.rs` (dense ternary transformer, no SSM/Jamba)
  ```bash
  ./mud.sh forge nano models/scratch.mud --vocab-size 32000
  cargo run --release --bin run_trainer -- models/scratch.mud --epochs 5
  ```
- Convert: `tools/bonsai_gguf_to_mud.rs` or `tools/universal_converter` + `audit-conv` / `--check`
- **Converter critical fix (2026-07-23):** The Q2_0→ELUT+PRQ path computed a single PRQ scale per row by averaging all 16 block scales (128 elements each). This destroyed weight semantics — block-level scale variance was lost, producing near-uniform logits. **All Q2_0 tensors now go through `Q2ToF32`**, dequantized block-by-block using each block's native f16 scale. Models are ~3× larger but produce coherent output. See `docs/audits/BUG_LOGITS_CORRUPTION.md` and `docs/sessions/MUD_SESSION_REPORT_2026-07-23.md`.
- **Converter known bug (FIX D, 2026-07-18):** `MudFile::save` rewrote mmap-backed tensors using `tensor.offset` as an *absolute* mmap offset, but `offset` is relative to the data region → emitted `~±1e38` garbage for all norms/scales. Fixed by adding `MudTensor::data_base` and reading `data_base + offset`. **Relapse corregido mismo día:** el read de paridad ECC en `mod.rs:770` también usaba `offset` absoluto → verificaba contra la región de metadata (corrupción silenciosa). Ahora usa `data_base + offset`. The converter `--check` only validates metadata, not tensor health; always `diagnose_model` the output.
- Corpus: `training/corpus/` (AOT `.bin` cache); bilingual ES/EN intent; `mud_docs.txt` (extracted code docs via `extract_code_docs` tool)
- Objective: **Sampled Softmax** (target + hard/random negatives); not NCE-8
- Checkpoint: prefer `weights/checkpoints/model_latest_checkpoint.mud`

---

## Common Commands

```bash
cargo check --release
cargo clippy --all-targets
cargo test --lib -- --test-threads=2
cargo build --release

./mud.sh tools          # catalog
./mud.sh ci             # tests + clippy + health
./mud.sh audit-full [model.mud]
./mud.sh health [model.mud]
./mud.sh scale-audit <model.mud> [source_dir]   # P1.1: dequant RMS vs BF16 source (detects vocab collapse)
./mud.sh extract-docs                         # Extract code docs → training/corpus/mud_docs.txt

cargo run --release --bin forge_llm -- path/to/model.mud
cargo run --release --bin run_trainer
cargo run --release --bin run_trainer -- models/smollm2.mud --circuit   # infinite seed-driven training circuit
./mud.sh circuit models/ternary_bonsai_1.7b.mud   # REQUIRES healthy .mud (smollm2.mud/checkpoint are healthy; collapsed base in models/smollm2.collapsed.bak); TUI + logs/circuit.log
cargo run --release --bin gemv_auto_bench   # GEMV break-even on this silicon
cargo run --release --bin mud_full_audit -- models/smollm2.mud
./mud.sh cert-loss models/smollm2.mud      # Stream K loss-must-fall (fast)
cargo test --lib loss_cert                 # pure trajectory gate (always CI)
```

**Ver progreso del circuito en vivo (otra terminal):**
```bash
tail -f logs/circuit.log        # eventos timestamp [HH:MM:SS], uno por línea
```
`logs/circuit.log` se escribe desde `trainer_ui::circuit_event` (creado por `mud.sh` antes de arrancar). Cada fase imprime: semilla, `FASE=`, batería restante, duración, total, y resultado de honors (`HONORES` / `SIN HONORES`).

---

## Tokenizer

- BPE from MUD metadata (`tokenizer.tokens` / merges) when present
- Fallback: raw bytes
- Vocab size is **metadata-driven** (e.g. 128256 on some models) — never assume

---

## Historical Fix Log (condensed)

Use for archaeology; status of features is **GEMINI §0 / §6**, not this list.

| # | Topic | Outcome |
|---|--------|---------|
| 1–7 | RMSNorm i8, JEPA var_ema, AR loop, workspace order, output_norm, save SIGSEGV, FFN scales | Fixed |
| 8–9 | JEPA collapse, residual/iscale/adaptive clip | Fixed; later superseded by f32 registers |
| 10 | **f16-packed dual (re, im) u32** register | Active contract (P-02); f32 compute, f16 storage |
| 11–13 | NaN i8 GEMV path, mHC alpha/beta collapse, iscale leak on logits | Fixed; FP32 GEMV path |
| 14 | DeepSeek-V4 research (mHC/Muon/DSpark/CSA) | mHC + Muon LIVE; CSA top-k (stream E); **DSpark now OPEN-SOURCE (MIT, DeepSpec 2026-07):** semi-autoregressive drafter + confidence head — see `docs/research/DEEPSEEK_DSPARK_RESEARCH_2026-07-20.md` (track DSP-1…DSP-5); **DSPark stubs: claim VERDAD ESTRUCTURA CORREGIDA (2026-08-03)** — AGENTS decía "WIRING COMPLETADO (DSparkConfig + DSparkDrafter en dypy.rs, MUD_DSPARK=1, 13 tests)" pero el código vivo es `SlimeDrafter<'a>` en `src/mud/speculative.rs` (7 tests); `dspy.rs` = `DeclarativeSignature` trait (DECL-01), no stubs; no hay env `MUD_DSPARK`; **D1 self-play COMPLETADO** (2026-07-28: `self_play_training_step` en `session.rs:1644`, `MUD_SELF_PLAY=1`, **22 tests** incluyendo `SelfPlayCurriculum` + `should_continue` + `self_play_reward`); **D2 bf16 shadow training COMPLETADO** (2026-07-28: `store_shadow_bf16` en `optim.rs:434/453`, `MUD_BF16_SHADOW=1`, 3 tests); **Vulkan backward shader (4.1) IMPLEMENTADO** (`ternary_backward_dx.comp`, `gemv_pipe` family, safe on Iris Xe); **miri COMPLETADO** (2026-07-28: 11/11 tests, 0 UB, dos bugs de alineación corregidos); 1M LSH still open |
| 15–19 | Aphasia/repetition, tanh JEPA, mHC radius, gate rewire, spring/repulsion removal | Fixed |
| 20–22 | P-13 paths, converter `--check`, telemetry columns | Done (cleanup remains L-07) |
| 23–31 | PCorePool×8, AOT cache, drafter P-01, ash clippy, loss temp scale, QAT zero-alloc, ash Drop UAF | Done |
| 32–35 | Vocab dump (WD on emb), emb dequant, batch4 stride, STE shadow overwrite, sparsity 0.7 | Done |
| 36 | Muon disconnected / dead ASM audit | **L-01/L-02/L-04 done; Adam moments stream A** |
| 37 | Zero-variance illusion (subnormal emb + print) | Fixed via adaptive `scale_up` |
| 38 | Empty telemetry TUI (live run not observed) | Fixed: `[TELEM]`→log + key parser + ΔW panel (TLM) |
| 39 | Hot-loop overhead (branchy dequant/pack) | Fixed: LUT + raw-ptr 8/u32 (P-00/P-01) |
| 40 | Word fusion bug (`corpus_trainer` / `cmud_train` line-split stripped `Ġ`) | Fixed: continuous document tokenization in `corpus_trainer` & `cmud_train`; added `has_space_prefix` + unit test |
| 41 | **Q2_0→ELUT+PRQ per-row scale averaging** — promediar 16 scales por fila destruye semántica; logits casi-uniforme | **Fixed 2026-07-23:** conversor almacena F32 dequantizado bloque a bloque (`Q2ToF32`); inferencia usa `f32_gemv_rowwise` / `is_f32_weights` flag; backward guardado contra SEGFAULT; dead code eliminado. Ver `docs/audits/BUG_LOGITS_CORRUPTION.md`. |
| 42 | **ΔW tracking no cubría el embedding** — `hash_trained_weights` hacía hash de `token_embd.weight` pero `sync_shadow_to_mud` no comparaba old-vs-new bytes → hash cambiaba pero ΔW reportaba 0 | **Fixed 2026-07-26:** agregada comparación old-vs-new para `token_embd.weight` (ternario + F32) y `token_embd.prq_scale` en `checkpoint.rs`. Ver `docs/research/MUD_VULKAN_DELEGATION_ANALYSIS_2026-07-26.md`. |
| 43 | **15+ `vec![]` allocations por chunk en `train_on_sequence`** (P-01 violation, ~100 KB/chunk) | **Fixed 2026-07-26:** 15 buffers pre-allocados en `SlimeBackwardWorkspace` (`x_data`, `pre_norm_x`, `final_x`, `class_embs_flat`, `class_logits`, `prob_q`, `d_logits`, `grad_in`, `grad_out`, `neg_ids`, `stp_hist`, `stp_grad_t/r/s`); `std::mem::take` para extraer/restaurar. 397 tests ✅. |
| 44 | **Vec intermedio `ternary_data` (~1.2 GB para embedding 152k×2048, ~4.7 MB por tensor 2048×576)** | **Fixed 2026-07-26:** packing directo en buffer de salida via raw pointers (`read_unaligned`/`write_unaligned`); eliminado `pack_ternary` de production path (marcado `#[cfg(test)]`). |
| 45 | **F32 backward era un no-op** — `backward_slime_block` retornaba early para `is_f32_weights`, copiando `grad_in→grad_out` sin calcular gradientes de pesos. El optimizer no tenía nada que actualizar para modelos Q2_0→F32. | **Fixed 2026-07-29:** implementado backward completo para F32 weights usando `f32_gemv_backward` (dL/dW = grad_y ⊗ x^T, dL/dx = grad_y^T @ W) en todos los 8 pasos (FFN Down/Up/Gate, FFN ReLU², FFN/Attn RMSNorm, Attn O/Q/K/V, SDP Attention STE Proxy). `f32_gemv_backward` agregado en `slime_forward.rs:448-481` con 2 tests. 413 tests ✅. |
| 46 | **4 errores de lifetime en `checkpoint.rs`** — el patrón `SYNC_PACKED_BUF.with(|b| { let mut v = b.borrow_mut(); v.resize(...); v })` retornaba un `RefMut` cuyo lifetime no podía escapar del closure `with()`. | **Fixed 2026-07-29:** reestructurado a `with_borrow_mut`/`with_borrow` con todo el trabajo dentro del closure. También eliminado `Vec::with_capacity` + `.to_vec()` en el path `scales_only` de `update_layer_tensor`. |

### Mislabelled claims (historical — all corrected)

| Old claim | Correct status (2026-07-16) |
|-----------|----------------|
| Muon not on step | **LIVE** L-01 + L-02 |
| SGD only | Strategies preprocess + STE; **Adam moments LIVE** |
| Dual-f16 accum law | **Reactivated** as current P-02 (f16 dual packing post 2026-07-22); f32 `matmul_accum` was an intermediate phase |
| Absolute Rayon ban all crates | **Runtime ban**; prefer no Rayon in autograd hot path |

---

## P-13 soft debt

**Done (L-07):** EPSILON SSOT; pool via `default_pcore_threads()` / `MUD_PCORE_THREADS`; mHC fail-fast.

**Still soft:**
- Display panels may `unwrap_or(0)` for printing only
- `max_gen` / some train HPs CLI-fixed
- Prefer `EPSILON_FLOOR` when touching stray `1e-8`

**Rule:** if model **dims** missing → panic/error, never invent hidden sizes.

---

## Optimizer selection (LIVE)

```text
square-ish        → Muon (NS iters 5; GPU NS if MUD_USE_VULKAN=1)
tall (rows≫cols)  → GaLore
wide (cols≫rows)  → ChunkedAdam
huge embedding    → SparseAdam (zero-row skip)
else              → Adam (moments)
```

`select_optimizer` + `apply_optimizer_cpu_step_and_pack` — **wired**.

---

*Agents: implement against **`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`** (`R-##`, backlog activo) and `GEMINI §0.3`/§6.2. `MUD_IMPROVEMENTS_POST_AE.md` es backlog histórico: úsalo sólo si no contradice la auditoría 2026-10-11. Do not reopen closed L-## without evidence.*
