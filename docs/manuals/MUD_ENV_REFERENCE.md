# MUD — Referencia de Variables de Entorno (`MUD_*`)

> **SSOT de ejecución:** este documento se genera a partir del código (`grep -rhoE 'MUD_[A-Z0-9_]+' src tools mud.sh`) y aprobado por `mud.sh env-audit`.
> **Regla:** ninguna variable `MUD_*` aparece en este archivo sin que exista un `env::var("MUD_*")` (o un export en `mud.sh`) que la consuma. Las variables documentadas‑y‑ausentes del código están en [§3 — Retiradas](#3-variables-retiradas).
> **Última regeneración:** 2026-10-11 (Ola 0, R-05).

---

## 0. Convenciones

- **`R`** — leída por el código Rust (`std::env::var` / `env::var`).
- **`W`** — escrita por el código (`std::env::set_var`).
- **`S`** — exportada por `mud.sh` (shell).
- **Default** — valor cuando la variable **no** está definida.
- **Valores aceptados** — forma canónica; la mayoría tolera `1|true|on|yes` (activado) y `0|false|off|no|cpu` (desactivado) vía `env_flag_true`/`env_flag_false` (c.º `src/mud/constants.rs`).
- Los defaults numéricos vienen de `src/mud/constants.rs`, con clamps documentados en cada reader.

---

## 1. Tabla maestra

> 117 variables leídas por el código (`R`). Organizadas por subsistema. Las de `mud.sh`‑only están marcadas `S` en la columna *Origen*.

### 1.1 Inferencia / REPL (`main.rs`, `inference.rs`)

| Variable | Default | Valores | Propósito | Origen |
|---|---|---|---|---|
| `MUD_DEBUG` | `0`/`false` | `1\|true` | Verbose logits + top‑10 + `reg_f32` por token en el REPL | `main.rs:229` R |
| `MUD_SYSTEM_PROMPT` | `""` | string | System prompt para el chat (formato Llama‑style) | `main.rs:461` R |
| `MUD_INFER_MAX_TOKENS` | `128` (clamp 8..1024) | usize | Máx tokens a generar por prompt | `main.rs:539` R |
| `MUD_INFER_TEMPERATURE` | `0.8` (>0) | f32 | Sampling temperature (0 = deterministic greedy) | `main.rs:464` R |
| `MUD_INFER_TOP_P` | `0.95` (0..1) | f32 | Top‑p nucleus sampling | `main.rs:471` / `debate_trainer.rs:510` R |
| `MUD_INFER_REP_PENALTY` | `1.25` | f32 | Penalización de repetición | `main.rs:478` R |
| `MUD_INFER_GREEDY` | `0`/`false` | `1\|true` | Forzar decodificación greedy (ignora temp/top_p) | `main.rs:688` R |
| `MUD_INFER_MIN_PEAK` | `0.05` | f32 | Pico mínimo tras DC‑bias para el boost termodinámico | `main.rs:646` R |
| `MUD_INFER_TARGET_PEAK` | `8.0` | f32 | Pico objetivo tras boost (clamp `0.5..16`) | `main.rs:650` R |
| `MUD_INFER_SCALE_UP` | `1` (ternario); `1` (f32) | `auto\|N` | Escala de logits (`auto` = clamp `1/rms` a `[1,8]`) | `main.rs:359` R |

### 1.2 Dispatch GEMV / Vulkan (`ash_backend.rs`, `gemv_policy.rs`, `gemv_gpu.rs`)

| Variable | Default | Valores | Propósito | Origen |
|---|---|---|---|---|
| `MUD_GPU_GEMV` | `auto` en código; `0` en `mud.sh:48` | `0\|1\|auto` | `0`=CPU siempre; `1`=GPU cuando `n_in·n_out ≥ min`; `auto`/unset = micro‑bench one‑shot. **Decisión R-48 (2026-10-13):** los scripts de producción fijan `0` (Iris Xe SIGSEGV en `submit_and_wait`); `auto` queda como opt-in explícito del operador en hosts sin ese bug | `gemv_policy.rs:70` R / `mud.sh:48` S |
| `MUD_GPU_GEMV_MIN` | `256·256 = 65536` (`GEMV_GPU_MIN_WORK`) | usize | Work‑units mínimos para activar GPU (sólo con `MUD_GPU_GEMV=1`) | `gemv_policy.rs:94` R |
| `MUD_GPU_GEMV_LOG` | `0`/`false` | `1\|true` | Imprime la tabla de calibración CPU‑vs‑GPU a stderr | `gemv_policy.rs:101` R |
| `MUD_USE_VULKAN` | `1` | `0\|false` | Activa el backend Vulkan (ash 0.38). `0` lo desactiva | `ash_backend.rs:248,2182` R |
| `MUD_GEMV_ROWS` | `4`; `8` con `MUD_GEMV_ROWS=8` | `4\|8` | Filas por kernel AVX2 de GEMV (opt‑in 8) | `gemv_gpu.rs:20` R |

### 1.3 KV cache — HCA / CSA / compresión (`kv_context.rs`, `csa_indexer.rs`, `kv_compressed.rs`)

| Variable | Default | Valores | Propósito | Origen |
|---|---|---|---|---|
| `MUD_MAX_POS` | `requested_max_pos` (clamp 1..32768) | usize | Contexto lógico máximo (override para ventanas largas) | `kv_context.rs:45` R |
| `MUD_HCA_WINDOW` | `256` (clamp 32..4096) | usize | Ventana densa reciente antes del compression | `kv_context.rs:51` R |
| `MUD_HCA_RATIO` | `10` (clamp 2..64) | usize | Ratio de compresión del bloque HCA | `kv_context.rs:57` R |
| `MUD_CSA` | on cuando `num_blocks > top_k` | `0\|false\|off` (off); `1\|true\|on\|auto`\|unset (on) | Activar CSA top‑k sobre HCA | `csa_indexer.rs:42` R |
| `MUD_CSA_TOP_K` | `64` (clamp 4..512) | usize | Bloques HCA a conservar tras ranking | `csa_indexer.rs:49` R |
| `MUD_CSA_INDEX_DIM` | `16` (0 = full head_dim) | usize | Dimensión coarse de indexado | `csa_indexer.rs:54` R |
| `MUD_CSA_TAIL` | `4` (clamp 0..64) | usize | Últimos N bloques siempre retenidos | `csa_indexer.rs:58` R |
| `MUD_CSA_LSH` | `0`/`false` | `1\|true` | LSH SimHash prefilter (16 bits, Hamming radius 4) | `csa_indexer.rs:230` R |
| `MUD_CSA_LSH_BITS` | `16` (clamp 4..64) | usize | Bits del hash LSH | `csa_indexer.rs:237` R |
| `MUD_CSA_LSH_RADIUS` | `4` | u32 | Radio Hamming para candidatos LSH | `csa_indexer.rs:246` R |
| `MUD_KV_COMPRESS` | `0`/`false` | `1\|true` | Activar compresión Delta+LZ4 de KV | `kv_compressed.rs:57` R |
| `MUD_KV_COMPRESS_DELTA` | `1` (sólo si `KV_COMPRESS=1`) | `0\|1` | Delta encoding entre bloques | `kv_compressed.rs:59` R |
| `MUD_KV_COMPRESS_CELLV5` | `0`/`false` | `1\|true` | **RESERVADO** Fase‑3 CellV5 (no implementado → fallback Delta+LZ4 + warn) | `kv_compressed.rs:42,68` R |
| `MUD_KV_BLOCK_SIZE` | `256` (min 16) | usize | Tokens por bloque de compresión | `kv_compressed.rs:63` R |
| `MUD_KV_DTYPE` | `F32` (unset) | `f32\|bf16\|bfloat16\|f16\|half\|float16` | Precisión del KV cache (IEEE half, no Google BF16) | `kv_dtype.rs:23` R |

### 1.4 Entrenamiento — LR, optimizador y clip (`constants.rs`, `optim.rs`, `train_step.rs`)

| Variable | Default | Valores | Propósito | Origen |
|---|---|---|---|---|
| `MUD_QAT_LR` | `0.0005` (align: `0.0008`); clamp `1e-6..0.1` | f32 | LR fija de QAT; al fijarla **desactiva** el schedule cosino | `constants.rs:64` R; `mud.sh:454` S |
| `MUD_LR_RESTART` | `1` (SGDR activado) | `0\|1` | `0` = single‑cosine; `1` = reinicios SGDR | `train_step.rs:63` R / `mud.sh:482` S |
| `MUD_LR_CYCLES` | `max(3, sqrt(total))` (clamp 3..16) | usize | Número de ciclos cosenoidales SGDR | `train_step.rs:73` R |
| `MUD_GRAD_CLIP` | `1.0` (>0 finito) | f32 | Umbral de clip global de gradiente (L2) | `constants.rs:41` R |
| `MUD_MUON_NS_ITERS` | `5` si `MUD_TRAIN_MAX_CHUNKS` unset; clamp 1..16 | usize | Iteraciones Newton‑Schulz de Muon | `muon.rs:38` R |
| `MUD_OPT` | `sgd` (en `run_trainer`); `auto` (shape‑dispatch) en trainer | `sgd\|adam\|muon\|galore\|chunked` | Estrategia de optimizador (override) | `optim.rs:120` R; `mud.sh:482` S |
| `MUD_TRAIN_OPT` | idem `MUD_OPT` | idem | Alias alternativo de `MUD_OPT` | `optim.rs:118` R |
| `MUD_TRAIN_PACK_POOL` | `0`/`false` | `1\|true` | Pack paralelo del optimizador (sólo si `rows≥64` y pool>1) | `optim.rs:500` R |
| `MUD_TRAIN_WCLAMP_K` | `8` (0=off) | f32 | Clamp de shadow a ±K·absmax antes de pack (previene inflación de escala) | `optim.rs:439` R / `mud.sh:439` S |

### 1.5 Entrenamiento — packing, ventanas y capas (`sequence_pack.rs`, `session.rs`, `train_step.rs`)

| Variable | Default | Valores | Propósito | Origen |
|---|---|---|---|---|
| `MUD_TRAIN_FULL_SEQ` | `1`/`true` (causal windows) | `0\|false\|off\|pairs` (pairs@pos0) | Entrenamiento full‑seq causal vs pares independientes | `sequence_pack.rs:543` R / `mud.sh:477` S |
| `MUD_TRAIN_SEQ_LEN` | `32` (clamp 2..512) | usize | Ventana de tokens full‑seq | `sequence_pack.rs:555` R |
| `MUD_TRAIN_STEPS_PER_CHUNK` | `batch×mult` (align/quick más densa) | usize (clamp 1..4096) | Pasos de optimización por chunk | `sequence_pack.rs:34` R |
| `MUD_TRAIN_AOT_TOKENS` | `512` si `MAX_CHUNKS` set; else auto | usize (clamp 64..CHUNK_SIZE) | Tokens AOT por chunk | `session.rs:843` R |
| `MUD_TRAIN_ALIGN` | `0`/`false` | `1\|true` | Modo *align/quick* (LR más alto, más densidad) | `sequence_pack.rs:38` R / `mud.sh:549` S |
| `MUD_TRAIN_MAX_CHUNKS` | unset (∞) | usize (>0) | Cap de chunks por época (quick mode) | `sequence_pack.rs:41` R |
| `MUD_TRAIN_NUM_NEG` | `63` (align/quick) / `255` (full) (clamp 1..511) | usize | Negativos del sampled softmax | `sequence_pack.rs:58` R |
| `MUD_TRAIN_LAST_N_LAYERS` | auto (align: min(8,nL); nL>16: 4; nL>8: 8; sino: todas) | usize\|`auto\|all\|full`\|`0`(todas) | Últimas N capas a entrenar | `sequence_pack.rs:91` R |
| `MUD_TRAIN_FWD_LAST_N` | todas (`n_layers`) | usize\|`0\|off\|full\|all\|auto` | Sólo forward: últimas N capas (para cálculo de residual) | `sequence_pack.rs:137` R |
| `MUD_TRAIN_EXPERT` | unset (denso `expert.0`) | N | Entrenar experto denso N (`ffn_up/gate/down`) | `debate.rs:246` R |
| `MUD_MOE_TRAIN` | `Off` | `0\|false\|off\|no`(off); `1\|true\|on\|yes\|round_robin\|rr\|all`(round_robin); `hash\|route\|joint\|2`(hash) | Modo MoE: off / round_robin / hash | `moe_train.rs:59` R |
| `MUD_MOE_CLONE` | unset | usize | Clonar experto 0 a N expertos (sólo si todos id 0) | `moe_load.rs:200` R |
| `MUD_MOE_TOP_K` | `2` (clamp 1..8) | usize | Top‑k experts por token | `moe_load.rs:269` R |
| `MUD_TRAIN_MATERIALIZE_FULL` | `0`/`false` | `1\|true` | Materializar todos los pesos a FP32 (vs STE lazy mmap) | `session.rs:67` R |
| `MUD_GRAD_CKPT` | auto (on si `seq_len ≥ MUD_TRAIN_CKPT_SEQ_THR` y unset) | `0\|1\|true\|false` | Checkpointing de activaciones (L‑15 segmented recompute) | `sequence_pack.rs:565,577` R |
| `MUD_GRAD_CKPT_SEG` | `4` (`DEFAULT_SEGMENT`, min 1) | usize | Capas por segmento de checkpoint (count = ceil(L/SEG)) | `grad_checkpoint.rs:56,59` R |
| `MUD_GRAD_CKPT_RESIDUAL` | `0`/`false` | `1\|true` | Residual‑bank recompute path (L‑15 stream H) | `grad_checkpoint.rs:286` R |
| `MUD_TRAIN_CKPT_SEQ_THR` | `64` | usize | Seq len ≥ N activa L‑15 segmentado | `sequence_pack.rs:565` R |
| `MUD_TRAIN_MAX_POS` | `512` (clamp 64..max_pos) | usize | Contexto lógico máximo para entrenamiento | `session.rs:760` R |
| `MUD_AWAKE_ITERS` | `12` (4 en quick si `MUD_TRAIN_MAX_CHUNKS ≤ 16`) | usize | AWAKE‑01 alignment iterations (deep local alignment) | `alignment.rs:94` R |
| `MUD_TRAIN_AWAKE` | `0`/`false` | `1\|true` | Opt‑in a AWAKE‑01 deep local alignment | `session.rs:289` R |
| `MUD_TRAIN_SKIP_AWAKE` | `0`/`false` | `1\|true` | Saltar la fase AWAKE-01 aun con `MUD_TRAIN_AWAKE=1` | `session.rs:292` R |
| `MUD_TRAIN_PROGRESS_EVERY` | `25` (>0) | usize | Cadencia de impresión de progreso | `session.rs:1609` R |

### 1.6 Entrenamiento — checkpoint, shadow y pesos (`checkpoint.rs`, `session.rs`, `constants.rs`)

| Variable | Default | Valores | Propósito | Origen |
|---|---|---|---|---|
| `MUD_TRAIN_FREEZE_EMB` | `1`/`true` | `0\|false\|off`(descongela) | Congelar token_embd (ahorra ~1 GiB FP32 en vocab grandes) | `constants.rs:148` R / `mud.sh:49` S |
| `MUD_TRAIN_SCALES_ONLY` | `0`/`false` | `1\|true` | Sólo refrescar PRQ scales, dejar códigos ELUT | `checkpoint.rs:159` R / `mud.sh:453` S |
| `MUD_TRAIN_DEBUG_DW` | `0`/`false` | `1\|true` | Reportar ΔW ternario + ΔPRQ scale por checkpoint en logs | `checkpoint.rs:164` R / `mud.sh:53` S |
| `MUD_BEST_CHECKPOINT` | `1`/`true` | `0\|false` | Guardar mejor checkpoint por eval (rollback safety net) | `mod.rs:121` R / `mud.sh:55` S |
| `MUD_TRAIN_CKPT_EVERY` | `50` (0 = sólo epoch/end) | usize | Frecuencia de checkpoint (chunks) | `mod.rs:53` / `run_trainer.rs:53` R,S |
| `MUD_TRAIN_CKPT_DIR` | `<cwd>/weights/checkpoints` | path | Directorio de checkpoints | `run_trainer.rs:137` R,S |
| `MUD_TRAIN_CORPUS_DIR` | `<cwd>/training/corpus` | path | Directorio de corpus | `run_trainer.rs:183` R,S |
| `MUD_TRAIN_RESET_EPOCH` | `0`/`false` | `1\|true` | Reinicia contador de época (evita resume no‑op) | `session.rs:819` R |
| `MUD_TRAIN_EZOP` | `1` | `0\|off\|false\|no\|cpu`(off); `1`(on) | Path EZOP (zero‑alloc raw‑ptr); `cpu` fuerza off | `session.rs:235` R / `mud.sh:49` S |
| `MUD_BF16_SHADOW` | `0`/`false` | `1\|true` | Sombra bf16 para Adam/SparseAdam (además FP32) | `adam_state.rs:46` R / `mud.sh:64` S |
| `MUD_SKIP_INTEGRITY` | `0`/`false` | `1\|true` | Saltar el gate de integridad pre‑entrenamiento (debug) | `session.rs:41` R / `mud.sh:55` S |
| `MUD_TRAIN_TEXT_ONLY` | `0`/`false` | `1\|true` | Corpus: sólo `*.txt`/`*.md` (excluye `*.rs`) | `corpus.rs:90` R / `mud.sh:64` S |

### 1.7 Entrenamiento — JEPA / mHC / STP / Self‑play (`st0_loss.rs`, `self_play.rs`)

| Variable | Default | Valores | Propósito | Origen |
|---|---|---|---|---|
| `MUD_TRAIN_STP` | `0`/`false` | `1\|true` | STP trajectory loss (JEPA→geodésica) | `stp_loss.rs:179` R |
| `MUD_TRAIN_STP_LAMBDA` | `0.05` (≥0, finito) | f32 | Peso λ del STP loss | `stp_loss.rs:186` R |
| `MUD_SELF_PLAY` | `0`/`false` | `1\|true` | Self‑play (generación sintética + jitter) | `session.rs:1059` R |
| `MUD_SELF_PLAY_TURNS` | `8` (>0) | usize | Longitud de generación por paso self‑play | `session.rs:1062` R |
| `MUD_SELF_PLAY_TEMP` | `0.8` (>0) | f32 | Temperatura self‑play | `session.rs:1067` R |

### 1.8 Debate / RLVR / circuit (`arena_judge.rs`, `circuit.rs`, `debate_trainer.rs`, `loss_cert.rs`)

| Variable | Default | Valores | Propósito | Origen |
|---|---|---|---|---|
| `MUD_DEBATE_MODE` | `debate` | `debate\|professor\|games` | Modo de debate (sólo circuit) | `debate.rs:196` R |
| `MUD_DEBATE_LEARN` | `1` (true en circuito) | `1\|true\|0` | Persisten shadow→.mud al final; `0` = replay sin writeback | `circuit.rs:43` R,W / `mud.sh:55` S |
| `MUD_DEBATE_MAX_NEW_TOKENS` | auto (free RAM); `3` en integrity bench | usize | Cap de tokens por respuesta | `debate_trainer.rs:151` R |
| `MUD_DEBATE_MAX_TIME` | `0` (infinito) | u64 | Segundos antes de auto‑stop (circuit: 120 s) | `debate_trainer.rs:156` R / `mud.sh:55` S |
| `MUD_DEBATE_BOS` | `0` | u32 | Token BOS para debate | `debate_trainer.rs:147` R |
| `MUD_DEBATE_RWIN` | `1.0` | f32 | Recompensa por ganar una ronda | `arena_judge.rs:39` R |
| `MUD_DEBATE_RLOSE` | `0.7` (`DEBATE_R_LOSE`) | f32 | Penalización por perder | `arena_judge.rs:43` R |
| `MUD_DEBATE_JEPB_LAMBDA` | `0.05` (`DEBATE_JEPA_LAMBDA`) | f32 | Peso JEPA intrínseca en debate | `debate_trainer.rs:643` R |
| `MUD_CIRCUIT_BATCH` | `16` | usize | Batería por fase en circuito | `circuit.rs:57` R |
| `MUD_CIRCUIT_EPOCHS` | `1` | usize | Épocas por fase de circuito | `circuit.rs:53` R |
| `MUD_CIRCUIT_EVAL` | `1`/`true` | `0\|false` | Ejecutar gate de eval en circuito (integridad + benchmark) | `circuit.rs:128` R |
| `MUD_CIRCUIT_MAX_CYCLES` | `100_000` (0 = ilimitado) | usize | Cap duro de ciclos del circuito infinito | `circuit.rs:69` R / `mud.sh:55` S |
| `MUD_CIRCUIT_MAX_PER_MODE` | `120` (seg) | u64 | Time‑box por fase (evita freeze) | `circuit.rs:48` R / `mud.sh:55` S |
| `MUD_CIRCUIT_TUI` | unset | `1` | Gate interno: suprime stdout raw cuando el TUI corre | `mod.rs:6` R |
| `MUD_TELEM_EVERY` | `1` (>0) | usize | Cadencia de líneas `[TELEM]` | `session.rs:1049` R |

### 1.9 C‑MUD complejo / SlimeX / Fenwick (`cmud.rs`, `cmud_params.rs`, `slime.rs`, `fenwick.rs`)

| Variable | Default | Valores | Propósito | Origen |
|---|---|---|---|---|
| `MUD_CMUD_THINK` | `0`/`false` | `1\|true` | C‑MUD thinker (manifold complejo) opt‑in | `cmud.rs:880` R / `mud.sh:64` S |
| `MUD_CMUD_PARAMS` | unset (`""`) | path | Sidecar JSON de parámetros C‑MUD | `cmud.rs:915` R / `mud.sh:64` S |
| `MUD_CMUD_ALPHA` | `0.05` (`CMUD_DEFAULT_ALPHA`) | f32 (0..1) | LR de fase posicional | `cmud.rs:907` R |
| `MUD_CMUD_SIGMA` | `p.sigma` | f32 (`0` = off) | Amplitud del "respirar" de fase | `cmud.rs:551,669` R |
| `MUD_CMUD_POS_PHI` | `0.1` (`CMUD_POS_PHASE_STEP`) | f32 | Paso de fase posicional | `cmud.rs:410` R |
| `MUD_CMUD_WIN` | `16` (`CMUD_WIN_HALF`) | usize | Ventana semialeatoria de fase | `cmud.rs:485,593` R |
| `MUD_CMUD_LOCK_EPS` | `1e-3` (`PHASE_LOCK_EPS`) | f32 | Épsilon de fase‑lock | `cmud.rs:852` R |
| `MUD_CMUD_CELLV5` | `0`/`false` | `1\|true` | Empaquetar estado C‑MUD en CellV5 u32 (8:8:8:8) — opt‑in (P‑02 safe) | `cmud_params.rs:148` R / `mud.sh:64` S |
| `MUD_SLIMEX` | `0`/`false` | `1\|true` | SlimeX CellV5 activo (16 KB → L1) | `slime.rs:322` R / `mud.sh:55` S |
| `MUD_FENWICK` | `1`/`true` | `0\|false` | Árboles de Fenwick (prefijo/sampling); fallback lineal si `0` | `fenwick.rs:551` R / `mud.sh:64` S |

### 1.10 Hilos / hardware (`constants.rs`, `pcore_pool.rs`)

| Variable | Default | Valores | Propósito | Origen |
|---|---|---|---|---|
| `MUD_PCORE_THREADS` | min(logical cores, `PCORE_THREADS_CAP=8`) (clamp 1..64) | usize | Hilos del PCorePool (Rayon OFF en runtime) | `constants.rs:112` R / `mud.sh:47` S |
| `MUD_PIPELINE_OVERLAP` | `0`/`false` | `1\|true` | Double‑buffer GPU GEMV + CPU readback (Vulkan) | `ash_qat_dispatcher.rs:882` R / `mud.sh:64` S |

### 1.11 Variables de *shell* (`mud.sh` — S) usadas en trainers/benches

> Estas NO se leen via `env::var` en el *runtime* Rust del trainer, pero `mud.sh` (o herramientas CLI) las exportan/consume. Se listan para que `env-audit` las reconozca como *shell‑vars*.

| Variable | Default | Propósito | Uso en `mud.sh` |
|---|---|---|---|
| `MUD_EPOCHS` | `5` | Épocas (cli `--epochs`) — **no leída por `env::var`**; pasada como flag a `run_trainer` | `mud.sh:453` S |
| `MUD_TOKENIZER_DIR` | `models/smollm2` | Dir de tokenizer (CLI `convert`) — **no leída por `env::var`** | `mud.sh:746` S |
| `MUD_DW_DEBUG` | `1` (en `mud.sh health`) | Alias shell de `MUD_TRAIN_DEBUG_DW` | `mud.sh:53` S |
| `MUD_MHC_SYNC` | `1` (en `mud.sh health`) | Echo de debug de sync mHC; la lógica la lee el trainer vía `MUD_DEBATE_LEARN`/checkpoint implícito | `mud.sh:55`, `mud.sh:295` S |
| `MUD_BEST_CHECKPOINT` | `1` | Exportada por `mud.sh train` | `mud.sh:64` S |
| `MUD_CI_LOSS_CERT` | `1` (en `./mud.sh ci`) | Fuerza loss‑cert en CI | `mud.sh:513` S |
| `MUD_TRAIN_FORCE_VULKAN` | unset | Override que fuerza Vulkan en `mud.sh train` (deshabilita el blacklist Iris Xe) | `mud.sh:320,438,458` S |
| `MUD_CMUD_THINK` / `MUD_CMUD_CELLV5` / `MUD_FENWICK` / `MUD_BF16_SHADOW` / `MUD_KV_COMPRESS` / … | (ver tabla 1.x) | Exportadas por `mud.sh` para `slimex-train`/`circuit` | `mud.sh:64` S |

### 1.12 Loss certification (`loss_cert.rs`, `tools/loss_certification_bench.rs`)

| Variable | Default | Valores | Propósito | Origen |
|---|---|---|---|---|
| `MUD_LOSS_CERT_FAST` | `true` (product default) | `1\|true\|0\|false` | Fast cert (endpoints/moving‑mean) vs strict | `loss_cert.rs:332` R |
| `MUD_LOSS_CERT_MODE` | `""` (= `WindowMeans`) | `endpoints\|window_means\|percentile` (case‑insensitive) | Modo de certificación de pérdida | `loss_cert.rs:335` R |
| `MUD_LOSS_CERT_HEAD_PCT` | `0.5` | f32 (0..1) | Cuantil cabeza (modo percentile) | `loss_cert.rs:337` R |
| `MUD_LOSS_CERT_TAIL_PCT` | `0.5` | f32 (0..1) | Cuantil cola (modo percentile) | `loss_cert.rs:341` R |
| `MUD_LOSS_CERT_MIN_REL` | `0.01` | f32 | Drop relativo mínimo para pasar cert | `loss_cert.rs:351` R |
| `MUD_LOSS_CERT_MIN_ABS` | `0.02` | f32 | Drop absoluto mínimo para pasar cert | `loss_cert.rs:356` R |
| `MUD_LOSS_CERT_MIN_POINTS` | `3` | usize | Mín puntos de loss‑curve para cert | `loss_cert.rs:361` R |
| `MUD_LOSS_CERT_WINDOW_FRAC` | `0.35` | f32 (0..1) | Fracción de ventana para head/tail mean | `loss_cert.rs:366` R |
| `MUD_LOSS_CERT_PERCENTILE_BINS` | `256` | usize | Bins del histograma FenwickU32 | `loss_cert.rs:371` R |
| `MUD_LOSS_CERT_BATCH` | `8` (fast) / `4` (full) | usize | Batch size del cert bench (tools‑only) | `loss_certification_bench.rs:87` R |
| `MUD_LOSS_CERT_EPOCHS` | `2` (fast) / `45` (full) | usize | Épocas del cert bench (tools‑only) | `loss_certification_bench.rs:83` R |

---

## 2. Orígenes de lectura (`R`) — índice por archivo

> Cada variable `R` aparece **al menos** en un `std::env::var("MUD_*")`. Si `env-audit` encuentra un nombre en este archivo con un `MUD_*` en `src/tools` que no esté listado, es divergencia.

Los call sites principales (ver `docs/architecture/MUD_ENV_REFERENCE_INDEX.json` para el índice completo de `file:line`):

- **`src/mud/constants.rs`** — `MUD_QAT_LR`, `MUD_TRAIN_ALIGN`, `MUD_GRAD_CLIP`, `MUD_PCORE_THREADS`, `MUD_TRAIN_FREEZE_EMB`
- **`src/mud/corpus_trainer/session.rs`** — `MUD_TRAIN_AOT_TOKENS`, `MUD_TRAIN_AWAKE`, `MUD_TRAIN_SKIP_AWAKE`, `MUD_TRAIN_MATERIALIZE_FULL`, `MUD_TRAIN_MAX_POS`, `MUD_TRAIN_RESET_EPOCH`, `MUD_SELF_PLAY*`, `MUD_TELEM_EVERY`, `MUD_SKIP_INTEGRITY`, `MUD_CIRCUIT_MAX_PER_MODE`
- **`src/mud/corpus_trainer/circuit.rs`** — `MUD_DEBATE_LEARN`, `MUD_CIRCUIT_*`, `MUD_SELF_PLAY`
- **`src/mud/corpus_trainer/train_step.rs`** — `MUD_QAT_LR`, `MUD_TRAIN_FULL_SEQ`, `MUD_LR_RESTART`, `MUD_LR_CYCLES`
- **`src/mud/checkpoint.rs`** — `MUD_TRAIN_SCALES_ONLY`, `MUD_TRAIN_DEBUG_DW`, `MUD_CIRCUIT_TUI`
- **`src/mud/optim.rs`** — `MUD_TRAIN_PACK_POOL`, `MUD_TRAIN_WCLAMP_K`, `MUD_OPT`, `MUD_TRAIN_OPT`, `MUD_MUON_NS_ITERS`
- **`src/mud/sequence_pack.rs`** — `MUD_TRAIN_STEPS_PER_CHUNK`, `MUD_TRAIN_ALIGN`, `MUD_TRAIN_MAX_CHUNKS`, `MUD_TRAIN_NUM_NEG`, `MUD_TRAIN_LAST_N_LAYERS`, `MUD_TRAIN_FWD_LAST_N`, `MUD_TRAIN_FULL_SEQ`, `MUD_TRAIN_SEQ_LEN`, `MUD_TRAIN_CKPT_SEQ_THR`, `MUD_GRAD_CKPT`, `MUD_TRAIN_CKPT_SEQ_THR`
- **`src/mud/kv_context.rs`** — `MUD_MAX_POS`, `MUD_HCA_WINDOW`, `MUD_HCA_RATIO`
- **`src/mud/csa_indexer.rs`** — `MUD_CSA*`, `MUD_CSA_LSH*`
- **`src/mud/kv_compressed.rs`** — `MUD_KV_COMPRESS*`
- **`src/mud/kv_dtype.rs`** — `MUD_KV_DTYPE`
- **`src/mud/cmud*.rs`** — `MUD_CMUD_*`
- **`src/mud/slime.rs`** — `MUD_SLIMEX`
- **`src/mud/fenwick.rs`** — `MUD_FENWICK`
- **`src/mud/stp_loss.rs`** — `MUD_TRAIN_STP`, `MUD_TRAIN_STP_LAMBDA`
- **`src/mud/self_play.rs`** — `MUD_SELF_PLAY*`
- **`src/mud/arena_judge.rs`, `debate_trainer.rs`** — `MUD_DEBATE_*`
- **`src/mud/loss_cert.rs`** — `MUD_LOSS_CERT_*`
- **`src/mud/adam_state.rs`** — `MUD_BF16_SHADOW`
- **`src/mud/moe_load.rs`, `moe_train.rs`** — `MUD_MOE_*`
- **`src/mud/gemv_gpu.rs`, `vulkan/gemv_policy.rs`** — `MUD_GPU_GEMV*`, `MUD_USE_VULKAN`, `MUD_GEMV_ROWS`
- **`src/main.rs`** — `MUD_DEBUG`, `MUD_SYSTEM_PROMPT`, `MUD_INFER_*`
- **`tools/run_trainer.rs`** — `MUD_TRAIN_CKPT_EVERY`, `MUD_TRAIN_CKPT_DIR`, `MUD_TRAIN_CORPUS_DIR`

---

## 3. Variables retiradas

> Estas variables aparecen en documentos históricos (`GEMINI.md` §0.3, `AGENTS.md`, `ROADMAP.md`) como si fueran reales. **No existen en el código** (ni `env::var` ni export en `mud.sh`). `env-audit` las reporta como `RETIRED` y exige que NO se usen.

| Variable | Estado | Razón |
|---|---|---|
| `MUD_BATCH_SIZE` | **RETIRED** | Nunca implementada; el tamaño de batch se derive de corpus/seq_len |
| `MUD_MERGE_ON_SAVE` | **RETIRED** | Nunca implementada; los merges usan `file_fusion` (no env) |
| `MUD_DEBATE_SEEDS` | **RETIRED** | El circuito usa LCG sin RNG (semilla derivada de tiempo) |
| `MUD_DSPARK` / `MUD_DSPARK_*` | **RETIRED** | DSpark fue reemplazado por `SlimeDrafter` (`src/mud/speculative.rs`). El nombre `MUD_DSPARK` NO es un env var real (ver `specs/DECL-01`) |
| `MUD_FENWICK_AVX2` | **RETIRED** | Reemplazado por el único `MUD_FENWICK` (dispatch interno AVX2×scalar) |
| `MUD_FENWICK_BATCH` | **RETIRED** | Reemplazado por `MUD_FENWICK` |

> Nota de honestidad: el `MUD_EPOCHS` seteado en `mud.sh:453` (`slimex-train`) **no se lee** via `env::var` — `run_trainer` recibe `--epochs` como CLI flag. Se mantiene como *shell‑var* documentada en §1.11 para no romper scripts existentes, pero **no** es una variable de entorno Rust.

---

## 4. `env-audit` — cómo verificar (R-05)

`./mud.sh env-audit` ejecuta tres comprobaciones y falla (exit ≠ 0) si hay divergencias:

1. **`missing-docs`** — variable `MUD_*` leída por `env::var` en `src`/`tools` que **no** aparece en §1 de este archivo ⇒ error.
2. **`stale-docs`** — variable listada en §1/§2 que **no** se lee en `src`/`tools` y **no** es shell‑var de `mud.sh` ⇒ error.
3. **`retired-usage`** — presencia de alguna variable §3 (retirada) en `src`/`tools`/`mud.sh` ⇒ error.

```bash
./mud.sh env-audit     # 0 = 0 divergencias (regla R-05)
```

---

## 5. Historia de cambios

| Fecha | Evento |
|---|---|
| 2026-10-11 | Generación inicial (R-05). 117 vars `R` + 7 shell‑vars + 7 retiradas. Elimina las 9 vars "documentadas y ausentes" y documenta las 14 "presentes y sin documentar". |
