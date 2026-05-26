---
lang: es
---

# MUD Audit Consolidado — Estado Vivo

> Este documento consolida los hallazgos de auditoría, resolución de bugs, estado del sistema, y tracking de issues. Reemplaza los siguientes documentos históricos:
> `MUD_AUDIT.md`, `MUD_AUDIT_REPORT_V1.md`, `MUD_AUDIT_RESOLUTION.md`,
> `MUD_CONVERSION_AUDIT.md`, `MUD_STATISTICAL_AUDIT.md`, `MUD_MATHEMATICAL_AUDIT.md`

**Última actualización:** 2026-05-25 (Fixes: INF-02, AT series, BUG-5; Embed ternarization integrado en converter; Qwen conversion; Auto-Trainer rewrite EN REVISIÓN)

---

## 1. Estado del Sistema

### Build & Tests
```
cargo check --release   ✅  0 errores, 0 warnings
cargo build --release   ✅  Éxito (2m 21s, optimized)
cargo test              ✅  21/21 passed, 0 failed
```

### Modelos en Producción / Prueba
| Métrica | core_skills.mud | qwen_mud.mud |
|---------|:---------------:|:------------:|
| Tamaño | 59 MB (ternary) | 122 MB (ternary + emb tern) |
| Parámetros | SmolLM2-135M | Qwen2.5-0.5B |
| Vocab | 49,152 | 151,643 |
| Atención | 9 heads × 64, KV=3, GQA=3 | 14 heads × 64, KV=2, GQA=7 |
| Capas | 30 | 24 |
| Expertos | 1 (MoE sintético) | 1 (MoE sintético) |
| Sigma | ~0.735 | ~0.70 |
| Velocidad | ~100 t/s | ~70 t/s |
| GPU | Vulkan iGPU ADL GT2 | Vulkan iGPU ADL GT2 |
| Knowledge DB | `models/knowledge.db` (314 MB) | — |
| Facts totales | 74,489 | — |
| Facts asimilados | 531 | — |

---

## 2. Bugs Activos (Trackeados)

### 🔴 Críticos — Resolver Primero

| ID | Archivo | Línea | Descripción | Fix Propuesto |
|:--|:--|:--|:--|:--|
| AT-01 | `auto_trainer.rs` | 106, 120 | ~~`.unwrap()` en hilo background → muerte silenciosa del daemon si el `.mud` está corrupto~~ | **FIXED** — Reemplazado con `match` + early return con `eprintln!` + `IS_TRAINING.store(false, ...)`. |
| AT-02 | `auto_trainer.rs` | 227–231 | ~~Acceso a embedding table sin bounds check para `t_in`, `t_target` → panic/OOB si el tokenizador emite IDs >= vocab_size~~ | **FIXED** — Añadido `if t_in >= vocab_size \|\| t_target >= vocab_size { continue; }`. |
| AT-03 | `auto_trainer.rs` + `inference.rs` | 110–117, 344–360 | ~~Data race: trainer escribe sobre mmap mientras inference lo lee — UB en Rust~~ | **FIXED** — Eliminado todo write-back a `.mud` durante training. Pesos quedan en memoria (ExpertShadow cache). Se pierden al cerrar sesión. |
| INF-01 | `inference.rs` | 231, 247 | ~~KV cache OOB: bucle `for t in 0..=_pos` con `_pos >= 4096`~~ | **FIXED** — `_pos.min(4095)` y bounds check implementados. |
| INF-02 | `inference.rs` | 226–228 | ~~`num_heads=4`, `head_dim=64` hardcoded: MHA incorrecta si `hidden_size != 256`~~ | **FIXED** — Ahora lee metadatos dinámicos. |
| MAIN-01 | `main.rs` | — | ~~Sin handler SIGINT: `SHOULD_TERMINATE` nunca se activa~~ | **FIXED** — `ctrlc::set_handler` implementado en `main.rs`. |
| AG-01 | `forge_autograd/lib.rs` | 282–296 | ~~`get_two_mut`/`get_three_mut` sin bounds check en índices~~ | **FIXED** — `assert!` añadidos para prevenir UB. |

### 🟠 Altos

| ID | Archivo | Descripción |
|:--|:--|:--|
| AT-04 | `auto_trainer.rs` | ~~Heurística hardcodeada `t_in / 16 % num_layers` para routing de expertos — entrena expertos equivocados, corrompe MoE~~ | **FIXED** (BUG-4) — Ahora carga `blk.{l}.gate.weight`, dequantiza, computa logits, usa `MudRouter::route()` para seleccionar experto real. Gate trainable en memoria. |
| AT-05 | `auto_trainer.rs` | ~~Weight decay sobre shadow FP32 se pierde en el ternary snap~~ | **FIXED** — Neural Kick v2 + Correct shadow update logic. |
| AT-08 | `auto_trainer.rs` | ~~`IS_TRAINING` no se resetea si ocurre un panic~~ | **FIXED** — `TrainingGuard` con `Drop` trait implementado. |
| INF-03 | `inference.rs` | ~~`norm_w` null no protegido → segfault~~ | **FIXED** — Null checks en todas las proyecciones de norma. |
| INF-04 | `inference.rs` | ~~`conversation_pos` crece sin límite → OOB~~ | **FIXED** — Ventana deslizante (4096 → 4000) implementada. |
| INF-08 | `inference.rs` | ~~`RwLock<usize>` para `active_experts` en hot path~~ | **FIXED** — Migrado a `AtomicUsize`. |
| INF-09 | `inference.rs` | ~~Incoherencia posicional (Word Salad)~~ | **FIXED** — Implementado **Split RoPE** (LLaMA-style) y restauración de escalas de cuantización. |
| PERF-01 | `inference.rs` | ~~`vec![0.0; _pos + 1]` en cada head~~ | **FIXED** — **Zero-Allocation Hot-Loop** integrado en `InferenceWorkspace`. |
| PERF-05 | `tokenizer.rs` | 🔴 Alto | BPE O(n²) → priority queue O(n log n) |
| PERF-08 | `auto_trainer.rs` | 🔴 Alto | `shadow_w{1,2,3}.clone()` × 3 por token para la tape → refactorizar para no clonar |
| PERF-03 | `inference.rs` | 🟠 Medio | `vec![0.0; ...]` × num_experts en rayon → buffer pool |
| PERF-04 | `main.rs` | 🟠 Medio | `type_writer` O(n²) → comparación directa sobre chars |
| PERF-06 | `inference.rs` | 🟡 Bajo | `format!("l{}_q", l)` en hot path → preallocar keys |
| PERF-07 | `forge_autograd` | 🟡 Bajo | `.clone()` en backward pass → usar `split_at_mut` o índices |
| PERF-09 | `inference.rs` | 🟡 Bajo | KV cache 96MB+; INT8 reduciría 75% → cuantización K/V |
| PERF-10 | `forge_autograd` | 🔵 Info | `axpy_avx2` sin prefetch manual → +15% posible en vectores >1024 |

---

## 3. Resolución de Bugs (Historial)

Issues corregidos y verificados en sesiones anteriores:

| # | Hallazgo | Archivos Modificados | Fix |
|---|---|---|---|
| 1 | RoPE incompatible (base vs half) | `src/model/transformer.rs` | Unificado a LLaMA-style (adjacent-pair, base=10k) |
| 2 | Softmax produce NaN si sum_exp=0 | `src/mud/inference.rs`, `transformer.rs` | Agregado `+ 1e-30` al denominador y `inv_sum` |
| 3 | PageRank pierde rango (dangling) | `src/mud/graph.rs` | Redistribución equitativa para nodos sin salida |
| 4 | Temperature=0 causa Inf | `src/mud/inference.rs` | `temperature.max(1e-8)` y tipos f32 explícitos |
| 5 | RngExt error + falta de tests | `tools/ternary_audit.rs`, `src/asm/tests.rs` | Corregido `rand` y agregados 9 tests AVX2 |
| 6 | hidden_size hardcodeado en facts | `src/mud/store.rs`, `web_search.rs` | Dinamización vía metadatos |
| 7 | Damping 1/sqrt(N) arbitrario | `src/mud/inference.rs` | Normalización `1/sqrt(2)` para pre-norm |
| 8 | Sesgo en sampleo (r > cum_sum) | `src/mud/inference.rs` | Guard `cum >= 1.0 - 1e-7` y fallback a UNK |
| 9 | Sandbox sin funciones avanzadas | `tools/math_sandbox.py` | Agregadas: sqrt, sin, cos, tan, log, exp, abs, round, pi, e |
| 10 | output_norm_w null sin check | `src/model/inference.rs` | Validación explícita `if !is_null()` |
| 11 | Alignment memory no verificado | `src/mud/inference.rs` | `assert!` para `hidden_size % 16` y `% 64` |
| 12 | Vulkan .unwrap() en hot-path | `src/mud/inference.rs` | Fallback silencioso a CPU si GPU falla |
| 13 | vec![] allocation en hot-loop | `src/mud/inference.rs` | Migrado a `InferenceWorkspace` |
| 14 | Pipeline recompilación Vulkan | `vulkan_backend.rs` | Cacheo de pipeline en `VulkanContext` |
| 15 | Weight persistence (Empty Brain) | `v37_master_trainer.py` | `combined_sd` con model + embed |
| 16 | Tokenizer whitespace (Word Salad) | `tokenizer.rs` | Implementado `Ġ` mapping |
| 17 | Escalas ternarias ignoradas | `MudExporter`, `MudInference` | Load y apply de `.scale` tensors |
| 18 | Prefill logic ausente | `MudInference` | `engine.prompt()` para procesar contexto inicial |
| 19 | Overflow en kv_cache allocation | `inference.rs` | `checked_mul()` con `expect()` |
| 20 | Expert indexing sin bounds | `inference.rs` | `layer.experts.get(expert_id)` con `filter_map` |
| 21 | Logit bounds sin protección | `inference.rs` | `ws.logits.get_mut(idx)` |
| 22 | embed_token sin checked_mul | `inference.rs` | `checked_mul()` + `debug_assert!` |
| 23 | mmap load sin bounds check | `mod.rs` | `checked_add()` + bounds assert |
| 24 | Router: empty/sum_exp=0 sin guard | `routing.rs` | Early return + fallback a mejor experto |
| 25 | SQLite: sin WAL, sin timeout | `store.rs` | `PRAGMA WAL` + `busy_timeout=5000ms` |
| 26 | SQLite: sin LIMIT en queries | `store.rs` | `LIMIT 1000` en `get_unassimilated()` |
| 27 | SQLite: NaN en rank | `store.rs` | `clamp(0.0, 1e6)` antes de persistir |
| 28 | Trainer: sin noise annealing | `trainer.py` | `_step_ratio` + `noise_std = 0.1 * (1.0 - step_ratio)` |
| 29 | Trainer: aux_coeff hardcodeado | `trainer.py` | Propagación por constructor |
| 30 | Space symbol auto-detection | `universal_converter`, `tokenizer.rs` | Auto-concordance de `Ġ` vs `\u{2581}` |
| 31 | INF-02: MHA hardcoded (num_heads=4, head_dim=64) | `inference.rs`, `universal_converter/main.rs` | Metadata dinámica: num_heads, num_kv_heads, head_dim desde Q/K shapes. GQA mapping Q→KV con kv_group. GEMV K/V usa kv_out real (elimina UB). |
| 32 | AT-03/AT-04: Trainer corrompe modelo (escribe .mud + heuristic routing) | `auto_trainer.rs` | Routing por gate network real. No más write-back a .mud (pesos en memoria). |
| 33 | AT-01/AT-02: Panics silenciosos + OOB en trainer | `auto_trainer.rs` | .unwrap() → match + early return. Bounds check en tokens. |
| 34 | BUG-5: LAST_ACTIVITY seconds vs ms mismatch | `auto_trainer.rs`, `main.rs` | Unificado a milisegundos. `saturating_sub(60_000)` para threshold 60s. |
| 35 | Embedding ternarization (prototipo) | `tools/embed_audit.rs`, `tools/embed_ternarize.rs` | Row-wise absmean + scales u8. 108MB → 6.8MB (15.9×). End-to-end inference sin crash con SmolLM2. |
| 36 | GEMV K/V n_out corregido | `inference.rs` | K/V projections ahora usan `kv_out = num_kv_heads × head_dim` en vez de `hidden`, eliminando UB por lectura fuera de bounds. |
| 37 | Embedding ternarization integrado en converter | `universal_converter/main.rs`, `quantizer.rs`, `inference.rs` | `--ternarize-emb` flag. Row-wise absmean + embed_scales tensor + dequant en embed_token(). |
| 38 | Qwen2.5-0.5B conversion | `universal_converter/parser.rs`, `downloaded_model.safetensors` | Bias tensors ignorados (parser fix). GQA 14:2 heads. Vocab 151k. 943 MB → 122 MB (7.7×). |
| 39 | Auto-Trainer rewrite (EN REVISIÓN) | `auto_trainer.rs` | Multi-capa, escalas, persistencia, gate trainable, full voc CE, type-safe. Pendiente revisión. |
| 40 | Doc overhaul | `docs/*` | Roadmap + Audit + Embed Ternarization docs actualizados. |

---

## 4. Hallazgos de Auditoría de Rendimiento

### 4.1. Memory Safety (unsafe)
- `src/asm/mod.rs` y `src/model/transformer.rs`: funciones que dereferencian raw pointers sin marcarse `unsafe`
- `src/vulkan_backend.rs`: funciones FFI correctamente marcadas `unsafe` pero sin bloque `# Safety`

### 4.2. Loop Vectorization
- `src/model/transformer.rs` y `src/vulkan_backend.rs`: uso extensivo de `for i in 0..n { arr[i] = ... }` en vez de `.iter_mut()`
- `src/mud/mod.rs` y `src/vulkan_backend.rs`: `div_ceil` manual → `.div_ceil()` nativo

### 4.3. Skill Injection Pendiente
- `src/mud/skills/logic_math.rs:64`: `// TODO: Inject this exact answer into the inference stream context`
- El resultado de la sandbox matemática se imprime en consola pero no se inyecta en el KV-cache

### 4.4. Action Plan V2.0 Prep
- [ ] Real Attention Execution: reemplazar placeholder con scaled dot-product attention en `src/mud/inference.rs`
- [ ] Zero-Allocation Rayon MoE: buffers pre-asignados thread-safe eliminando vec![] dinámicos
- [ ] Active Skills Intent Trigger: driver asíncrono en `src/main.rs` para `should_activate`/`execute_autonomous_action`
- [ ] Tokenizer Parity Integration: unificar regex y `Ġ` mapping entre tokenizers
- [ ] Vulkan Shader Fusion: fusionar RMSNorm y RoPE en el shader GEMV SPIR-V

### 4.5. Embedding Ternarization (Completado)
- [x] `tools/embed_audit.rs` — Análisis de distribución del embedding
- [x] `tools/embed_ternarize.rs` — Prototipo row-wise absmean + scales
- [x] `docs/MUD_EMBED_TERNARIZATION.md` — Documentación técnica
- [x] `--ternarize-emb` flag en `universal_converter` — ternariza durante conversión, almacena `embed_scales` tensor
- [x] Dequant on-the-fly en `MudInference::embed_token()` — carga `embed_scales`, aplica per-row scale
- [x] Verificación end-to-end: Qwen2.5-0.5B (519 MB → 33 MB embedding), inferencia estable

### 4.6. Auto-Trainer Rewrite (EN REVISIÓN)
Estado: implementado, pendiente de revisión y pruebas extensivas.

| Mejora | Detalle |
|--------|---------|
| Embedding type-aware | Lee Float32 o Ternary2Bit + embed_scales. Sin UB. |
| Escalas de expertos | `dequantize_tensor_f32()` aplica w1/w2/w3.scale |
| Type guards | Match sobre MudTensorType antes de castear punteros |
| Multi-capa FFN | `LAYERS_TO_TRAIN=3` capas consecutivas por token |
| Content-aware layer hash | Embedding hash en vez de `(t_in/16) % num_layers` |
| Gate entrenable | Gate weights en Float32, reciben gradientes |
| Full vocab CE | Para vocabs ≤50k: proyección full + cross-entropy |
| N negativos | `NUM_NEGATIVES=5` para vocabs grandes |
| Persistencia | `save_shadows_to_mud()` con requantización Ternary2Bit + grid-search scales. Atómico (tmp + rename). |

**Pendiente para revisión:**
- [ ] Verificar que la persistencia no corrompe el .mud existente
- [ ] Probar con Qwen (151k vocab, full CE desactivado)
- [ ] Benchmark de velocidad (multi-capa vs single-layer)
- [ ] Prueba de recuperación tras Ctrl+C

---

## 5. Baseline Estadístico y Matemático

### 5.1. Distribución de Pesos Ternarios (Sigma)

| Tensor | Tipo | Sigma | Sparsity |
|--------|------|-------|----------|
| `token_embd.weight` | Float32/Ternary2Bit | 0.131 / 2.014 bits | 0% / 33% zeros |
| `blk.0.attn_q.weight` | Ternary | 0.569 | 67.7% |
| `blk.15.expert.0.w2.weight` | Ternary | 0.680 | 53.7% |
| `blk.29.attn_output.weight` | Ternary | 0.689 | 52.6% |

**Distribución ideal:** 37% +1, 26% 0, 37% -1 (simétrica). Skewness ~0, Kurtosis ~ -1.6.

### 5.2. Estado de Inferencia (Runtime)

| Métrica | Rango Saludable | Observado |
|---------|----------------|-----------|
| LogitVar | < 30.0 | 15–49 (explosiones detectadas) |
| Entropía | < 1.0 | 0.08–6.52 (inestable) |
| X-Move (Delta) | < 30.0 | 41–58 (caótico) |

### 5.3. Conversión Dense → Ternary
- **Problema:** PTQ directo destruye correlaciones entre capas
- **Fix requerido:** QAT con KL-Divergence loss sobre corpus de calibración (`tools/universal_converter/calibration.rs`)
- **Issue adicional:** Tokenizer mismatch — el conversor inyecta tokenizer de `core_skills` (32k vocab) en modelos con vocabularios diferentes (ej. 49k)

---

## 6. Conversión Universal (Dense → .mud)

### Soportado
- Safetensors → `.mud` con parámetros dinámicos (hidden_size, ffn_hidden, kv_dim, num_experts, num_heads, num_kv_heads, head_dim)
- GQA: inferencia dinámica de `kv_dim` → `num_kv_heads = kv_dim / head_dim`. Attention loop con `kv_group = num_heads / num_kv_heads`.
- Dense → MoE: mapeo de MLPs a expert.0

### No Soportado / En Progreso
- [ ] Vocabulary syncing: parsear `tokenizer.json` de HuggingFace en vez de heredar metadatos
- [ ] QAT Distillation: calibración con teacher FP32 y KL-Divergence
- [ ] GQA threading: paralelizar atención con rayon
- [ ] MoE bypass: cuando `num_experts == 1` saltar el router
- [ ] Auto-trainer: `weight_decay` colapsa pesos ternarios a cero (BUG-6 pendiente)

---

## 7. Monitoreo Matemático (Salud del Modelo)

| Métrica | Rango Ideal | Descripción |
|---------|-------------|-------------|
| Media (Expectation) | ~0.0 | Sesgo > 0.1 indica dependencia de un solo valor |
| Sigma (StdDev) | 0.5–0.8 | Bajo → amnesia ternaria; Alto → ruido excesivo |
| Skewness | < 0.5 | Asimetría entre pesos +1 y -1 |
| Kurtosis | < 1.0 | Leptokurtic → cuantización demasiado rígida |
| Autocorrelación (Lag 2) | < 0.1 | > 0.1 indica bucles recurrentes en MoE |
| Traza de Confianza | > 60% | `1.0 - (Entropía Media / Entropía Máxima)` |

---

## Notas de Versión

Los siguientes documentos contienen información histórica ya reflejada aquí:
- `MUD_AUDIT.md` — Auditoría estructural v1 (reemplazado)
- `MUD_AUDIT_REPORT_V1.md` — Reporte de estabilización v1.1 (histórico)
- `MUD_AUDIT_RESOLUTION.md` — Log de resolución de bugs (histórico)
- `MUD_CONVERSION_AUDIT.md` — Auditoría de conversión (reemplazado)
- `MUD_STATISTICAL_AUDIT.md` — Auditoría estadística profunda (reemplazado)
- `MUD_MATHEMATICAL_AUDIT.md` — Auditoría matemática (reemplazado)
- `MUD_ROADMAP.md:81-139` — Bugs críticos auditados (migrados aquí)
