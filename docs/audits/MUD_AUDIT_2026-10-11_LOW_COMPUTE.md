# Auditoría Forge LLM / MUD — 2026-10-11

**Alcance:** `src/**` (62.6k LOC Rust), `tools/**`, `build.rs`, `Cargo.toml`, `mud.sh`, `.github/workflows/ci.yml`, `docs/**` vs código.
**Enfoque:** (1) correctitud y eficiencia de la inferencia de bajo consumo, (2) seguridad de superficies de entrada, (3) deriva documentación↔código, (4) contraste con el estado del arte arXiv 2024-2026.
**Método:** lectura directa de código en los puntos calientes, ejecución real de la suite (`cargo test --lib --release`), ejecución real de inferencia sobre `models/ale.mud`, parseo forense del `.mud` y de los logs de telemetría, más 6 auditorías paralelas read-only (seguridad, correctitud numérica, rendimiento/roofline, entrenamiento, calidad de código, deriva documental).
**Marcas:** `[V]` verificado de primera mano en esta sesión · `[A]` hallado por auditoría paralela (evidencia citada) · `[I]` inferencia.

---

## 0. Resumen ejecutivo

**Veredicto: el proyecto NO está en el estado que declara su documentación.** `cargo test --lib --release` **falla**, el CI **no puede ejecutarse** (workflow inválido), el modelo insignia produce **ensalada de palabras**, y hay **dos bugs matemáticos críticos en el backward** que explican por qué el loss no baja.

| # | Severidad | Hallazgo | Evidencia |
|---|---|---|---|
| 1 | 🔴 CRÍTICO | **La cabeza entrenada no es la cabeza evaluada**: el trainer optimiza filas de `token_embd`; inferencia usa `output.weight`, que **nunca se escribe** (valores idénticos antes/después) ⇒ nada de lo aprendido cambia la salida | `train_step.rs:160`, `inference.rs:395`, `checkpoint.rs` `[V]` |
| 2 | 🔴 CRÍTICO | Deriva de escala PRQ sin punto fijo (`s ← 0.52·s` por guardado) y reescritura del layout de bytes en cada `save` (88 % de bytes distintos, valores idénticos) | `checkpoint.rs:203-208`, forense binaria `[V]` |
| 3 | 🔴 CRÍTICO | El backward del FFN deriva **ReLU²** mientras el forward calcula **SiLU** → gradientes de `ffn_gate`/`ffn_up` erróneos en **cada paso** | `slime_forward.rs:613-637` (SiLU) vs `slime_backward.rs:1109-1170` (ReLU²) `[V]` |
| 4 | 🔴 CRÍTICO | El backward mHC **omite los factores** `alpha` y `(1-gate)·beta` → todo gradiente aguas arriba mal escalado | `slime_forward.rs:479` vs `slime_backward.rs:1055-1066` `[V]` |
| 5 | 🔴 CRÍTICO | `cargo test --lib --release` **FALLA** (1/461): UB en `prefix_sum_batch4_avx2` (store por puntero sobre binding inmutable) — pasa en debug, falla con `-O` | `fenwick.rs:307-312`, test `fenwick.rs:1020-1029` `[V]` |
| 6 | 🔴 CRÍTICO | El workflow de CI es **inválido** (`on:` anidado dentro de un job) → GitHub rechaza el fichero → **CI nunca corre**; el guard usa una ruta local como slug de repo | `.github/workflows/ci.yml:76-88` `[V]` |
| 7 | 🔴 CRÍTICO | El formato ternario en disco/hot-path es **4 bits/peso** (ELUT nibble) = **2.52× la entropía** de 1.585 bits → ~2.5× de tráfico de pesos | `mod.rs:145`, `asm/ternary_gemv.s` `[V]` |
| 8 | 🟠 ALTO | El LM head corre en **f32 denso** (`vfmadd231ps` sobre 77.6 MB/token en `ale.mud`) en vez de ternario (9.7 MB) → ~8× tráfico y cuello de botella dominante | `main.rs:616`, `asm/lm_head.s:15-55` `[V]` |
| 9 | 🟠 ALTO | Inferencia materializa `token_embd` y `output` a f32 al cargar: **155 MB de RSS** para un modelo de 22 MB | `main.rs:226, 292` `[V]` |
| 10 | 🟠 ALTO | **Deadzone inconsistente**: conversor usa `0.7·s`, checkpoint/trainer usa `0.5·s` → los pesos se **re-cuantizan al primer guardado** (deriva silenciosa) | `universal_converter/quantizer.rs:125` vs `checkpoint.rs:211, 683` `[V]` |
| 11 | 🟠 ALTO | Sin validación `metadata ↔ shape`: `.mud` malicioso/desalineado ⇒ OOB read y OOM (release `panic=abort`) | `inference.rs:350-433`, `mod.rs:618-635` `[A]` |
| 12 | 🟠 ALTO | El **parser GGUF no tiene ninguna cota** (counts, longitudes, recursión, offsets) — contradice los SEC-H1/H2 | `gguf/mod.rs:91-190, 264-296` `[A]` |
| 13 | 🟡 MEDIO | `skills/` (16 ficheros, 663 LOC) **nunca se compila** (no declarado en ningún `mod`) | `[V]` |
| 14 | 🟡 MEDIO | 4 round-trips i8↔f32 por capa en el forward, sin beneficio (el GEMV consume f32) → sólo pérdida de precisión | `slime_forward.rs:679-709, 771-832, 1254-1274, 1383-1410` `[V]` |

**Medición real (`models/ale.mud`, hidden=128, 6 capas, vocab 151643):** ~**46 tok/s** (512 tokens en 11.13 s, ~22 ms/token), **9% del ancho de banda** disponible. Salida: `"contributionsexclusive defect Famous operations Sikh Email hello man 323 Cha Hampshire…"` — **sin coherencia**. Para el prompt `"The capital of France is"`, el top-1 es `'ton'` (2.92) y el top-10 son tokens vacíos/sin relación: el modelo no ha aprendido el dato. `[V]`

---

## 1. Baseline reproducible

| Comprobación | Resultado | Nota |
|---|---|---|
| `cargo clippy --all-targets` | **0 warnings** ✅ | dev profile `[V]` |
| `cargo test --lib --release` | **459 passed / 0 failed / 2 ignored** ✅ | UB fenwick purgado (R-03, 2026-10-11) — 2 tests de batch4/batch8 eliminados |
| `cargo test --lib` (debug) | 459 passed ✅ | |
| `./mud.sh ci` | corre `cargo test --lib` (**debug**) + `clippy --lib` | nunca detecta el fallo release; no usa `--all-targets` `[V]` |
| CI remoto | inválido → no ejecuta | `[V]` |
| Repositorio git | **no existe** (`fatal: no es un repositorio git`) | sin historia, sin diff, sin auditoría de cambios `[V]` |

> La documentación afirma "460 tests ✅ | clippy 0 ✅" (GEMINI §0, AGENTS). El número es correcto **sólo en debug**; el release está roto y ninguna automatización lo ve.

---

## 2. Correctitud — el entrenamiento no puede aprender (P0)

### 2.1 CRÍTICO — Forward SiLU vs backward ReLU²

Forward (`slime_forward.rs:613-637`):

```rust
silu_vectorial_avx2(ffn_mid, ws.ffn_gate_f32.as_ptr(), ws.ffn_mid_f32.as_mut_ptr());
// mid[i] *= up[i]     →  mid = silu(gate) * up
```

Backward ternario — la ruta viva para todos los modelos ternarios (`slime_backward.rs:1109-1170`):

```rust
// Forward: ffn_mid_f32[i] = (if g > 0.0 { g * g } else { 0.0 }) * up[i]   ← comentario FALSO
let g_sq = _mm256_mul_ps(g, g);
let up_contrib   = _mm256_mul_ps(gy, g_sq);                       // grad_up   = gy·g²
let gate_contrib = _mm256_mul_ps(gy, _mm256_mul_ps(two, _mm256_mul_ps(g, u))); // = gy·2gu
let mask = _mm256_cmp_ps(g, zero, _CMP_GT_OQ);                    // relu' (g>0)
```

`relu2_backward_avx2` (`ezop.rs:358-390`) hace lo mismo en la ruta F32 (`slime_backward.rs:687`).

Correcto para `mid = silu(g)·u`:

* `∂mid/∂u = silu(g)` → `grad_up = gy·silu(g)` (el código usa `g²`)
* `∂mid/∂g = u·silu'(g)`, con `silu'(g) = σ(g)·(1 + g·(1-σ(g)))` (el código usa `2gu`)

**Impacto:** todos los gradientes de `ffn_up.weight` / `ffn_gate.weight` son incorrectos en todo paso, para ambos formatos de peso, sin NaN ni clamp que lo delate. El comentario del tape (`slime_backward.rs:245`) y AGENTS.md:85 (`SiLU`) ya se contradicen entre sí.

**Falsación (5 min):** finite-difference de `L = ½Σ(silu(g_i)·u_i)²` respecto a `g` y `u` contra `relu2_backward_avx2`; difieren para todo `g < 1`.

### 2.2 CRÍTICO — Backward mHC sin regla de la cadena

Forward (`slime_forward.rs:479`): `val = alpha·h_in + (1-gate)·beta·f_h`
Backward (`slime_backward.rs:1055-1066`):

```rust
let g = *gi.add(i);
*gb.add(i) = g + kinetic_grad;   // grad_branch (f_h)  = grad_in·1        ← falta (1-gate)·beta
*go.add(i) = g;                  // grad_out (h_in)    = grad_in·1        ← falta alpha
```

Los factores `alpha[i]` y `(1-gate[i])·beta[i]` existen y son entrenables (clamp [0,4]); con `gate = σ(v_jepa) ≈ 0.5` en equilibrio el error en la rama es ~0.5×. Como `grad_branch` siembra todo el backward del FFN y de Q/K/V, **todos los pesos aguas arriba quedan mal escalados**. `mhc_backward_vec` (gradiente de alpha/beta) sí es correcto — sólo la propagación está mal. `models/ale.mud` contiene `blk.N.mhc_alpha/beta/radius` (18 tensores) → aplica al modelo real. `[V]`

### 2.3 CRÍTICO — UB en el kernel "Sprint 4b" y suite release rota

```rust
let result_arr = [0.0f32; 8];                                  // binding inmutable
_mm256_storeu_ps(result_arr.as_ptr() as *mut f32, new_sums);   // escribe por puntero derivado de &T
sum0 = result_arr[0];
```

Escribir a través de un `*mut` derivado de una referencia compartida/`let` inmutable es **UB**. Con `-O` LLVM pliega `result_arr = [0.0; 8]` y el kernel devuelve ceros: el test pasa en debug y falla en release. `prefix_sum_batch4_avx2` / `batch8_avx2` **no tienen ningún llamador de producción** (grep global) → es a la vez UB y código muerto que rompe el CI (P-08/P-06).

### 2.4 Deriva de cuantización al guardar (deadzone 0.7 vs 0.5)

| Ruta | Umbral | Fichero |
|---|---|---|
| Conversión (ternariza y empaqueta ELUT) | `delta = 0.7·s` | `tools/universal_converter/quantizer.rs:125` |
| Checkpoint / trainer (`pack_ternary_into`) | `delta = 0.5·s` | `src/mud/checkpoint.rs:211, 683` |

Todo peso con `0.5s < |w| ≤ 0.7s` cambia de `0` a `±1` en el primer `sync_shadow_to_mud`, **sin relación con el aprendizaje**. Esto produce "el MD5 cambia pero el loss no mejora" y explica la necesidad del `MUD_TRAIN_WCLAMP_K`.

### 2.5 Otros hallazgos de correctitud `[A]`

* `chunked_muon_step`: stride de fila incorrecto en el último chunk de columnas → chunks corruptos.
* Ruta `scales_only` del checkpoint: escribe escalas PRQ **√2 veces mayores**.
* `MUD_KV_DTYPE=bf16`: reserva buffers F32 y despacha a un buffer f16 vacío → panic.
* `rms_norm_scale_asm` (usado sólo por `q4_0_gemv_fused`, que a su vez sólo lo usan tests) omite el factor `1/rms` en una de sus dos variantes.
* `unpack_ternary2bit_to_f32` en la ruta **U8/I2_S** del conversor: orden de bits **LSB-first** y los 4 valores de cada byte se reparten entre **4 filas distintas**; la especificación oficial de Microsoft (`BitNet/docs/bitnet-embeddings-i2s-guide.md`) usa **MSB-first** (`byte = c0<<6 | c1<<4 | c2<<2 | c3`) y 4 valores **consecutivos de la misma fila**, y el propio `docs/architecture/BITNET_SPEC.md` de este repo dice `01→+1, 10→-1, 00→0` mientras el código usa `00→-1, 01→0, 10→+1`. `tools/conversion_verifier.rs:80-105` **reimplementa la misma decodificación**, por lo que su SQNR no puede detectar el desajuste (test auto-consistente, no contra referencia). **No existe ningún test de paridad de logits/perplejidad contra el modelo original.** `[A]`+`[V]` (verificado el código y la spec; la severidad real depende del productor de los safetensors usados).

### 2.6 El entrenamiento no puede mejorar el modelo: la cabeza entrenada no es la cabeza evaluada

Forense binaria de `models/ale.mud` vs `models/weights/checkpoints/ale_best.mud` (mismo `.mud`, tras una sesión completa de entrenamiento) `[V]`:

| Tensor | bytes que difieren | valores de-cuantizados |
|---|---|---|
| `output.weight` (LM head usado en inferencia) | 88.34 % | **idénticos** (`mean|Δ| = 0.00000`) |
| `output.prq_scale` | 0 % (idéntico) | — |
| `token_embd.weight` | 88.35 % | **Δ medio 0.261 ≈ 40.7 % de `mean|x|`** |
| `token_embd.prq_scale` (media) | 100 % | 0.740 → **0.414 (×0.56)** |
| `blk.0.attn_q.prq_scale` (media) | 100 % | 0.743 → **0.388 (×0.52)** |
| `output_norm.weight` | 95.7 % | — (mean sigue 1.0000) |

Lectura:

1. **La cabeza que el trainer optimiza no es la que se evalúa.** `train_step.rs:160` construye las *class embeddings* desde `token_embd.weight`; `inference.rs:395`/`main.rs:288` usan `output.weight` cuando existe (existe en `ale.mud`). `checkpoint.rs` **nunca escribe `output.weight`** (sólo `token_embd.*`, tensores de capa, mHC y `output_norm`). Consecuencia: `output.weight` es **semánticamente idéntico** antes y después del entrenamiento → *nada de lo que aprenda el trainer puede cambiar la distribución de salida del modelo*. `[V]`
2. **Deriva de escala sin punto fijo.** La escala PRQ se recalcula en cada guardado como `absmean(shadow)·1/√2` sobre una shadow ya escalada ⇒ `s_new ≈ 0.52·s_old`, geométricamente, para siempre. Se observa exactamente ese factor (0.52 / 0.56) y una segunda aplicación en `fresh_fixed2_latest` (0.33-0.41). El "aprendizaje" registrado en los artefactos es, en su mayor parte, **re-escalado uniforme**: `mean|Δ|` es ~0.26 en *todos* los tensores, incluido el embedding (que el trainer apenas toca). `[V]`
3. **El empaquetado ternario se reescribe con otro layout en cada `save`**: el 88 % de los *bytes* cambia mientras los *valores decodificados* son idénticos (`output.weight`). Por tanto `hash_trained_weights`, el panel `[DW]` y cualquier métrica a nivel de byte **no son fiables** para detectar cambios reales. `[V]`
4. **Las normas son identidad** (`attn_norm`, `ffn_norm` con `mean = 1.0000`, `output_norm` `all_one=true` en el base): no hay escalado aprendido por capa. `[V]`
5. **El loss no baja** en ninguna sesión registrada (§5.3) y los logs muestran `loss` por encima de `ln(64) = 4.159` (uniforme del sampled softmax con 63 negativos) — firma de función de score rota, no de corpus difícil `[A]`.
6. **El deadzone STE es absoluto** en la práctica: con `|w| ≈ s ≈ 0.74` haría falta `ΔW ≈ 0.47` para voltear un código, pero el presupuesto por sesión es ~`40 chunks × 5e-4 ≈ 0.02` (~24× corto) `[A]`.
7. `[A]` El backward omite el `scale_up` que el forward aplica (`train_step.rs:654` vs `:665`), el clip de gradiente nunca se activa (`grad_norm` observado 0.08-0.27 « 1.0) y sólo afecta al gradiente de activación, y Adam usa L2 acoplado (no decoupled) dentro del gradiente.

---

## 3. Eficiencia de inferencia (el núcleo de "bajo consumo")

### 3.1 Formato de pesos: 4 bits/peso, no 1.58

```
MudTensorType::Ternary2Bit => elements.div_ceil(8) * 4      // mod.rs:145
```
8 pesos por `u32` = **4 bits/peso**. El alfabeto ternario tiene entropía `log2(3) = 1.585` bits; los formatos reales de 1.58 bits empaquetan **5 trits por byte (1.6 bits)**: `TQ1_0`/`PTQ1_0` de llama.cpp miden **1.67-1.69 bpw** (arXiv:2502.11880 Fig.1; PR ggml-org/llama.cpp#29077).

| Modelo | 4 bits (actual) | 1.6 bits (posible) | factor |
|---|---|---|---|
| `ale.mud` ternario (40.3 M pesos) | 20.1 MB | 8.0 MB | 2.5× |
| clase 2B (objetivo del producto) | ~1.0 GB/token | ~0.4 GB/token | 2.5× |

En decode batch=1 el tiempo es ∝ bytes de pesos ⇒ **~2.5× de throughput regalado**, más 2 bits/peso de estado (los códigos 0x1/0xF son sólo 2 de 16). La literatura específica del "cuarto estado" (`arXiv:2508.05905`, Signed-Zero Ternary) usa el código libre para mejorar la cuantización.

### 3.2 El LM head es f32 denso (cuello de botella dominante)

```rust
pub fn lm_head_logits_avx2(vocab_size, hidden, regs: *const f32, weights: *const f32, out: *mut f32)
```
`asm/lm_head.s` hace `vfmadd231ps` fila a fila (stride `hidden*4`) y `main.rs:616` le pasa `out_slice_ternary`, que es `out_f32_owned` — **el tensor ya de-cuantizado a f32** (`main.rs:289-397`).

| | bytes/token (`ale.mud`) | a ~30 GB/s |
|---|---|---|
| LM head f32 (actual) | 151643×128×4 = **77.6 MB** | ~2.6 ms |
| LM head ternario 4-bit | 9.7 MB | ~0.32 ms |
| LM head ternario 1.6-bit | 3.9 MB | ~0.13 ms |

El head es ≈99% del tráfico de pesos del modelo completo (`ale.mud`: 0.75 MB ternarios de capas + 77.6 MB head). Además está **single-threaded** (`[A]` PerfRoofline): no usa `PCorePool`, y con `hidden=128` el bucle interno (8 iteraciones de 16) está dominado por el overhead por fila (151643 filas).

### 3.3 Inferencia materializa embeddings/head a f32

`main.rs:226` (`emb_f32 = vec![0.0; vocab*hidden]`) y `main.rs:292` (`out_f32_owned`) ⇒ **155 MB de RSS** para un `.mud` de 22 MB (7× el tamaño del modelo), cuando la fila de embedding se puede de-cuantizar on-the-fly (128 valores) con `dequantize_ternary_row`.

### 3.4 Precisión: el estado oculto se almacena en f16 y las activaciones se cuantizan a i8 4×/capa

* `SlimeRegister` guarda `(re, im)` en f16 (`slime.rs:33-77`): **cada capa redondea el residual a f16** (ε≈5e-4 relativo, subnormales <6e-5 → 0, y >65504 → Inf). El "guard" de `slime_forward.rs:521` comprueba finitud **en f32 antes** de la conversión, por lo que no protege del rango f16. `[V]`
* 4 round-trips `f32 → i8 → f32` por capa (`slime_forward.rs:679-709`, `771-832`, `1254-1274`, `1383-1410`): cuantización a i8 con **pico global** del vector (todo valor `< peak/127` → 0), y luego de-cuantización para alimentar un GEMV que consume f32. Es el resto del diseño W2A8 retirado: hoy **no aporta cómputo, sólo error**. `[V]`

### 3.5 Roofline medido

| Métrica | Valor |
|---|---|
| Throughput real `ale.mud` | **~46 tok/s** (22 ms/token) |
| Tráfico de pesos/token | ~78.4 MB |
| Ancho de banda efectivo | **~3.6 GB/s** (≈9% de lo disponible en i7-1260P) |
| Paralelismo | `user 68s / real 11.1s` ≈ 6.1× (pool de 8 hilos, HT de P-cores) |
| Hilos | `PCORE_THREADS_CAP = 8` (`constants.rs:82`); 8 E-cores **ociosos**; slots del pool en la misma línea de 64 B (false sharing) `[A]` |
| Fases seriales | RMSNorm/atención/softmax/JEPA/mHC corren en el hilo principal mientras 8 workers giran `[A]` |
| Eficiencia vs roofline | ~10-30% (147M) y ~2-5% (2B) `[A]` |

### 3.6 Parches de sampling: síntoma de un modelo degenerado

`main.rs:634-680` aplica, en este orden: eliminación de **DC bias** (restar la media), **re-escalado termodinámico** (`boost = clamp(8/peak, 0.5, 16)`, con umbral mágico `0.05`), **penalización de repetición** con ventana 64, y `logit_scale = 1/√hidden` aplicado a los pesos del head al cargar (`main.rs:371-375`). La cadena "divido por √H al cargar y multiplico por un boost arbitrario al muestrear" es calibración compensada con parches. Evidencia de degeneración: para `"The capital of France is"`, `LOGITS min=-3.54 max=2.92 std=0.64` y top-10 = `ton, '', '', '', '', ' Claims', ' documents', ' Shock', '', ' turnovers'`. `[V]`

### 3.7 Otros `[A]`

* `ternary_gemv.s` `.leftover8` procesa un bloque completo de 8 aunque queden 1-7 elementos (lee hasta 7 f32 y 4 B de pesos de más); `row_u32s = n_in/8` (división entera) desalinea filas si `n_in % 8 ≠ 0` (`gemv_gpu.rs:263`). Latente hoy (todas las dims son múltiplos de 8).
* `PCorePool`: spin-wait + slots contiguos (`AtomicPtr` en la misma línea).
* Conversión de dims a `u32` en el backend Vulkan sin comprobación (63 casts en `ash_backend.rs`).

---

## 4. Seguridad `[A]`

| ID | Sev | Hallazgo |
|---|---|---|
| SEC-MUD-01 | Crítico | Metadata (`hidden_size`, `vocab_size`, …) parseada sin cota y usada como longitud de `from_raw_parts` sobre el mmap, sin cruzarla con `tensor.shape` ni con la extensión validada ⇒ OOB read / SIGSEGV / OOM desde el CLI (`inference.rs:350-433`, `main.rs:229-323`) |
| SEC-MUD-02 | Alto | `shape.iter().product()` y `ptr_offset + expected_bytes` sin `checked_*` ⇒ overflow que **salta** el guard del loader (`mod.rs:618-635`, release sin overflow-checks) |
| SEC-GGUF-01 | Alto | Parser GGUF sin cotas: longitudes, counts, `Vec::with_capacity(n)` desde el fichero, recursión ilimitada en arrays, `data_ptr` sin validar |
| SEC-MUD-03 | Medio | `ecc_verify_all` lee la paridad con el conteo del tensor base, no el de la ECC |
| SEC-MUD-04 | Medio | `worm.rs:282-293`: `from_raw_parts` f32 sobre un tensor que puede ser ternario, longitud del llamador |
| SEC-WS-01 | Bajo | `workspace_agent.rs:92-134`: symlink colgante permite escribir fuera del root (latente, sin llamadores) |
| SEC-SKILL-01 | Bajo | `skills/data_analysis.rs:34-52`: lectura de cualquier `*.csv` a partir de texto generado |
| SEC-DOC-01 | Info | `SECURITY.md:11-12` ("valida shapes", "todo `unsafe` tiene SAFETY") y AGENTS.md:8 ("0 RCE + 0 OOB") son **falsos**: 19 `unsafe` sin `// SAFETY:` en código no-test (mod.rs, ash_backend.rs, kv_compressed.rs, ecc.rs, worm.rs); las cotas SEC-H1 limitan **counts/rank/longitud**, no la magnitud de las dims |

**0 RCE**: cierto en builds por defecto, pero es **feature-gated, no estructural**: `--features unsafe-cli`/`unsafe-rce` reintroducen `rustc`/`bash -c` por diseño (`rlvr.rs:9-45`, `sandbox.rs:36-70`).

---

## 5. Deriva documentación ↔ código y deuda estructural

### 5.1 Afirmaciones falsas o engañosas (verificadas)

| Afirmación | Realidad |
|---|---|
| "460 tests ✅" | release falla (UB fenwick) `[V]` |
| "Sprint 1: 0 RCE + 0 OOB" | 0 OOB falso (SEC-MUD-01/02/05, GGUF) `[A]` |
| "MUD_GPU_GEMV default `auto` en mud.sh" | `mud.sh:48` lo fuerza a `0` `[A]` |
| "Rayon removed from runtime crate" | `rayon = "1.12.0"` sigue en `[dependencies]` de `Cargo.toml:44` `[A]` |
| "Wormhole Network peer-to-peer… reduce el cómputo del host hasta 50%" | `worm.rs:762-810`: ambas ramas ejecutan **el mismo trabajo** con un multiplicador `strength`; no hay red, ni sockets, ni peers `[V]` |
| "1.58-bit" en toda la documentación | el wire format es **4 bits/peso** `[V]` |
| "CellV5 Fase 2/3 COMPLETADO" | ya corregido en AGENTS (D-3/D-5); el código real es fallback Delta+LZ4 `[V]` |
| "Fase 1.1 Fenwick batch prefix sum COMPLETADO (~4×)" | kernel UB, sin llamadores, rompe la suite release `[V]` |
| `MUD_MHC_SYNC`, `MUD_DW_DEBUG`, `MUD_EPOCHS`, `MUD_BATCH_SIZE`, `MUD_MERGE_ON_SAVE`, `MUD_DEBATE_SEEDS`, `MUD_DSPARK*`, `MUD_FENWICK_AVX2/BATCH`, `MUD_TOKENIZER_DIR` | documentados y **ausentes del código** (9 vars) `[A]` |
| 14 env vars presentes en código y sin documentar | `MUD_AWAKE_ITERS`, `MUD_LOSS_CERT_*`, `MUD_TELEM_EVERY`, `MUD_TRAIN_AOT_TOKENS`, … `[A]` |

### 5.2 Deuda estructural (P-08 / P-18 / P-09)

* `src/mud/skills/` — **16 ficheros / 663 LOC nunca compilados** (no hay `mod skills` en ningún sitio) `[A]`.
* `mud.sh` invoca 3 bins inexistentes en `Cargo.toml` (`converter_auditor`, `variance_inspector`, `pointer_audit`) → comandos `audit-conv`, `inspect-var`, `pointer-audit` **rotos** `[A]`.
* `session.rs` (1933 LOC, bucle de entrenamiento) **sin un solo test**; `gguf/mod.rs` con 4 tests sin aserciones; `tests/test_tokenizer.rs` es un `fn main` (no un test) con modelo inexistente y `unwrap`s `[A]`.
* La feature `proptest` (fuzzing SEC-H2) **nunca se activa** en CI ni en `mud.sh` → el único fuzz de loaders es letra muerta `[A]`.
* `q4_0_gemv.s` + wrappers: muertos fuera de tests `[A]`. `ezop.rs`: ~22 cuerpos `cfg(not(target_arch))` compilados fuera, con riesgo de deriva `[A]`.
* `build.rs`: compara **mtime** para los shaders (checkout limpio → no recompila) y `elementwise.spv` se embebe sin `rerun-if-changed` `[A]`.
* Root con deuda P-18: `arreglar_trainer.md` (39 KB), `informe.md`, `corpus.txt`, `test_tok.rs`, 15 `mud_train_metrics.log.bak.*`, `scratch_sessions_summary.json` (579 KB) `[V]`.
* 2 stacks de tokenizer (`model/tokenizer.rs` vs `mud/smart_tokenizer.rs`) y `SmartTokenizer` inventa defaults (`hidden=256`, `vocab=32000`) — violación P-13 `[A]`.

### 5.3 Telemetría

* `[TELEM]` real: 18 chunks en la sesión actual; loss **5.4795 → 5.4129** (oscila ±0.1, sin tendencia); sesión anterior (`.bak` de 138 KB): **6.62 → 6.70** (plano) `[V]`. Coherente con §2.1/§2.2: no hay aprendizaje.
* Paneles/charts que nunca reciben datos (`gate_mean`, `stp_λ_eff`, `loss_cv`), métricas C-MUD hardcodeadas a 0 en el emisor `[A]` (hallazgos previos del 2026-08-03, no corregidos).

---

## 6. Estado del arte (arXiv) aplicable

### 6.1 Cuantización / kernels ternarios

| Paper | Aporte | Relevancia para MUD |
|---|---|---|
| **bitnet.cpp** arXiv:2502.11880 / 2410.16144 | kernels mpGEMM **sin de-cuantizar**, I2_S 2-bit, lossless; 1.37-5.07× en ARM | El head y las capas MUD de-cuantizan a f32 antes del GEMV ⇒ justo lo que este paper elimina |
| **T-MAC** arXiv:2407.00088 | mpGEMM por **LUT** (bit-wise table lookup), sin multiplicaciones; hasta **4× throughput y -70% energía** vs llama.cpp; BitNet-3B: 30 tok/s 1 core, 71 tok/s 8 cores (M2-Ultra), 11 tok/s RPi5 | Alternativa directa al bit-twiddling `vpsrlvd/vpcmpeqd/vpand` actual; escala lineal con el bit-width |
| **FairyFuse** arXiv:2604.20913 | kernels ternarios fusionados **sin multiplicaciones** (masked add/sub), AVX-512; **29.6× kernel speedup**, 32.4 tok/s en Xeon, near-lossless (ppl 5.52 vs 5.47) | MUD ya hace add/sub en `ternary_gemv.s` (bien); el paper cuantifica el techo: la compresión 16× mueve el GEMV de memory-bound a compute-bound |
| **BitNet b1.58 2B4T** arXiv:2504.12285, BitNet JMLR 2025 | Receta nativa 1.58-bit; embeddings en mayor precisión | `BITNET_SPEC.md:29` del propio repo lo dice; `ale.mud` ternariza `token_embd` y `output` |
| **PT²-LLM** arXiv:2510.03267 (ICLR'26) | Ternarización **post-training** con optimización de parámetros y compensación de error | El conversor MUD usa absmean por fila + umbral fijo, sin rotación ni compensación |
| **Tequila** arXiv:2509.23809, **CAT-Q** arXiv:2606.26650 | Ternary PTQ sin "trapping" / coste-eficiente | Ataca exactamente la deadzone/STE que el repo sufre |
| **ITQ3_S / TurboQuant** arXiv:2603.27914, **QuIP#** arXiv:2402.04396 | Rotación (FWHT/Hadamard) para suavizar outliers antes de cuantizar | Ausente por completo en MUD; es la mejora PTQ más rentable |
| **Signed-Zero Ternary** arXiv:2508.05905 | Usa el 4º estado del código de 2 bits | MUD desperdicia 14 de 16 códigos del nibble |
| **TernaryLM** arXiv:2602.07374 | Transformer 132M nativo ternario con escalado adaptativo por capa | Comparable directo al objetivo de MUD |
| **Vec-LUT** arXiv:2512.06443, **LUT Tensor Core** arXiv:2408.06003 | LUT vectorial para ≤1.58-bit en edge | Camino para CPU sin AVX-512 |
| **ParetoQ** arXiv:2502.02631 | Leyes de escala: 2-3 bits ≥ 4 bits en tamaño-precisión | Justifica 2-bit como objetivo |

### 6.2 KV cache y contexto largo

| Paper | Aporte |
|---|---|
| **KV Cache Compression Review** arXiv:2508.06297 | Taxonomía (eviction/quant/low-rank/architecture) |
| **MiniKV** arXiv:2411.18077, **KVQuant** NeurIPS'24 | 2-bit KV con co-diseño |
| **H2O / SnapKV / Ada-KV / DuoAttention** (ICLR'25) | Eviction y cabezas retrieval/streaming |
| **RetrievalAttention** NeurIPS'25 | KV por recuperación vectorial |

MUD ya tiene `kv_dtype` (f32/f16/bf16), `kv_compressed` (Delta+LZ4) y `csa_indexer` (top-k + LSH) — alineado con la familia; falta KV sub-8-bit (MiniKV) y cuantización **por cabeza** (`[A]` PerfRoofline: el KV se guarda por elemento).

### 6.3 Decodificación y arquitectura

| Paper | Aporte | Estado en MUD |
|---|---|---|
| **Speculative Decoding Survey** arXiv:2401.07851, **LayerSkip** arXiv:2404.16710, **Draft&Verify** ACL'24 | Self-speculative por salto de capas | `speculative.rs` (`SlimeDrafter`) existe; sin evidencia de aceleración medida |
| **MoESD** NeurIPS'25 | SD sobre MoE sparse | — |
| **LLM-JEPA** arXiv:2509.14252 | Objetivos JEPA en espacio de embeddings para LLM | Valida la idea del tracker JEPA de MUD (que aquí es determinista, sin pesos) |
| **mHC: Manifold-Constrained Hyper-Connections** arXiv:2512.24880 | Proyección del residual a una variedad; base del `mHC` de MUD | El nombre y la idea están respaldados; **la implementación del backward no** (§2.2) |
| **FrugalGPT** arXiv:2305.05176, **RouteLLM** arXiv:2406.18665, **UCCI** arXiv:2605.18796 | Cascadas/routing coste-óptimo | Es el análogo académico de `worm.rs` (WORM); MUD no tiene router aprendido ni medición |
| **LLMLingua-2 / LongLLMLingua** arXiv:2403.12968, 2310.06839 | Compresión de prompt | Alternativa barata a contextos largos |
| **Energy**: T-MAC (-70%), arXiv:2511.11624, 2606.23001 (EnerInfer), 2511.05597 | Medición de energía por token en edge | MUD **no mide energía** en absoluto |

### 6.4 Gap analysis (qué adoptar, por orden de retorno)

1. **Empaquetado ternario real (5 trits/byte, 1.6 bpw)** → ~2.5× de throughput y 2.5× menos disco/RSS. Es un cambio de formato + kernels, no de modelo.
2. **LM head ternario, batched (4/8 filas) y paralelizado** → elimina el 99% del tráfico de pesos del decode y el único cuello single-thread.
3. **Eliminar la de-cuantización a f32 de `token_embd`/`output` en carga** (fila on-the-fly) → -155 MB de RSS.
4. **mpGEMM sin de-cuantización** (bitnet.cpp/T-MAC): el GEMV de capas ya es ternario, pero las **activaciones** hacen 4 round-trips i8 por capa; eliminarlos o pasar a i8 real en el kernel.
5. **Rotación (FWHT/Hadamard) previa a ternarizar** (QuIP#/ITQ3_S) + **PTQ con compensación** (PT²-LLM/Tequila) → calidad a igual bit-width.
6. **Mantener embeddings/head/norms en mayor precisión** (BitNet b1.58 §, propio `BITNET_SPEC.md:29`).
7. **Test de paridad contra referencia** (logits/perplejidad del modelo original) — hoy no existe.
8. **KV sub-8-bit** (MiniKV) y cuantización por cabeza.
9. **Medición de energía por token** para sostener el discurso de "bajo consumo".

---

## 7. Plan de acción priorizado

| P | Acción | Ficheros | Criterio de aceptación |
|---|---|---|---|
| P0 | Corregir el backward del FFN a SiLU (kernel + escalar) | `slime_backward.rs:1109-1170`, `ezop.rs:358` | finite-diff de `silu(g)·u` verde; loss de `loss_cert` cae monótono |
| P0 | Restaurar factores mHC (`alpha`, `(1-gate)·beta`) en `grad_out`/`grad_branch` | `slime_backward.rs:1055-1066` (y ruta F32 :635-641) | finite-diff de la rama verde con alpha/beta≠1 |
| P0 | Arreglar UB de `prefix_sum_batch4_avx2` (o borrarlo, P-08) y añadir `cargo test --lib --release` a CI | `fenwick.rs:307` | `cargo test --lib --release` verde |
| P0 | Reparar `.github/workflows/ci.yml` (`on:` a raíz; guard con slug real) | `ci.yml:76-88` | workflow válido (`actionlint`) |
| P1 | Unificar deadzone (una constante, p. ej. `TERNARY_DEADZONE`) conversor+trainer | `quantizer.rs:125`, `checkpoint.rs:211,683` | test de idempotencia: convertir→guardar→hash idéntico |
| P1 | LM head ternario batched + `PCorePool`; no materializar `output` a f32 | `main.rs:289-397, 616`, `asm/lm_head.s` | ≥3× tok/s en `ale.mud`; RSS < 40 MB |
| P1 | Validación `metadata ↔ shape` con `checked_mul` en el loader; cotas de magnitud | `mod.rs:618-635`, `error.rs:36` | proptest con header válido pero dims incoherentes → `Err`, no panic/OOB |
| P1 | Cotas + bounds en el parser GGUF (reusar SEC-H1/H2) | `gguf/mod.rs:91-296` | mismo proptest para GGUF |
| P1 | Test de paridad de conversión (logits/perplejidad vs referencia) | `tools/conversion_verifier.rs` | gate SQNR **contra referencia**, no auto-consistente |
| P2 | Formato 5 trits/byte (base-3) + kernels | `mod.rs:145`, `asm/ternary_gemv*.s` | 2.5× menos bytes/token, equivalencia bit-exacta con el decode actual |
| P2 | Eliminar round-trips i8 (o pasar a i8 real en el kernel) | `slime_forward.rs:679-709…` | paridad numérica y menos pases |
| P2 | Rotación previa a ternarizar + PTQ con compensación | `tools/universal_converter/quantizer.rs` | mejora de ppl a igual bit-width |
| P2 | Revisar precisión del residual (f16 por capa) | `slime.rs:33-77` | ablación: f32 vs f16 y su efecto en ppl |
| P2 | Limpieza P-08/P-18: `skills/`, bins fantasma en `mud.sh`, `q4_0_*`, root docs, `Rayon` en `Cargo.toml` | varios | `clippy --all-targets -D warnings`, `cargo build --all-targets` |
| P2 | Quitar los parches de sampling (DC bias, boost) tras arreglar la calibración | `main.rs:634-680` | salida coherente con `top_p` estándar |

---

## 8. Reproducibilidad

```bash
# baseline
cargo clippy --all-targets                       # 0 warnings
cargo test --lib --release                       # ❌ 1 fallo (fenwick UB)
cargo test --lib                                 # ✅ (debug, enmascara el UB)

# inferencia medida
cargo build --release --bin forge_llm
printf 'hello\nquit\n' | MUD_INFER_MAX_TOKENS=512 ./target/release/forge_llm models/ale.mud
printf 'The capital of France is\nquit\n' | MUD_DEBUG=1 MUD_INFER_MAX_TOKENS=8 \
  ./target/release/forge_llm models/ale.mud | grep -E 'LOGITS|TOP-10'

# estructura del modelo (densidad de bits real)
python3 - <<'PY'
import struct
d=open('models/ale.mud','rb').read(); off=4
u32=lambda o: struct.unpack_from('<I',d,o)[0]; u64=lambda o: struct.unpack_from('<Q',d,o)[0]
n=u32(off); off+=4
for _ in range(n):
    kl=u32(off); off+=4+kl; vl=u32(off); off+=4+vl
t=u32(off); off+=4; tern=0
for _ in range(t):
    nl=u32(off); off+=4; off+=nl; tt=u32(off); off+=4; sl=u32(off); off+=4
    sh=[u64(off+8*i) for i in range(sl)]; off+=8*sl+8
    e=1
    for s in sh: e*=s
    if tt==0: tern+=e
print(f"ternary elems={tern}  4bit={tern*4/8/1e6:.1f}MB  1.585bit={tern*1.585/8/1e6:.1f}MB")
PY
```

---

*Auditoría realizada el 2026-10-11. 6 auditorías paralelas read-only + verificación directa. Los hallazgos `[A]` citan `file:line`; los `[V]` fueron reproducidos en esta sesión.*
