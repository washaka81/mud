# MUD Audit Report V13 — Opportunidades de Mejora y Corrección

**Fecha**: 2026-06-10  
**Auditor**: mud-expert agent  
**Scope**: 195 Rust source files, 8 GLSL compute shaders, 11 x86_64 ASM files, 1 autograd sub-crate

---

## Resumen

| Severidad | Count | Categoría |
|-----------|-------|-----------|
| CRÍTICO   | 6     | UB, buffer overflows, corrupción silenciosa |
| ALTO      | 5     | Allocaciones en hot-path (zero-alloc policy violations) |
| MODERADO  | 8     | Estabilidad numérica, dead code, arquitectura |
| BAJO      | 5     | Naming, shader stubs, portabilidad |

**Total: 24 issues identificados**

---

## PRIORITY 1 — CRÍTICO (Correctness / Undefined Behavior)

### 1.1 Alignment UB en ECC module (`as_u32_slice_le` / `as_u32_slice_le_mut`)
- **File**: `src/mud/ecc.rs:103-119`
- **Problem**: `from_raw_parts(buf.as_ptr() as *const u32, ...)` requiere alineación 4-byte, pero `Vec<u8>` solo garantiza 1-byte. El `assert!` verifica divisibilidad de longitud pero no alineación. Esto es **undefined behavior** en Rust safe y unsafe cuando el puntero está desalineado.
- **Fix**: Usar `bytemuck::try_cast_slice` o copiar a buffer `[u32]` alineado. Alternativamente, cambiar `owned_data` de `Vec<u8>` a `Vec<u32>` para tensores ternarios.

### 1.2 `gemv_cpu` usa división entera (round-down) en vez de `div_ceil`
- **File**: `src/vulkan/vulkan_backend.rs:45-46`
- **Problem**: `let blocks_per_row = n_in / 16;` — si `n_in` no es múltiplo de 16, el último bloque parcial se saltea silenciosamente, produciendo resultados GEMV incorrectos. El path principal en `forward.rs:1442` usa correctamente `n_in.div_ceil(16)`, creando una inconsistencia.
- **Fix**: Cambiar a `n_in.div_ceil(16)`.

### 1.3 Vulkan buffer length mismatch entre `vulkan_backend.rs` y `vulkan/mod.rs`
- **File**: `src/vulkan/vulkan_backend.rs:232` vs `src/vulkan/mod.rs:450`
- **Problem**: `vulkan_backend.rs` calcula `w_len = (n_in.div_ceil(16)) * n_out` mientras `mod.rs` calcula `w_len = (n_in / 16) * n_out`. Si `n_in` no es múltiplo de 16, el buffer cached en `mod.rs` es demasiado chico, y `from_raw_parts(packed_w, w_len)` lee más allá de la asignación. Esto es UB.
- **Fix**: Unificar a `n_in.div_ceil(16)` en ambos archivos.

### 1.4 KV-scales buffer dimensionado con `hidden_size / 64` hardcodeado
- **File**: `src/mud/inference.rs:688-701`
- **Problem**: `kv_scales_k` y `kv_scales_v` se alocan como `num_layers * max_pos * (hidden_size / 64)`. Esto asume `num_kv_heads == hidden_size / 64`. Si `num_kv_heads` difiere (común en modelos GQA), el buffer es demasiado chico (out-of-bounds writes en `forward.rs:211-214`) o sobredimensionado (desperdicio de memoria). En línea 211, `scale_offset = l * KV_CACHE_MAX_POS * nkv + max_pos * nkv` — si `nkv` excede `hidden_size/64`, escribe fuera de la asignación.
- **Fix**: Usar `num_kv_heads` en vez de `hidden_size / 64` para la asignación del buffer de scales.

### 1.5 Shader `shadow_optimizer.comp` nunca escribe a bindings de output
- **File**: `assets/shaders/shadow_optimizer.comp`
- **Problem**: El shader declara bindings para `PrqScales` (binding 2) y `OutputTernary` (binding 3), pero `main()` solo actualiza `shadow_w[idx]`. Los PRQ scales nunca se recalculan y el packed ternary output nunca se re-cuantiza. QAT training en GPU falla silenciosamente al no producir pesos ternarios actualizados.
- **Fix**: Agregar lógica de re-cuantización para escribir `packed_w` y actualizar `scales`.

### 1.6 Hash routing — loop infinito potencial
- **File**: `src/mud/routing.rs:236-248`
- **Problem**: El loop `while results.len() < self.max_k` intenta encontrar expertos únicos vía hash perturbation. Si `max_k > num_experts`, o si la secuencia de hash entra en un ciclo que solo mapea a expertos ya seleccionados, el loop nunca termina.
- **Fix**: Agregar contador máximo de reintentos (e.g., `max_k * num_experts * 2`) y hacer break con los expertos encontrados.

---

## PRIORITY 2 — ALTO (Allocaciones en Hot-Path)

### 2.1 UnifiedBuffer allocation por token en output projection
- **File**: `src/mud/sampling.rs:162-169`
- **Problem**: Dentro de `generate()` (llamado cada token), se aloca un `UnifiedBuffer` nuevo para el path Ternary2Bit output projection. GPU buffer allocation es costoso. Debería usar un workspace buffer pre-alocado.
- **Fix**: Agregar `out_proj_buf: UnifiedBuffer` a `InferenceWorkspace` y reutilizarlo.

### 2.2 RoPE cos/sin tables alocados en `mamba_step` cada llamada
- **File**: `src/mud/forward.rs:1097-1098`
- **Problem**: `vec![0.0f32; d_state / 2]` se heap-aloca en cada invocación de `mamba_step` (una vez por capa por token). Para un modelo de 24 capas, son 24 allocaciones por token.
- **Fix**: Pre-alocar en `InferenceWorkspace` y reutilizar.

### 2.3 `format!()` string allocations en LoRA adapter lookup (hot path)
- **File**: `src/mud/forward.rs:1239, 1263`
- **Problem**: `format!("expert.{}.w1", _expert_id)` se llama dentro de `run_expert_ffn` que ejecuta por cada experto, cada capa, cada token. Cada llamada aloca.
- **Fix**: Pre-computar LoRA lookup keys durante model loading, o usar stack-allocated key format.

### 2.4 `noisy_logits` allocation en Q-Head routing
- **File**: `src/mud/routing.rs:130`
- **Problem**: `Vec::with_capacity(logits.len())` se aloca en cada iteración LDT cuando la certeza es baja.
- **Fix**: Reutilizar `indexed` o un workspace buffer.

### 2.5 `vec![0u8; padding]` allocations en MUD writer
- **File**: `src/mud/mod.rs:131, 145, 210, 228, 237`
- **Problem**: Pequeñas heap allocations para padding bytes durante serialización. Pueden ser hasta 31 bytes.
- **Fix**: Usar stack-allocated `[0u8; 32]` slice.

---

## PRIORITY 3 — MODERADO (Estabilidad Numérica + Arquitectura)

### 3.1 Ternary quantization scale usa fórmula no estándar
- **File**: `src/vulkan/vulkan_backend.rs:21`
- **Problem**: `gamma = mean(|W|)`, `scale = gamma * 0.707`. El paper BitNet b1.58 [2402.17764] usa `scale = 2 * mean(|W|)` para el factor de dequantization. El factor 0.707 (1/sqrt(2)) sugiere una intención de normalización diferente pero no está documentado. Este mismatch puede causar errores de scaling sistemáticos al convertir modelos.
- **Fix**: Documentar la derivación o alinear con la fórmula de referencia.

### 3.2 TTT layer initialization detectada por igualdad floating-point
- **File**: `src/mud/forward.rs:814`
- **Problem**: `if w_t[0] == 0.0` se usa para detectar una state matrix TTT no inicializada. Después de gradient updates, pesos legítimos pueden ser exactamente 0.0, causando re-inicialización falsa.
- **Fix**: Usar un boolean flag o sentinel value.

### 3.3 Softmax overflow handling en atención
- **File**: `src/mud/forward.rs:322-336`
- **Problem**: Después de computar `sum_exp`, si overflows a `+inf`, el código cae al branch `else` que setea todos los scores a 0 excepto `scores[0] = 1.0`. Esto colapsa silenciosamente la distribución de atención a un solo token, lo cual es un problema de correctness para secuencias largas con patrones de atención de alta entropía.
- **Fix**: Usar log-sum-exp throughout para evitar overflow, o clampear valores exp individuales.

### 3.4 `approx_p2` descarta toda la mantisa
- **File**: `src/mud/inference.rs:11-14`
- **Problem**: `f32::from_bits(bits & 0xFF800000)` mantiene solo sign+exponent, produciendo una aproximación power-of-2 cruda usada para KV-cache LOP pruning scoring. Valores como 1.99 y 1.01 mapean al mismo resultado (1.0). Pierde fidelidad de ranking para la selección top-32 keys.
- **Fix**: Considerar mantener el top mantissa bit (mask `0xFFC00000`) para precisión 2-bit, que es esencialmente gratis.

### 4.1 Dual inference engines
- **Files**: `src/model/inference.rs` vs `src/mud/inference.rs`
- **Problem**: Existen dos motores de inferencia completos. La versión `model/` parece ser un engine legacy GGUF-based mientras `mud/` es el engine actual. Ambos compilan, ambos tienen tests, ambos tienen métodos `step()`. Incrementa build time y crea confusión.
- **Fix**: Gate el legacy engine detrás de un feature flag o mover a `tools/legacy/`.

### 4.2 ~50 `[[bin]]` targets en Cargo.toml
- **File**: `Cargo.toml`
- **Problem**: Aproximadamente 50 binary targets incluyendo diagnósticos, auditores, y scripts one-shot. Todos compilan en `cargo build`, incrementando significativamente compile times. Algunos referencian archivos que posiblemente no existen (e.g., `tools/jamba_benchmark.rs`, `tools/vulkan_simulator.rs`, `tools/phase14_audit.rs`).
- **Fix**: Mover herramientas de diagnóstico detrás de un feature flag `tools` o usar workspace con definiciones `[[bin]]` separadas.

### 4.3 Duplicación masiva en Vulkan dispatch methods
- **File**: `src/vulkan/mod.rs`
- **Problem**: `run_ternary_gemm_cached` y `run_ternary_gemm_cached_async` comparten ~80% código idéntico. `pulse_heartbeat` y `dispatch_imagination_async` comparten ~70%. La construcción de descriptor set, pipeline binding, y push constant setup están copy-pasted.
- **Fix**: Extraer helper methods para descriptor set creation y command buffer recording.

### 4.4 `sample_probs` field alocado pero nunca usado
- **File**: `src/mud/workspace.rs:233`
- **Problem**: `sample_probs: Mutex<Vec<(usize, f32)>>` se aloca en `InferenceWorkspace::new` pero nunca se lee ni escribe en ningún lado.
- **Fix**: Eliminarlo.

### 4.5 Unused trace variables
- **File**: `src/mud/forward.rs:887-888`
- **Problem**: `_cos_sim` y `_l2_shift` se computan pero nunca se consumen. Dead code en lo que debería ser hot path.
- **Fix**: Eliminar o integrar en el flujo.

---

## PRIORITY 4 — BAJO (Calidad / Mantenibilidad)

### 5.1 `_pos` parameter naming convention violation
- **File**: `src/mud/forward.rs:12`
- **Problem**: `_pos` tiene prefijo underscore (indicando "unused") pero se usa en línea 208 (`let max_pos = _pos.min(...)`). Mismo caso para `_context`.
- **Fix**: Renombrar a `pos` y `context`.

### 5.2 `build.rs` usa `-march=native`
- **File**: `build.rs:15`
- **Problem**: Hardcodea compilación para CPU de la máquina de build. Los binarios no corren en otros x86_64 CPUs sin el mismo feature set. Rompe cross-compilation.
- **Fix**: Usar `-mavx2 -mfma -mbmi2` explícitamente, o detectar features via environment variable.

### 5.3 RoPE `do_rope` branch vacío en unified shader
- **File**: `assets/shaders/ternary_gemv_unified.comp:75-76`
- **Problem**: `if (pcs.do_rope == 1u) { }` — block vacío. Si algún code path setea `do_rope=1`, el shader produce resultados incorrectos silenciosamente.
- **Fix**: Implementar la lógica RoPE o eliminar el push constant field.

### 5.4 `ghost_align.comp` double-pass sin caching
- **File**: `assets/shaders/ghost_align.comp:30, 53`
- **Problem**: Dos pasadas sobre `x[c]` (global memory reads) sin shared memory caching. Para `cols` grandes, desperdicio significativo de bandwidth.
- **Fix**: Cargar `x[]` en shared memory en una primera pasada cooperativa, luego usarlo para ambos dot products.

### 5.5 Ternary packing format: `00` y `11` ambos decodifican a 0
- **File**: `src/mud/mod.rs:556-560`
- **Problem**: En `dequantize_ternary_row`, bits `11` se tratan igual que `00` (peso cero). Una región de memoria no inicializada (llena de 0xFF) decodificaría a todos ceros en vez de generar error, dificultando detección de corrupción. El módulo ECC solo corrige single-bit flips, así que un u32 completamente corrompido (todos los bits flippados) decodifica a 0 sin alarma.
- **Fix**: Considerar tratar `11` como sentinel/error o usar un patrón de encoding diferente.

---

## Top 3 Fixes Más Impactantes

1. **ECC alignment UB** (`ecc.rs:103-119`) — puede causar crashes o corrupción silenciosa en cualquier plataforma
2. **KV-scales buffer sizing** (`inference.rs:688-701`) — escrituras fuera de bounds en modelos GQA
3. **`div_ceil` vs `/` mismatch** en Vulkan (`vulkan_backend.rs:45-46` + `mod.rs:450`) — inferencia incorrecta o UB

---

## Plan de Acción Sugerido

### Fase 1 — Correcciones Críticas (inmediato)
- [x] ~~Fix alignment UB en ECC con `bytemuck::try_cast_slice`~~ **(DONE 2026-06-10)** — replaced unsafe `from_raw_parts` with `bytemuck::cast_slice`/`cast_slice_mut` + aligned allocators via `bytemuck::cast_vec`
- [x] ~~Unificar `div_ceil(16)` en ambos paths Vulkan~~ **(DONE 2026-06-10)**
- [ ] Dimensionar KV-scales con `num_kv_heads`
- [ ] Agregar retry limit en hash routing
- [ ] Completar shader `shadow_optimizer.comp` output bindings

### Fase 2 — Zero-Alloc Restoration
- [ ] Pre-alocar workspace buffers para output projection, RoPE, y noisy_logits
- [ ] Pre-computar LoRA lookup keys
- [ ] Usar stack padding en MUD writer

### Fase 3 — Estabilidad Numérica
- [ ] Documentar o corregir fórmula de escala ternaria
- [ ] Reemplazar `== 0.0` check en TTT con flag
- [ ] Implementar log-sum-exp en softmax attention
- [ ] Mejorar `approx_p2` con 1-bit mantisa

### Fase 4 — Cleanup Arquitectural
- [ ] Feature-flag o eliminar legacy inference engine
- [ ] Consolidar ~50 bin targets
- [ ] DRY en Vulkan dispatch methods
- [ ] Eliminar dead code (sample_probs, trace variables)
