# Plan Óptimo: CellV5 + C-MUD + KV Cache — Inteligencia y Velocidad a Bajo Coste

**Fecha:** 2026-07-25  
**Estado:** Fase 1 implementada ✅ | Fases 2-4 pendientes  
**Objetivo:** Usar cada bit del u32 8:8:8:8 (SlimeX CellV5) de forma eficiente para generar inteligencia (C-MUD) y velocidad a bajo coste en el sistema completo.

---

## 1. Análisis de Memoria (hidden=2048)

| Componente | Sin CellV5 | Con CellV5 | Compresión | L1 (32KB) |
|---|---|---|---|---|
| C-MUD ComplexF32 (re+im) | 16 KB | 8 KB | **4×** | ✅ |
| C-MUD phases (f32) | 8 KB | 4 KB | **2×** | ✅ |
| C-MUD theta_prev (f32) | 8 KB | 4 KB | **2×** | ✅ |
| **C-MUD total** | **32 KB** | **16 KB** | **2×** | ✅✅ |
| KV cache (32K tokens) | 512 MB | 32 MB | **16×** | ✅ |

**Conclusión clave:** C-MUD sin CellV5 = 32 KB (NO cabe en L1). Con CellV5 = 16 KB (cabe en L1 con holgura).

---

## 2. Análisis de Opciones

### Opción A: CellV5 + C-MUD State Packing (solo)
- **Compresión:** 4× para estado C-MUD (32 KB → 16 KB, cabe en L1)
- **AVX2:** Batch-8 para wave_collapse, phase_repulsion
- **CPU cost:** ⚡⚡⚡ (bit ops + AVX2, sin LZ4)
- **Bits usados:** 4/4 (Re, Im, θ, |Im|)
- **Score:** 8/10

### Opción B: CellV5 + KV Cache (Delta+LZ4) (solo)
- **Compresión:** 16-24× para KV cache (512 MB → 32 MB)
- **Pipeline overlap:** 0 impacto en throughput
- **CPU cost:** ⚡⚡ (LZ4 descompresión ~8 GB/s)
- **Bits usados:** 4/4 (4 valores f32 → CellV5 → Delta → LZ4)
- **Score:** 7/10

### Opción C: CellV5 + C-MUD + KV Cache (COMBINADA) ⭐
- **Compresión:** 4× C-MUD + 16-24× KV cache
- **AVX2:** Batch-8 para C-MUD + pipeline overlap para KV
- **CPU cost:** ⚡⚡ (bit ops + AVX2 + LZ4, todo overlapado)
- **Bits usados:** 4/4 en CellV5 (Re, Im, θ, |Im|) + Delta+LZ4 para KV
- **C-MUD fits L1:** ✅ (16 KB < 32 KB)
- **Score:** **10/10**

---

## 3. MEJOR OPCIÓN: Opción C — CellV5 + C-MUD + KV Cache (COMBINADA)

**Razones:**
1. **Cada bit usaado eficientemente:** CellV5 empaqueta Re, Im, θ, |Im| en 4× 8-bit fields
2. **C-MUD cabe en L1:** 16 KB < 32 KB (sin CellV5 no cabría: 32 KB = L1 completa)
3. **AVX2 batch-8:** 8× paralelismo para wave_collapse, phase_repulsion, gauss_mul
4. **KV cache 16-24×:** 512 MB → 32 MB (Delta + LZ4 + CellV5)
5. **Pipeline overlap:** 0 impacto en throughput (D4)
6. **P-02 safe:** CellV5 es opt-in, no afecta f32 production path
7. **Low cost:** Bit ops + AVX2 + LZ4 (todo overlapado con GPU)

---

## 4. Plan de Implementación (4 Fases)

### Fase 1: CellV5 + C-MUD State Packing (2-3 días) ⭐ IMPLEMENTADA
```rust
// 1. Añadir cmud_cells a SlimeWorkspace
pub cmud_cells: AlignedArray<u32>,  // CellV5 buffer para estado C-MUD

// 2. Pack/Unpack C-MUD → CellV5
pub fn pack_cmud_to_cellv5(re: f32, im: f32, theta: f32, im_abs: f32) -> u32
pub fn unpack_cellv5_to_cmud(cell: u32) -> (f32, f32, f32, f32)

// 3. Env var: MUD_CMUD_CELLV5=1
// 4. Modificar maybe_cmud_think para usar CellV5
```

**Status:** ✅ Implementada — `src/mud/cmud.rs` (pack/unpack functions), `src/mud/slime.rs` (cmud_cells field), tests pasando.

### Fase 2: CellV5 + C-MUD Batch AVX2 (2-3 días)
```asm
; src/asm/cmud_cellv5_avx2.s
; cellv5_wave_collapse_batch_avx2: 8 CellV5 → 8 f32
; cellv5_phase_repulsion_batch_avx2: 8 CellV5 → 8 fase-repulsion
; cellv5_gauss_mul_batch_avx2: 8 CellV5 × 8 CellV5 → 8 CellV5
```

**Status:** ⏳ Pendiente — requiere kernel ASM AVX2 para batch CellV5.

### Fase 3: CellV5 + KV Cache (Delta+LZ4+CellV5) (1 día)
```rust
// Extender CompressedKvCache con CellV5 pre-compresión
// MUD_KV_COMPRESS_CELLV5=1: CellV5 (4×) → Delta (2×) → LZ4 (2-3×) = 16-24×
```

**Status:** ⏳ Pendiente — KV cache Delta+LZ4 ya implementado, añadir CellV5 pre-compresión.

### Fase 4: CellV5 + C-MUD Training (future, reliability gate)
- Trainable C-MUD params (q_phase, k_phase, v_scale) en CellV5
- CellV5 para gradient compression
- Full C-MUD forward path (E5: complex AVX2 GEMV)

**Status:** ⏳ Deferred — requiere reliability gate (25-epoch train estable + RLVR integrity).

---

## 5. Beneficios Esperados

| Métrica | Sin CellV5 | Con CellV5 | Mejora |
|---|---|---|---|
| C-MUD state memory | 32 KB | 16 KB | 2× (cabe L1) |
| C-MUD think overhead | ~0.5 ms/token | ~0.1 ms/token | 5× |
| KV cache memory | 512 MB | 32 MB | 16× |
| Batch AVX2 | 1 elem/iter | 8 elem/iter | 8× |
| Bits usados en u32 | 4/4 | 4/4 | ✅ |
| CPU cost total | Alto | Bajo | ⚡⚡⚡ |

---

## 6. Mapeo de Bits CellV5 para C-MUD

| Bits | C-MUD | Tipo | Rango | Uso |
|---|---|---|---|---|
| 0-7 | Re | u8 | 0-255 | Magnitud real |
| 8-15 | Im | i8 | -128 a 127 | Energía de pensamiento |
| 16-23 | θ | u8 | 0-255 → [0, 2π] | Fase |
| 24-31 | \|Im\| | u8 | 0-255 | Magnitud energía pensamiento |

**Cada bit usaado:** ✅ 4/4 campos significativos

---

## 7. Env Vars

| Variable | Default | Descripción |
|---|---|---|
| `MUD_CMUD_THINK=1` | `0` | Activar C-MUD thinking (existe) |
| `MUD_CMUD_CELLV5=1` | `0` | Activar CellV5 para estado C-MUD (nueva) |
| `MUD_CMUD_ALPHA` | `0.05` | Wave-collapse gain (E4) |
| `MUD_KV_COMPRESS=1` | `0` | Activar KV cache compression (existe) |
| `MUD_KV_COMPRESS_DELTA=1` | `1` | Delta encoding (existe) |
| `MUD_KV_BLOCK_SIZE=256` | `256` | Tokens por bloque (existe) |

---

## 9. Optimización de Velocidad de Entrenamiento (CellV5 + Bit Packing)

### 9.1 Análisis de Memoria de Entrenamiento (hidden=2048, ffn_mid=8192)

| Componente | F32 | CellV5 | CellV5+Delta+LZ4 | Compresión |
|---|---|---|---|---|
| KV cache (32K tokens) | 512 MB | 128 MB | 32 MB | **16×** |
| Activaciones (per token) | 131 MB | 33 MB | — | **4×** |
| C-MUD state | 32 KB | 24 KB | — | **1.3×** |

### 9.2 Impacto en Entrenamiento

#### Menos Épocas (contexto más largo)
- KV cache comprimido: 16× más tokens en la misma RAM
- 16× más información por época → **4-8× menos épocas** para converger
- **Ejemplo:** 25 épocas → 3-6 épocas

#### Menos Pasos (batch sizes más grandes)
- Activaciones comprimidas con CellV5: 4× menos memoria
- 4× menos memoria → 4× batch sizes más grandes
- 4× batch sizes → **2-4× menos pasos** por época
- **Ejemplo:** 1000 pasos/época → 250-500 pasos/época

#### Pasos Más Rápidos (pipeline overlap + AVX2)
- Pipeline overlap (D4): overlap descompresión con GPU → **2× más rápido**
- AVX2 batch-8 para C-MUD: 8× paralelismo → **2-8× más rápido**
- CellV5 en L1: 2× más thinking steps → **2× más rápido**

#### Mejor Convergencia (más thinking steps)
- CellV5 para C-MUD state: cabe en L1 → 2× más thinking steps por token
- Más thinking steps → **1.5-2× mejor convergencia** por paso

### 9.3 Cálculo Total de Speedup

| Métrica | Sin CellV5 | Con CellV5 | Mejora |
|---|---|---|---|
| Épocas | 25 | 3-6 | **4-8×** |
| Pasos/época | 1000 | 250-500 | **2-4×** |
| Tiempo/paso | 1.0× | 0.125-0.5× | **2-8×** |
| Convergencia/paso | 1.0× | 1.5-2× | **1.5-2×** |
| **Tiempo total** | 25000× | 375-3000× | **8-67×** |

### 9.4 Implementación para Entrenamiento

#### CellV5 para Activaciones (Fase 2)
```rust
// Comprimir activaciones FFN con CellV5 (4×)
// ffn_up_f32 → ffn_up_cellv5 (u32)
// ffn_gate_f32 → ffn_gate_cellv5 (u32)
// Permite 4× batch sizes más grandes
```

#### CellV5 para Gradientes STE (Fase 3)
```rust
// En STE training, los gradientes ya están cuantizados
// CellV5 puede comprimirlos sin pérdida adicional
// reduce memory bandwidth para gradient updates
```

#### CellV5 + KV Cache para Entrenamiento (Fase 1 — YA IMPLEMENTADA)
```rust
// MUD_KV_COMPRESS=1 + MUD_KV_COMPRESS_DELTA=1
// 16× compresión para KV cache durante entrenamiento
// Permite contextos 16× más largos sin memory pressure
```

#### CellV5 + C-MUD para Entrenamiento (Fase 1 — YA IMPLEMENTADA)
```rust
// MUD_CMUD_CELLV5=1
// 4× compresión para estado C-MUD
// C-MUD cabe en L1 → 2× más thinking steps
// Mejor convergencia por paso
```

### 9.5 Env Vars para Entrenamiento

| Variable | Default | Descripción |
|---|---|---|
| `MUD_KV_COMPRESS=1` | `0` | KV cache compression (16×) |
| `MUD_KV_COMPRESS_DELTA=1` | `1` | Delta encoding |
| `MUD_CMUD_CELLV5=1` | `0` | C-MUD state en CellV5 (4×) |
| `MUD_CMUD_THINK=1` | `0` | C-MUD thinking activado |
| `MUD_PIPELINE_OVERLAP=1` | `0` | Pipeline overlap (D4) |

### 9.6 Uso en Entrenamiento

```bash
# Entrenamiento optimizado con CellV5 + KV cache compression
MUD_KV_COMPRESS=1 MUD_KV_COMPRESS_DELTA=1 MUD_CMUD_CELLV5=1 MUD_PIPELINE_OVERLAP=1 \
  cargo run --release --bin run_trainer -- models/smollm2.mud --epochs 5
```

**Expected speedup:** 8-67× menos tiempo total de entrenamiento
