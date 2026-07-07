# Hamming Codes para BitNet — Error Correction en Pesos Ternarios

## Motivación

BitNet almacena pesos ternarios `{-1, 0, +1}` en formato empaquetado de **2 bits por peso**:

| Valor | Bits  | Hex  |
|-------|-------|------|
| `+1`  | `01`  | 0x1  |
| `-1`  | `10`  | 0x2  |
| `0`   | `00`  | 0x0  |
| inv.  | `11`  | 0x3  |

Un solo **bit flip** puede mutar `+1` en `-1`, `0` en `+1`, o peor — convertir `00` en `11` (valor inválido). En modelos ternarios con millones de pesos, los bit flips inducidos por DRAM sin ECC degradan la perplejidad silenciosamente.

En el engine MUD, los pesos se mapean **zero-copy** desde el `.mud` vía `mmap()` y se consumen directamente como `*const u32` en los kernels AVX2 y shaders Vulkan. No existe ninguna verificación de integridad entre disco y cómputo.

---

## Hamming(15,11) para pesos empaquetados

Protege **11 bits de datos** (5 pesos = 10 bits + 1 bit reservado) con 4 bits de paridad en un solo `u16`.

| Campo   | Bits | Descripción                     |
|---------|------|---------------------------------|
| Datos   | 11   | 5 pesos ternarios (10 bits) + 1 |
| Paridad | 4    | Síndrome Hamming(15,11) pos 1,2,4,8 |

Bits de paridad en posiciones potencia-de-2 (1-indexed, little-endian):

```
p1 = d1 ⊕ d3 ⊕ d5 ⊕ d7  ⊕ d9  ⊕ d11
p2 = d2 ⊕ d3 ⊕ d6 ⊕ d7  ⊕ d10 ⊕ d11
p4 = d4 ⊕ d5 ⊕ d6 ⊕ d7
p8 = d8 ⊕ d9 ⊕ d10 ⊕ d11
```

Síndrome (posición del error, 0 = sin error):

```
s = p8*8 + p4*4 + p2*2 + p1
```

Si `s ≠ 0` y `s ≤ 11` → flip en bit de datos `s`. Si `s > 15` → error doble.

### Implementación Rust

```rust
fn hamming_encode(packed_5_weights: u16) -> u16 {
    let d = packed_5_weights & 0x7FF;
    let p1 = (d >> 0) & 1 ^ (d >> 2) & 1 ^ (d >> 4) & 1
           ^ (d >> 6) & 1 ^ (d >> 8) & 1 ^ (d >> 10) & 1;
    let p2 = (d >> 1) & 1 ^ (d >> 2) & 1 ^ (d >> 5) & 1
           ^ (d >> 6) & 1 ^ (d >> 9) & 1 ^ (d >> 10) & 1;
    let p4 = (d >> 3) & 1 ^ (d >> 4) & 1 ^ (d >> 5) & 1 ^ (d >> 6) & 1;
    let p8 = (d >> 7) & 1 ^ (d >> 8) & 1 ^ (d >> 9) & 1 ^ (d >> 10) & 1;
    d | (p1 << 11) | (p2 << 12) | (p4 << 13) | (p8 << 14)
}

fn hamming_decode(encoded: u16) -> (u16, bool, bool) {
    let d = encoded & 0x7FF;
    let s1 = ((encoded >> 11) & 1) ^ (d >> 0) & 1 ^ (d >> 2) & 1
           ^ (d >> 4) & 1 ^ (d >> 6) & 1 ^ (d >> 8) & 1 ^ (d >> 10) & 1;
    let s2 = ((encoded >> 12) & 1) ^ (d >> 1) & 1 ^ (d >> 2) & 1
           ^ (d >> 5) & 1 ^ (d >> 6) & 1 ^ (d >> 9) & 1 ^ (d >> 10) & 1;
    let s4 = ((encoded >> 13) & 1) ^ (d >> 3) & 1 ^ (d >> 4) & 1
           ^ (d >> 5) & 1 ^ (d >> 6) & 1;
    let s8 = ((encoded >> 14) & 1) ^ (d >> 7) & 1 ^ (d >> 8) & 1
           ^ (d >> 9) & 1 ^ (d >> 10) & 1;
    let syndrome = (s8 << 3) | (s4 << 2) | (s2 << 1) | s1;
    match syndrome {
        0 => (d, false, false),
        _ if (1..=11).contains(&syndrome) => (d ^ (1 << (syndrome - 1)), true, false),
        _ if syndrome < 16 => (d, true, false),   // error en paridad
        _ => (d, true, true),                     // error doble
    }
}
```

### Overhead

| Formato          | bits/peso | Overhead |
|------------------|-----------|----------|
| Raw (sin ECC)    | 2.0       | 0%       |
| Hamming(15,11)   | 2.91      | ~45%     |
| SECDED (32+7)    | 2.44      | ~22%     |
| Parity bit c/u32 | 2.06      | ~3%      |

**Recomendación:** SECDED (32+7) da el mejor balance overhead/protección para el `.mud` format. Hamming(15,11) es mejor para integridad in-memory por su alineación natural con `u16`.

---

## Auditoría del Codebase: Puntos de Integración

El flujo actual de pesos ternarios en MUD:

```
tools/universal_converter/quantizer.rs   ← PACKING (producción)
       ↓
src/mud/mod.rs::save()                   ← ESCRITURA .mud
       ↓
src/mud/mod.rs::load()                   ← mmap() zero-copy
       ↓
src/mud/inference.rs                     ← DISPATCH (gemm_vulkan_or_cpu)
       ↓
  ┌──── VULKAN ────┐    ┌──── CPU ──────────────────┐
  │ ternary_gemv    │    │ pext_unpack_ternary.s     │
  │ (shader)        │    │   + ternary_gemv_lut.s    │
  └─────────────────┘    └───────────────────────────┘
```

### 5 Puntos de Inserción de ECC

#### 1. On-disk: `src/mud/mod.rs` — `save()` (L54) / `load()` (L140)

**Dónde:**
- `save()` escribe tensores raw con padding 32-byte — sin checksums
- `load()` setea `data_ptr` directo al mmap — sin verificar

**Implementación:** Agregar tensor auxiliar de paridad por cada tensor ternario durante `save()`. Verificar durante `load()`. No modificar el layout existente — usar un skill/metadata separado (ej. `_ecc.parity`).

**Prioridad:** ALTA — detecta corrupción silenciosa en disco/transferencia.

#### 2. Pre-GPU upload: `src/mud/inference.rs` — `gemv_vulkan_or_cpu()` (L2588)

**Dónde:** Antes de llamar `vk.run_ternary_gemv_cached()`, las filas empaquetadas viajan como `*const u32`.

**Implementación:** Iterar cada fila ternaria, decodificar Hamming(15,11) bloques. Si hay errores, corregir en un buffer temporal antes del upload al GPU buffer.

**Prioridad:** MEDIA — GPU normalmente tiene VRAM con ECC, pero el bus PCIe puede tener flips.

#### 3. PEXT unpack: `src/asm/ternary_pext.s` (L1–69)

**Dónde:** `pext_unpack_ternary` lee 64 bits (2 u32s = 32 pesos) y produce 32 `i8`. Este es el punto exacto donde los pesos se decodifican en el path CPU fallback.

**Datos de referencia:** `src/asm/mod.rs` L10–60 (FFI declarations).

**Implementación:** Insertar verificación ECC después de la extracción vía `pext`. Cada 32 pesos = 64 bits raw = 24 bits de paridad adicional (3 `u16`s ECC). Comparar síndrome contra paridad almacenada.

**Ventaja:** Corrección antes de que entre al LUT GEMV (`ternary_gemv_lut.s`). Cero overhead en el hot-loop porque el unpack ya es cold-path (se hace una vez por fila, no por token).

**Prioridad:** ALTA — protege el path de inferencia CPU.

#### 4. Conversión model: `tools/universal_converter/quantizer.rs` — `ternarize_and_pack()` (L112)

**Dónde:** `pack_ternary_from_f32()` (L191) produce los bytes empaquetados que luego se escriben al `.mud`. Este es el **mejor lugar para inyectar ECC al origen**.

**Implementación:** Después de `pack_ternary_from_f32()`, calcular paridad SECDED por cada 32 valores (2 u32s) y almacenar en un vector de paridad adjunto al tensor.

**Prioridad:** ALTA — es el punto de producción; si se genera ECC aquí, todos los demás niveles pueden usarlo.

#### 5. `dequantize_ternary_row()`: `src/mud/mod.rs` (L253)

**Dónde:** Convierte packed `*const u32` a `&mut [f32]` — usado en embedding lookup y debugging.

**Implementación:** Verificar ECC inline durante la descompresión del embedding table. La tabla de embeddings tiene ~32000 filas × hidden_size; un bit flip aquí produce tokens fantasma.

**Prioridad:** MEDIA — solo protege embeddings, no el core del transformer.

---

## Plan de Implementación Priorizado

### Fase 1 — Protección en escritura/lectura `.mud`
- Agregar `ecc_parity: Vec<u8>` a `MudTensor` (`src/mud/mod.rs:28`)
- En `save()`: calcular SECDED(32+7) por cada bloque 32 valores y guardar en tensor `_ecc`
- En `load()`: verificar paridad de cada tensor ternario, loguear warning si hay errores
- En `load()`: si se detectaron errores y hay paridad, corregir en `owned_data`

### Fase 2 — Protección in-memory (CPU path)
- En `gemv_vulkan_or_cpu()` (`src/mud/inference.rs:2588`): antes de `pext_unpack_ternary`, verificar ECC
- Modificar `pext_unpack_ternary` para aceptar puntero de paridad opcional y corregir al vuelo
- Agregar flag de hardware ECC detection (`/proc/cpuinfo` o `cpuid`) para saltar verificación si el hardware ya protege

### Fase 3 — Integración en conversión
- En `quantizer.rs:ternarize_and_pack()`: generar ECC inline durante el empaquetado
- Los archivos `.mud` nuevos incluirán ECC desde origen

---

## Referencias

- Hamming, R. W. "Error Detecting and Error Correcting Codes." Bell System Technical Journal, 1950.
- Microsoft BitNet b1.58: arXiv:2402.17764
- bitnet.cpp: arXiv:2410.16144
- Código fuente MUD: `src/mud/mod.rs`, `src/mud/inference.rs`, `src/asm/ternary_pext.s`, `tools/universal_converter/quantizer.rs`
