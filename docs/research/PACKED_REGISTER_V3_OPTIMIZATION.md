# PackedRegister v3 / v4 — Dynamic Cell Architecture

## 1. Formato de Célula de Transporte (u32)

```text
SlimeRegister = PackedRegister(u32)
  bits  0-15: real = f16(matmul_accum)       — Acumulador GEMV ternario (real)
  bits 16-23: imag = u8(v_jepa)              — Integral / Velocidad orbital JEPA
  bits 24-27: grad = u4(sgn(∇W), STE_flag)   — Dirección de gradiente STE QAT
  bits 28-31: addr = u4(routing_tag)         — Dirección / Tag en Tabla de Continuidad

CMUD thinker → Complex<f16> = (real, imag) + dirección en Tabla de Continuidad
```

## 2. Mapa de registro AVX2

Cada `PackedRegister` = 32 bits = **½ YMM lane** (256-bit YMM = 8 registros).

```
YMM (256 bits) = 8 × PackedRegister
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  R0  │  R1  │  R2  │  R3  │  R4  │  R5  │  R6  │  R7  │
├──┬───┼──┬───┼──┬───┼──┬───┼──┬───┼──┬───┼──┬───┼──┬───┤
│re│im │re│im │re│im │re│im │re│im │re│im │re│im │re│im │
└──┴───┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┘
 0    4    8   12   16   20   24   28   32   36   40   44   48   52   56   60
```

## 3. Operaciones AVX2 batch (8 dimensiones)

### 3.1 Pack → f32 (registers_to_f32)

```asm
; Entrada: rsi = &PackedRegister[8] (256 bits, 32 bytes alineados)
; Salida:  rdi = &f32[8] (real components)

vmovdqa     (%rsi), %ymm0          ; Cargar 8 registros = 256 bits
vextracti128 $0, %ymm0, %xmm1     ; XMM1 = bits 0-127 (re de dims 0-7)
vextracti128 $1, %ymm0, %xmm2     ; XMM2 = bits 128-255 (im de dims 0-7)

; XMM1[0..7] = f16(real[0..7]), XMM2[0..7] = f16(imag[0..7])
vcvtph2ps   %xmm1, %ymm3          ; YMM3 = f32(real[0..7])
vcvtph2ps   %xmm2, %ymm4          ; YMM4 = f32(imag[0..7])

vmovups     %ymm3, (%rdi)         ; Store real f32
vmovups     %ymm4, 32(%rdi)       ; Store imag f32 (offset 256 bits)
```

**Costo**: 6 instrucciones, 8 ciclos (Haswell+) para 16 valores f32.
**vs escalar**: 16 × vcvtss2si = ~32 ciclos. **Speedup 4×**.

### 3.2 f32 → Pack (writeback thinker)

```asm
; Entrada: rdi = &PackedRegister[8] (salida)
;          rsi = &f32[8] real
;          rdx = &f32[8] imag

vmovups     (%rsi), %ymm0          ; Cargar 8 real f32
vmovups     (%rdx), %ymm1          ; Cargar 8 imag f32

vcvtps2ph   $0, %ymm0, %xmm2      ; XMM2 = f16(real[0..7])
vcvtps2ph   $0, %ymm1, %xmm3      ; XMM3 = f16(imag[0..7])

; Interleave: XMM2 (re) en low 64b, XMM3 (im) en high 64b de cada lane
vmovdqa     %xmm2, %xmm4          ; Copia
vpunpcklwd  %xmm3, %xmm2, %xmm0   ; XMM0[0]=re0, XMM0[1]=im0, XMM0[2]=re1, XMM0[3]=im1, ...
vpunpckhwd  %xmm3, %xmm4, %xmm1   ; XMM1[0]=re4, XMM1[1]=im4, ...
vinserti128 $1, %xmm1, %ymm0, %ymm0  ; YMM0 = 8 packed registers

vmovdqa     %ymm0, (%rdi)          ; Store 8 × PackedRegister
```

**Costo**: 9 instrucciones, ~12 ciclos para 8 registros completos.
**vs escalar**: 16 × cvtss2si + 8 stores = ~40 ciclos. **Speedup 3.3×**.

### 3.3 Batch gate (σ(imag) para 8 dimensiones)

```asm
; Entrada: ymm4 = f32(imag[0..7])
; Salida:  ymm5 = f32(gate[0..7]) donde gate = 1/(1+exp(-x))

; 1. exp(-x) via polinomio grado 4 (rutina existente en silu.s)
vmovups     .Llog2e(%rip), %ymm6   ; log2(e)
vmulps      %ymm6, %ymm4, %ymm7   ; -x * log2(e)
; ... (usar el mismo polinomio exp2 de silu_vectorial_avx2)

; 2. gate = 1/(1+exp(-x))
vmovups     .Lone(%rip), %ymm0    ; 1.0
vaddps      %ymm0, %ymm7, %ymm1   ; 1 + exp(-x)
vrcpps      %ymm1, %ymm5          ; 1/(1+exp(-x)) aproximación
; (opcional: iteración NR para mayor precisión)
```

**Costo**: ~20 ciclos para 8 gates (vs 8 × scalar = ~80 ciclos). **Speedup 4×**.

---

## 4. Ganancia total con batch AVX2

| Operación | Escalar (8 dims) | Batch AVX2 (8 dims) | Speedup |
|-----------|-----------------|---------------------|---------|
| registers_to_f32 | 32 ciclos | 8 ciclos | **4.0×** |
| f32_to_registers | 40 ciclos | 12 ciclos | **3.3×** |
| Gate σ(imag)×8 | 80 ciclos | 20 ciclos | **4.0×** |
| RMSNorm (hidden=256) | ~1024 ciclos | ~256 ciclos | **4.0×** |
| mHC blend (hidden=256) | ~768 ciclos | ~192 ciclos | **4.0×** |
| Output norm (hidden=256) | ~512 ciclos | ~128 ciclos | **4.0×** |

**Ahorro total por token (layers=12):**
- RMSNorm: 12 × (1024 - 256) = 9,216 ciclos
- mHC blend: 12 × (768 - 192) = 6,912 ciclos
- Output norm: 1 × (512 - 128) = 384 ciclos
- **Total: ~16,512 ciclos ≈ 5 μs a 3.3 GHz**

Esto es ~0.5% del costo total por token (~1ms). La optimización real está en pipeline, no en aritmética.

---

## 5. Exterminación de Vec en hot path

### 5.1 Estado actual (SlimeWorkspace)

```rust
pub struct SlimeWorkspace {
    pub registers: Vec<SlimeRegister>,     // 24B overhead
    pub registers_tmp: Vec<SlimeRegister>, // 24B overhead
    pub q_f32: Vec<f32>,                   // 24B overhead
    pub k_f32: Vec<f32>,
    pub v_f32: Vec<f32>,
    pub scores: Vec<f32>,
    pub o_act_f32: Vec<f32>,
    pub ffn_up_f32: Vec<f32>,
    pub ffn_gate_f32: Vec<f32>,
    pub ffn_mid_f32: Vec<f32>,
    pub ffn_out_f32: Vec<f32>,
    pub gemv_accum: Vec<f32>,
    pub norm_i8: Vec<i8>,
    pub kv_row_scratch: Vec<f32>,
    // ... más Vecs
}
```

Cada `Vec<T>` = 24 bytes en la struct (ptr + len + cap) + heap allocation.
- **15 campos Vec activos** en hot path → 15 × 24 = **360 bytes de overhead** solo en la struct
- Cada acceso `ws.registers[i]` → check `i < len` (bounds check implícito)
- Cada realloc potencial → capacity tracking

### 5.2 Propuesta: raw pointers + AlignedArray

```rust
#[repr(C)]
pub struct AlignedArray<T> {
    ptr: *mut T,
    len: usize,     // 16 bytes total (vs 24 de Vec)
    _phantom: PhantomData<T>,
}

impl<T: Copy> AlignedArray<T> {
    pub fn new(size: usize, align: usize) -> Self {
        let layout = Layout::from_size_align(size * mem::size_of::<T>(), align).unwrap();
        let ptr = unsafe { alloc::alloc_zeroed(layout) as *mut T };
        Self { ptr, len: size, _phantom: PhantomData }
    }
    
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, i: usize) -> T {
        *self.ptr.add(i)
    }
    
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, i: usize, val: T) {
        self.ptr.cast::<T>().add(i).write(val);
    }
}
```

Ganancia:
- 24 → 16 bytes por campo (8 bytes menos por buffer = ~120 bytes en workspace)
- 0 bounds checks en acceso directo (el hot path nunca accede out of bounds)
- 0 capacity tracking (no realloc)
- Alignment controlado (64-byte para AVX2)

### 5.3 Hot path sin Vec — puntero directo

**Antes (con Vec):**
```rust
// slime_forward.rs:607
let val = h_in[i].read_accum();     // bounds check + vtable dispatch
h_out[i].write_accum(result);       // bounds check + vtable dispatch
```

**Después (raw ptr):**
```rust
let regs = workspace.registers_ptr;  // *const SlimeRegister
let h_in = regs.add(i);
let re = (*h_in).read_accum();      // sin bounds check
let h_out = regs_mut.add(i);
(*h_out).write_accum(result);
```

### 5.4 Eliminación de registers_tmp

**Propósito actual**: `registers_tmp` existe para almacenar el resultado de FFN antes del mHC residual final que lo mezcla de vuelta a `registers`. Es un doble buffer.

**Optimización**: Con `PackedRegister` de 4 bytes, podemos usar stack allocation para hidden_size ≤ 384:

```rust
// hidden_size ≤ 384: 384 * 4B = 1.5KB de stack (within 4KB frame)
let mut tmp: [PackedRegister; 384] = unsafe { mem::zeroed() };
// hidden_size > 384: heap (raro, solo modelos ultra)
```

O directamente: eliminar `registers_tmp` y usar `ffn_out_f32` (que ya existe como Vec<f32> de hidden_size) como buffer temporal, pues el PackedRegister cabe en 4 bytes y podemos convertir f32 → PackedRegister al vuelo.

Ganancia: **eliminar 1 Vec de 24B + 2KB de heap por workspace**.

---

## 6. Mapa de exterminación total

| Componente | Antes | Después | Ahorro |
|-----------|-------|---------|--------|
| SlimeRegister | 2 × f32 (64 bits) | 1 × u32 (32 bits) | **50%** |
| registers Vec | 24B + hidden×8B | raw ptr + hidden×4B | **16B + 50% heap** |
| registers_tmp | 24B + hidden×8B | **eliminado** (usa ffn_out_f32) | **24B + hidden×8B** |
| q/k/v/scores Vec | 5 × 24B + data | 5 × raw ptr + AlignedArray | **5 × 16B** |
| o_act/ffn_*/gemv_accum Vec | 6 × 24B + data | 6 × raw ptr | **6 × 16B** |
| norm_i8 Vec | 24B + data | raw ptr + AlignedArray | **16B** |
| Total workspace overhead | 15 × 24B = 360B | 15 × 16B = 240B | **120B** (33%) |
| Bounds checks por capa | hidden × 12 × 5 = 15,360 | 0 (unsafe unchecked) | **100%** |
| Capacity tracking | 15 × 8B = 120B | 0 | **100%** |

---

## 7. Optimizaciones futuras (research)

### 7.1 LM head AVX2 con registros empaquetados

El kernel `lm_head_logits_avx2` actualmente lee de un `reg_f32` plano. Se puede modificar para leer directamente del array `PackedRegister` convirtiendo f16→f32 on-the-fly:

```asm
; En vez de vmovups reg_f32 (load f32), hacer:
vmovdqa     (%rsi), %ymm0          ; Load 8 PackedRegister
vextracti128 $0, %ymm0, %xmm1     ; Extract real f16
vcvtph2ps   %xmm1, %ymm2          ; Convert 8 real → f32
; YMM2 ready para FMA con pesos
```

Esto elimina el paso intermedio `registers → reg_f32` completamente.
**Ahorro**: hidden × 4B = 1KB de tráfico de memoria por token.

### 7.2 KV cache como PackedRegister

Si la KV cache (actualmente f32 o f16 separado) usara el mismo formato PackedRegister:
- K cache = array de PackedRegister
- V cache = array de PackedRegister
- Cada load de KV ya trae el par (real = K, imag = V) en un solo acceso de 32 bits
- **Ahorro**: 50% en footprint de KV cache

### 7.3 Dot product sobre PackedRegister

El GEMV ternario actualmente descomprime ELUT → f32 y acumula en f32. Se podría modificar para que la salida del GEMV sea directamente un PackedRegister (low 16 bits = resultado ternario cuantizado a f16, high 16 bits = escala PRQ):

```asm
; Al final de ternary_gemv_avx2:
; xmm0 = resultado f32
vcvtps2ph   $0, %ymm0, %xmm1     ; f32 → f16
; xmm1 = f16(resultado) en low 16 bits
; xmm2 = f16(PRQ_scale) en high 16 bits (desde scales[r])
vpinsrw     $1, %xmm2, %xmm1, %xmm1  ; embeber escala
vmovq       %xmm1, (%rdx)         ; Store PackedRegister (64 bits = 2 registros)
```

### 7.4 JEPA stabilizer SIMD

Actualmente la JEPA stabilizer procesa cada dimensión individualmente:
```rust
for i in 0..hidden {
    let y = block_out[i];
    let z = z_buf[i];
    let z_next = z * (1-alpha) + alpha * y + jitter;
    v_jepa = (z_next - mu) * inv_sigma;
    registers[i].update_integral(v_jepa);
}
```

Se puede SIMD-izar con AVX2:
```asm
; Procesar 8 dimensiones por iteración
; ymm0 = block_out[0..7] (f32)
; ymm1 = z_buf[0..7] (f32)
; ymm2 = broadcast(1-alpha)
; ymm3 = broadcast(alpha)
; ymm4 = broadcast(mu)
; ymm5 = broadcast(inv_sigma)

vmulps      %ymm2, %ymm1, %ymm6     ; z * (1-alpha)
vmulps      %ymm3, %ymm0, %ymm7     ; alpha * y
vaddps      %ymm6, %ymm7, %ymm1     ; z_next
vsubps      %ymm4, %ymm1, %ymm6     ; z_next - mu
vmulps      %ymm5, %ymm6, %ymm7     ; v_jepa = (z-mu) * inv_sigma

; Pack v_jepa (f32) → f16 y store en high 16 bits de registers
; (requiere load de register, merge, store)
```

**Speedup**: 8× vs escalar.

---

## 8. Resumen de optimizaciones (proyecto completo)

| # | Optimización | File(s) afectados | Ganancia estimada |
|---|-------------|-------------------|-------------------|
| 1 | PackedRegister (f16×2) | `slime.rs` | **50% register footprint** ✅ HECHO |
| 2 | Avx2 batch registers_to_f32 | Nuevo: `asm/pack.s` | **4× en conversión** |
| 3 | Eliminar registers_tmp | `slime_forward.rs`, `workspace.rs` | **2KB + bounds checks** |
| 4 | Raw ptr workspace (no Vec) | `workspace.rs` + 15 archivos | **33% struct overhead** |
| 5 | Avx2 batch gate(σ) | Nuevo: `asm/gate.s` | **4× en gate JEPA** |
| 6 | LM head directo desde packed | `main.rs`, `lm_head.s` | **1KB menos BW/token** |
| 7 | KV cache como PackedRegister | `slime.rs`, `kv_dtype.rs` | **50% KV footprint** |
| 8 | JEPA stabilizer SIMD | `slime_jepa.rs` | **8× en JEPA** |
| 9 | Dot product → PackedRegister | `ternary_gemv.s` | **elimina paso f32** |
| 10 | Stack alloc para hidden≤384 | `slime.rs` (workspace new) | **0 heap para modelos nano/micro** |

**Ganancia total estimada en throughput**: ~5-15% en inferencia, ~10-20% en training.
El bottleneck real sigue siendo el GEMV (FFN QKV), pero las optimizaciones de registros eliminan overhead secundario y preparan el pipeline para futuras mejoras SIMD.
