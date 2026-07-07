# Investigación: Papers para Aplicar a MUD

## Hallazgos Clave (Jun 2026)

### 1. BitNet: Scaling 1-bit Transformers (arXiv:2310.11453)
**Autores:** Hongyu Wang et al. (Microsoft Research)  
**Fecha:** Oct 2023

**Aportes principales:**
- **BitLinear**: Reemplazo drop-in de nn.Linear que entrena weights 1-bit desde scratch
- **Quantization-aware training**: Usa straight-through estimator (STE) para backprop
- **Scaling law**: BitNet sigue scaling laws similares a Transformers full-precision
- **Resultados:**
  - Mismo perplexity que FP16 con misma cantidad de tokens
  - 3× menos uso de memoria
  - 4× menos consumo energético

**Aplicabilidad a MUD:**
- ✅ Ya implementamos pesos ternarios {-1, 0, 1}
- ✅ Inference optimizado con SIMD (AVX2)
- ❌ Faltan técnicas de training (no es foco de MUD)
- 💡 **Idea:** Implementar BitLinear como opción de export desde PyTorch

---

### 2. The Era of 1-bit LLMs: BitNet b1.58 (arXiv:2402.17764)
**Autores:** Shuming Ma et al. (Microsoft Research)  
**Fecha:** Feb 2024

**Aportes principales:**
- **BitNet b1.58**: Cada parámetro es ternario {-1, 0, 1} exactamente 1.58 bits
- **Zero quantization overhead**: No hay degradación vs FP16 en perplexity
- **Nueva escala de eficiencia:**
  - Latencia: 4.1× más rápida que FP16
  - Throughput: 5.2× mayor
  - Energía: 10.3× menos consumo

**Relevancia crítica:**
- Este es el modelo que tenemos convertido (`bitnet-b1.58-2B-4T.mud`)
- Confirma que nuestra aproximación ternaria es state-of-the-art
- **Bug actual:** La corrupción de logits nos impide usar este avance

**Aplicabilidad inmediata:**
- 💡 **Prioridad:** Fixear bug de logits para desbloquear inference b1.58
- 💡 **Futuro:** Explorar hardware especializado para 1-bit LLMs

---

### 3. FlashAttention: IO-Aware Exact Attention (arXiv:2205.14135)
**Autores:** Tri Dao et al. (Stanford)  
**Fecha:** May 2022

**Aportes principales:**
- **IO-awareness:** Reduce transfers entre GPU HBM y on-chip SRAM
- **Tiling:** Divide Q, K, V en tiles que caben en SRAM
- **Resultados:**
  - 15% speedup en BERT-large (seq=512)
  - 3× speedup en GPT-2 (seq=1K)
  - 2.4× speedup en Long Range Arena (seq=1K-4K)
  - Permite seq lengths de 16K-64K

**Aplicabilidad a MUD:**
- ⚠️ **GPU-only:** MUD ahora es CPU-first (Vulkan opcional)
- 💡 **Idea para Vulkan:** Implementar FlashAttention en compute shaders
- 💡 **CPU variant:** Explorar tiling para cache L2/L3 (similar principio IO-aware)
- 📊 **Impacto potencial:** 2-3× speedup en atención para seq>1K

**Implementación sugerida:**
```rust
// Pseudo-código para CPU tiling
const TILE_SIZE = 64; // Cabe en L2 cache
for i in (0..seq_len).step_by(TILE_SIZE) {
    for j in (0..seq_len).step_by(TILE_SIZE) {
        // Load Q[i:i+TILE], K[j:j+TILE] to L2
        // Compute attention tile
        // Write result
    }
}
```

---

### 4. Attention Free Transformer (arXiv:2105.14103)
**Autores:** Shuangfei Zhai et al. (Meta AI)  
**Fecha:** May 2021

**Aportes principales:**
- **AFT (Attention Free Transformer):** Elimina dot-product self-attention
- **Mecanismo:** `output = Q * (K ⊙ position_bias) * V` (element-wise)
- **Complejidad:** O(n·d) vs O(n²·d) de attention tradicional
- **Variantes:**
  - AFT-local: Ventanas locales
  - AFT-conv: Weight sharing espacial

**Aplicabilidad a MUD:**
- ⚠️ **Arquitectural:** Requiere cambiar arquitectura del modelo
- ❌ **No compatible:** Modelos existentes usan attention tradicional
- 💡 **Futuro:** Explorar AFT para nuevos modelos desde scratch
- 📊 **Trade-off:** Menos memoria vs posible pérdida de calidad en tareas largas

---

### 5. Co-Designing Model Architectures with Hardware (arXiv:2401.14489)
**Autores:** Quentin Anthony et al.  
**Fecha:** Jan 2024

**Aportes principales:**
- **Model shape optimization:** Ajustar hidden_size, num_heads, num_layers para hardware target
- **Guidelines para GPUs:**
  - hidden_size múltiplo de 64-128 (para tensor cores)
  - num_heads que divide hidden_size uniformemente
  - Evitar shapes que causan padding en GEMM
- **Resultados:** 39% más throughput con mismo número de parámetros

**Aplicabilidad a MUD:**
- ✅ **Ya aplicado:** BitNet tiene hidden_size=2560 (múltiplo de 64)
- ✅ **MoE alignment:** Expert count power-of-2 (8 expertos)
- 💡 **Mejora:** Optimizar para CPU cache lines (64 bytes)
  - Alinear estructuras a 64 bytes (ya hecho con `#[repr(align(64))]`)
  - Padding explícito para evitar false sharing

**Recomendación específica:**
```rust
// Actual: 64-byte alignment
#[repr(align(64))]
pub struct AlignedBuffer { ... }

// Optimización adicional: prefetching
#[inline]
fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}
```

---

## Síntesis: Acciones Prioritarias para MUD

### 🔴 Crítico (esta semana)
1. **Fix bug de corrupción de logits** (`BUG_LOGITS_CORRUPTION.md`)
   - Sin esto, no hay inference usable
   - Bloquea todo progreso con BitNet b1.58

### 🟡 Alto (próximas 2 semanas)
2. **FlashAttention para Vulkan**
   - Implementar tiling en compute shaders
   - Benchmark vs atención actual
   - Esperado: 2-3× speedup para seq>1K

3. **CPU tiling para atención**
   - Adaptar principio IO-aware para CPU
   - Tile size = 64-128 (cabe en L2)
   - Esperado: 1.5-2× speedup

### 🟢 Medio (próximo mes)
4. **BitLinear converter**
   - Soporte oficial para exportar desde PyTorch
   - Documentar proceso de conversión b1.58

5. **Hardware co-design guidelines**
   - Documentar optimal model shapes para CPU
   - Agregar validación en converter

### 🔵 Bajo (exploratorio)
6. **AFT experiments**
   - Train modelo pequeño desde scratch con AFT
   - Evaluar trade-off calidad/velocidad

---

## Bibliografía Completa

| Paper | arXiv | Citas | Estado |
|-------|-------|-------|--------|
| BitNet | 2310.11453 | 500+ | ✅ Implementado (inference) |
| BitNet b1.58 | 2402.17764 | 1200+ | ✅ Modelo convertido |
| FlashAttention | 2205.14135 | 3000+ | 🔄 Pendiente (Vulkan) |
| AFT | 2105.14103 | 800+ | 🔮 Futuro |
| Co-Design | 2401.14489 | 150+ | ✅ Parcialmente aplicado |

---

## Notas Adicionales

### Técnicas No Investigadas (para próxima)
- **Speculative decoding:** Draft + verify con modelo pequeño
- **KV cache compression:** Quantize KV cache a INT8/FP4
- **MoE load balancing:** Auxiliary loss para expert routing
- **Activation checkpointing:** Trade-off memoria vs recomputación

### Hardware Emergente
- **BitNet-specific ASICs:** En desarrollo (Microsoft)
- **Ternary ALUs:** 16× más eficientes que FP16 MACs
- **Recomendación:** Mantener MUD agnóstico para portabilidad