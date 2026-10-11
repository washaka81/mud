# PLAN MAESTRO: ARQUITECTURA HÍBRIDA CPU+VULKAN PARA ELUT+JEPA

## 🎯 Objetivo Estratégico
Maximizar **throughput** (Vulkan/iGPU) + **fidelidad semántica** (CPU/AVX2) minimizando latencia en hardware con memoria unificada.

---

## 1. ARQUITECTURA DE EJECUCIÓN

### Pipeline Híbrido Asíncrono

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORIA RAM (16GB)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Pesos ELUT   │  │ Activaciones │  │ Acumuladores     │  │
│  │ 2 bits/peso  │  │ int8         │  │ int32 (buffer)   │  │
│  └──────▲───────┘  └──────▲───────┘  └────────┬─────────┘  │
└─────────│─────────────────│───────────────────│────────────┘
          │                 │                   │
          │ Vulkan Read     │ Vulkan Write      │ CPU Read
          ▼                 │                   ▼
┌──────────────────┐       │          ┌─────────────────────┐
│  VULKAN (iGPU)   │       │          │   CPU (Núcleos P)   │
│  Kernel ELUT     │       │          │  - LayerNorm fp32   │
│  MatMul Ternario │       │          │  - JEPA Predictor   │
│  (bitfieldExtract)───────┘          │  - Corrección fp32  │
│  workgroup_size=256                 │  - VICReg loss      │
└──────────────────┘                  └─────────────────────┘
```

---

## 2. DESGLOSE MATEMÁTICO POR CAPA

### A. **CPU: Preparación (Pre-MatMul)**
```rust
// Capa N: Token → Activaciones
1. LayerNorm(x) → x_norm (fp32)
2. Cuantización: xq = round(x_norm * scale) → int8
3. Transferencia a buffer Vulkan (staging buffer)
```

### B. **Vulkan: MatMul Ternario ELUT**
```glsl
// Compute Shader (ELUT kernel)
layout(std430) buffer Weights { uint packed_weights[]; }; // 2 bits/peso
layout(std430) buffer Activations { int8_t xq[]; };
layout(std430) buffer Output { int32_t accumulators[]; };

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    
    // Desempaquetado SIMD en GPU
    uint weight_packed = packed_weights[(row * dim + col) / 16];
    uint shift = ((row * dim + col) % 16) * 2;
    int8_t w = int8_t((weight_packed >> shift) & 0x3) - 1; // {-1, 0, 1}
    
    // Acumulación
    accumulators[row] += int32_t(xq[col]) * w;
}
```

### C. **CPU: Corrección JEPA (Post-MatMul)**
```rust
// Capa N: Acumuladores → Salida Corregida
1. De-cuantización: y = accumulators * scale_y → fp32
2. Cálculo JEPA:
   - Predictor latente: z_hat = predictor(z_context)
   - Energy constraint: E = ||z - z_hat||²
   - Corrección: y_final = y - λ * ∇E
3. Activación no-lineal (SiLU/GELU)
```

---

## 3. ESTRATEGIA DE SINCRONIZACIÓN

### Modelo Asíncrono con Double Buffering

```rust
struct AsyncPipeline {
    // CPU threads
    jepe_predictor_thread: Thread,
    fusion_thread: Thread,
    
    // Vulkan resources
    command_buffers: [VkCommandBuffer; 2], // Double buffer
    fences: [VkFence; 2],
    semaphores: [VkSemaphore; 2],
    
    // Shared memory
    staging_buffers: [StagingBuffer; 2],
    accumulators: [AccumulatorBuffer; 2],
}

impl AsyncPipeline {
    fn execute_layer(&mut self, frame_idx: usize) {
        // Frame N: CPU prepara activaciones
        self.prepare_activations(&self.staging_buffers[frame_idx]);
        
        // Frame N-1: Vulkan completa MatMul
        vkWaitForFences(self.fences[(frame_idx - 1) % 2]);
        
        // Frame N-1: CPU aplica JEPA sobre resultado Vulkan
        self.apply_jepe_correction(&self.accumulators[(frame_idx - 1) % 2]);
        
        // Frame N: Vulkan ejecuta MatMul (asíncrono)
        vkQueueSubmit(vulkan_queue, 
                     &self.command_buffers[frame_idx],
                     &self.semaphores[frame_idx]);
    }
}
```

---

## 4. DECISIÓN ARQUITECTÓNICA: TIPO DE JEPA

### ✅ **Recomendado: Predictor Latente Continuo**

**Razones:**

| Métrica | Energy Filter | Predictor Latente |
|---------|--------------|-------------------|
| **Latencia CPU** | Baja (1 pase) | Media (2 pases) |
| **Fidelidad Semántica** | Media (corrector reactivo) | **Alta (modelo generativo)** |
| **Control Alucinaciones** | Limitado (post-hoc) | **Proactivo (predice contexto)** |
| **Paralelización** | Serial con MatMul | **Paralelo con MatMul** |
| **VICReg compatibility** | No | **Sí (nativo)** |

### Implementación del Predictor Latente

```rust
struct JEPAredictor {
    // Contexto: últimos N tokens (embedding space)
    context_window: Vec<f32>, // [batch, seq_len, hidden_dim]
    
    // Predictor: MLP ligero en CPU
    predictor_net: MLP, // 2 capas, hidden=512, activ=GELU
    
    // VICReg loss components
    variance_loss: f32,
    covariance_loss: f32,
    invariance_loss: f32,
}

impl JEPAredictor {
    fn predict(&self, context: &[f32]) -> Vec<f32> {
        // Predice el embedding del siguiente token
        self.predictor_net.forward(context)
    }
    
    fn compute_vicreg(&self, z_true: &[f32], z_pred: &[f32]) -> f32 {
        // VICReg = λ₁*Var + λ₂*Cov + λ₃*Inv
        let var = self.variance_penalty(z_pred);
        let cov = self.covariance_penalty(z_pred);
        let inv = (z_true - z_pred).pow(2).mean();
        
        0.3 * var + 0.3 * cov + 0.4 * inv
    }
    
    fn apply_correction(&mut self, y: &mut [f32], z_context: &[f32]) {
        let z_pred = self.predict(z_context);
        let gradient = self.compute_vicreg_gradient(&z_pred);
        
        // Corrección en espacio latente
        for i in 0..y.len() {
            y[i] -= 0.1 * gradient[i]; // λ=0.1 (hyperparameter)
        }
    }
}
```

---

## 5. PLAN DE IMPLEMENTACIÓN

### **Fase 1: Infraestructura Vulkan** (Semana 1-2)
```rust
// tools/vulkan_elut_kernel.rs
struct VulkanELUTKernel {
    instance: vk::Instance,
    device: vk::Device,
    compute_queue: vk::Queue,
    pipeline: vk::Pipeline,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl VulkanELUTKernel {
    fn matmul(&mut self, weights: &[u8], activations: &[i8]) -> Vec<i32>;
    fn sync_with_cpu(&mut self, fence: vk::Fence);
}
```

### **Fase 2: Operador JEPA en CPU** (Semana 2-3)
```rust
// src/jepa/predictor.rs
struct ContinuousLatentPredictor {
    mlp: MLP,
    vicreg_lambda: f32,
    context_buffer: RingBuffer<f32>,
}

impl ContinuousLatentPredictor {
    fn new(hidden_dim: usize) -> Self;
    fn forward(&mut self, x: &[f32]) -> Vec<f32>;
    fn compute_correction(&self, y: &[f32]) -> Vec<f32>;
}
```

### **Fase 3: Integración Híbrida** (Semana 3-4)
```rust
// src/hybrid_layer.rs
struct HybridBitLinear {
    vulkan_kernel: VulkanELUTKernel,
    jepe_predictor: ContinuousLatentPredictor,
    scale: f32,
    bias: Option<Vec<f32>>,
}

impl HybridBitLinear {
    fn forward(&mut self, x: &[f32]) -> Vec<f32> {
        // 1. LayerNorm + cuantización (CPU)
        let xq = self.quantize(x);
        
        // 2. MatMul ternario (Vulkan, asíncrono)
        let accum = self.vulkan_kernel.matmul(&self.weights, &xq);
        
        // 3. De-cuantización + JEPA (CPU)
        let y = self.dequantize(&accum);
        let y_corr = self.jepe_predictor.apply_correction(y);
        
        // 4. Bias + activación
        self.activation(y_corr)
    }
}
```

### **Fase 4: Optimización AVX2** (Semana 4-5)
```rust
// src/cpu_kernels/avx2_elut.rs
#[target_feature(enable = "avx2")]
unsafe fn unpack_and_multiply_avx2(
    packed_weights: &[u8],
    activations: &[i8],
) -> Vec<i32> {
    // Usar _mm256_loadu_si256 para cargar 256 bits
    // _mm256_and_si256 para máscaras
    // _mm256_srli_epi16 para shifts
    // _mm256_maddubs_epi16 para multiplicación
}
```

---

## 6. MÉTRICAS OBJETIVO

| Métrica | Objetivo | Hardware |
|---------|---------|----------|
| **Time-to-First-Token** | <50ms | i7-1260P |
| **Tokens/segundo** | >100 tok/s | Iris Xe + CPU |
| **Ancho de banda efectivo** | >40 GB/s | Memoria unificada |
| **Precisión (vs fp16)** | >99.5% | JEPA VICReg |
| **Consumo energético** | <15W | Núcleos E + iGPU |

---

## 7. RIESGOS Y MITIGACIÓN

| Riesgo | Mitigación |
|--------|------------|
| **Overhead de sincronización Vulkan** | Double buffering + command buffers pre-grabados |
| **Thread Director mueve hilos a núcleos E** | Affinity mask + priority real-time |
| **Memory contention CPU-iGPU** | Alinear buffers a 64 bytes + prefetching |
| **Precisión JEPA degrada throughput** | Ejecutar JEPA en hilo separado (paralelo) |

---

## 8. COMANDO DE IMPLEMENTACIÓN INMEDIATA

```bash
# Estructura de directorios
mkdir -p src/{vulkan,jepa,hybrid_layers}
mkdir -p shaders/compute

# Crear shader ELUT base
cat > shaders/compute/elut_matmul.comp << 'EOF'
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer Weights { uint packed_weights[]; };
layout(std430, binding = 1) buffer Activations { int activations[]; };
layout(std430, binding = 2) buffer Output { int results[]; };

layout(push_constant) uniform PushConstants {
    uint dim_in;
    uint dim_out;
} push;

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (row >= push.dim_out || col >= push.dim_in) return;
    
    int acc = 0;
    for (uint k = 0; k < push.dim_in; k++) {
        uint idx = row * push.dim_in + k;
        uint packed = packed_weights[idx / 16];
        uint shift = (idx % 16) * 2;
        int w = int((packed >> shift) & 0x3) - 1;
        acc += activations[k] * w;
    }
    
    results[row * push.dim_in + col] = acc;
}
EOF
```

---

## 9. REFERENCIAS TÉCNICAS

### Hardware Objetivo
- **CPU**: Intel i7-1260P (4 núcleos P + 8 núcleos E)
- **iGPU**: Intel Iris Xe (80 EUs, 1.4 GHz max)
- **Memoria**: 16GB LPDDR4X, dual-channel
- **Instrucciones**: AVX2, FMA3

### Dependencias Rust
```toml
[dependencies]
ash = "0.38" # Vulkan bindings
gpu-alloc = "0.9" # Gestión de memoria GPU
gpu-descriptor = "0.2" # Descriptor sets

[features]
hybrid = ["ash", "gpu-alloc", "gpu-descriptor"]
avx2 = []
```

---

**Documento creado**: Junio 2026  
**Estado**: Plan Maestro Aprobado  
**Próximo Hito**: Fase 1 - Infraestructura Vulkan