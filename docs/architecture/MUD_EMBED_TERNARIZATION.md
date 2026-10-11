---
lang: es
---

# Ternarización de Embeddings — Técnica, Resultados y Propuesta

> **Estado:** Implementado (prototipo) — `tools/embed_ternarize.rs`
> **Modelo de prueba:** SmolLM2-135M (`models/smolmud_tern_emb.mud`)

## 1. Técnica

### Row-wise AbsMean + Scale u8

Cada fila del embedding table (un vector por token) se ternariza independientemente:

```
for each row r in embedding:
    scale[r] = mean(abs(row))        # promedio de valores absolutos
    ternary[r][j] = clamp(round(row[j] / scale[r]), -1, +1)
```

Las escalas se empaquetan como **u8** con mapeo lineal:

```
scale_u8 = round((scale_f32 - scale_min) / (scale_max - scale_min) * 255)
```

Esto da un overhead de **1 byte por fila** (ej: 48 KB para vocab=49k).

### Variantes evaluadas (descartadas)

| Variante | Cos-sim | Problema |
|----------|:-------:|----------|
| Global absmean (único scale) | 0.832 | Fila con scale pequeño se aplana a 0 |
| LSQ (Learned Step Size) | 0.784 | Scale óptimo para CNN, no para embedding |
| Optimal LS per-row | 0.863 | Marginal vs absmean (0.863 vs 0.857 con u8) |

**Conclusión:** Row-wise absmean con 8 bits de escala es el punto dulce calidad/complejidad.

## 2. Resultados Cuantitativos (SmolLM2-135M, 49k × 576)

### Calidad (10,000 filas)
```
Cosine similarity: mean = 0.857, median = 0.872
MSE:              mean = 0.0167
Rows con cos > 0.9:  1,024 / 10,000 (10%)
Rows con cos > 0.8:  8,911 / 10,000 (89%)
```

### Compresión
| Formato | Tamaño | Ratio |
|---------|:------:|:-----:|
| FP32 original | 108.0 MB | 1× |
| FP16 original | 54.0 MB | 2× |
| Row-wise ternary (2.014 bits/param) | 6.8 MB | **15.9×** |
| Modelo total antes (FP32 emb + ternary weights) | 135 MB | — |
| Modelo total después (todo ternary) | 34 MB | **4.0×** |

### Distribución ternaria
```
+1: 33.97%   0: 33.21%   -1: 32.82%   (asimetría: 1.15%)
```
Distribución prácticamente perfecta para un modelo ternario balanceado.

## 3. Inferencia End-to-End

El modelo ternarizado (`models/smolmud_tern_emb.mud`) carga y ejecuta inferencia sin errores:

- ✅ Sin crash
- ✅ Sin segfault
- ✅ Sin NaN
- ✅ IQ score idéntico (8.9)
- ✅ Consumo de memoria: 2.7 GB (vs 3.0 GB con emb FP32)

## 4. Propuesta para Conversión Exitosa

### 4.1 Integración en `universal_converter`

Añadir flag `--ternarize-emb` que durante la conversión Safetensors → `.mud`:

1. Detecta `token_embd.weight` (shape [vocab, hidden])
2. Aplica row-wise absmean ternarization
3. Almacena como `MudTensorType::Ternary2Bit`
4. Guarda escalas en metadata global:
   ```
   embed_ternarized = "row_absmean"
   embed_scale_min = "0.056"
   embed_scale_max = "0.207"
   embed_scale_bits = "8"
   embed_scales = [u8 × vocab]  → almacenar como tensor separado o metadata
   ```
5. Inference engine lee las escalas y dequantiza on-the-fly en `embed()`.

### 4.2 Inference (`inference.rs`): Carga de Embeddings Ternarizados

Actualmente `embed()` lee directamente de `token_embd.weight` como Float32. Para ternarizados:

```rust
// embed() actual:
fn embed(&self, ws: &mut InferenceWorkspace, token: usize) {
    let src = &self.model.embed_weight[token * hidden .. (token + 1) * hidden];
    ws.x.copy_from_slice(src);
}

// embed() con embeddings ternarizados:
fn embed_ternary(&self, ws: &mut InferenceWorkspace, token: usize, scales: &[f32]) {
    let src = &self.model.embed_ternary[token * hidden .. (token + 1) * hidden];
    let scale = scales[token];
    for i in 0..hidden {
        let q = ternary_to_f32(src[i]); // -1, 0, +1
        ws.x[i] = q * scale;
    }
}
```

### 4.3 Calidad vs Tamaño: Trade-off

| Técnica | Tamaño emb | Cos-sim | Training? | Complexidad |
|---------|:----------:|:-------:|:---------:|:-----------:|
| FP32 (actual) | 108 MB | 1.0 | No | 0 |
| FP16 | 54 MB | 1.0 | No | 0 |
| Row-wise absmean | 6.8 MB | 0.857 | No | Baja |
| Row-wise + finetune | 6.8 MB | ~0.95 | QAT (1 paso) | Media |
| DLT + OFF (TernaryLLM) | 6.8 MB | ~0.98 | QAT + KD | Alta |

**Recomendación:** Implementar row-wise absmean como default en el converter. Para producción, añadir QAT opcional con `--calibrate` usando un corpus pequeño (~1000 tokens).

## 5. Cómo probar

```bash
# Análisis de embedding actual
cargo run --release --bin embed_audit models/smolmud.mud

# Ternarizar embedding
cargo run --release --bin embed_ternarize models/smolmud.mud models/smolmud_tern_emb.mud

# Probar inferencia
cargo run --release --bin forge_llm models/smolmud_tern_emb.mud
```

## 6. Enlaces

- `tools/embed_audit.rs` — Herramienta de análisis
- `tools/embed_ternarize.rs` — Herramienta de ternarización
- `docs/MUD_ROADMAP.md` — Roadmap con embedding ternarization track
- `docs/MUD_AUDIT_LATEST.md` — Estado de bugs y features
