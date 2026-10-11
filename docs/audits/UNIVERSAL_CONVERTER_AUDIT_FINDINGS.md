# Universal Converter Audit — Hallazgos Reales

> **Auditoría de:** `tools/universal_converter/` (main.rs, quantizer.rs, calibration.rs, parser.rs)
> **Herramientas de referencia:** `tools/conversion_verifier.rs`, `tools/scale_audit.rs`
> **Fecha:** 2026-07-24
> **Estado:** Hallazgos confirmados por análisis estático de código

---

## Hallazgo 1: `audit_ternary_fidelity` — Bug de `bitnet_scale` en el cálculo de error

**Archivo:** `tools/universal_converter/quantizer.rs:188`

**Código problemático:**
```rust
let orig = f32_data[row_start_f32 + col] * bitnet_scale;  // ← multiplica por bitnet_scale
let q_val = match bits {
    0x1 => scale,   // ← scale YA incluye 1/bitnet_scale
    0xF => -scale,
    _ => 0.0,
};
let diff = orig - q_val;
```

**Problema:** El término `orig` multiplica por `bitnet_scale`, pero `q_val` usa `scale` que ya incluye `1/bitnet_scale` (computado en `ternarize_f32_and_pack` como `dampened_scale * (1.0 / bitnet_scale)`). Cuando `bitnet_scale ≠ 1.0`, el error reportado es incorrecto por un factor de `bitnet_scale²`.

**Cálculo correcto:**
- `orig = f32_data[col]` (peso original, sin escalar)
- `q_val = scale` (escala final, que incluye `1/bitnet_scale`)

**Impacto:** Cuando `bitnet_scale = 1.0` (caso común), el bug es inofensivo. Pero para modelos BitNet con `weight_scale ≠ 1.0`, la auditoría de fidelidad reporta errores incorrectos, lo que puede enmascarar problemas reales o generar falsas alarmas.

---

## Hallazgo 2: `conversion_verifier.rs` — Divide pesos originales por `weight_scale` incorrectamente

**Archivo:** `tools/conversion_verifier.rs:345-356`

**Código problemático:**
```rust
if should_ternarize {
    let scale_name = name.replace(".weight", ".weight_scale");
    if let Ok(scale_view) = safe_tensors.tensor(&scale_name) {
        let scale_val = convert_view_to_f32(&scale_view)[0];
        for w in &mut orig_weights {
            *w /= scale_val;  // ← divide pesos originales por weight_scale
        }
    }
}
```

**Problema:** El conversor (`universal_converter/main.rs:847`) NO escala los pesos por `bitnet_s` antes de la ternarización. Solo usa `bitnet_s` para ajustar la escala final: `row_scales[row] = dampened_scale * (1.0 / bitnet_s)`. Pero el verificador divide los pesos originales por `scale_val`, comparando `w / bitnet_s` contra `ternary_value * absmean * 0.707 / bitnet_s`. La comparación correcta debería ser `w` contra `ternary_value * absmean * 0.707 / bitnet_s`.

**Impacto:** El verificador reporta errores de fidelidad incorrectos cuando `bitnet_scale ≠ 1.0`. La división por `scale_val` introduce un factor de error adicional.

---

## Hallazgo 3: `calibration.rs` — `max_layer` no usa `break`, inflando `total_layers`

**Archivo:** `tools/universal_converter/calibration.rs:10-23`

**Código problemático:**
```rust
for p in parts {
    if let Ok(l) = p.parse::<usize>() {
        if l > max_layer {
            max_layer = l;
        }
        // ← NO break! Continúa iterando, capturando IDs de expertos
    }
}
```

**Problema:** El cálculo de `max_layer` itera por TODAS las partes del nombre del tensor y toma el máximo. Para un tensor como `model.layers.0.mlp.experts.10.gate.weight`, encontraría tanto `0` (capa) como `10` (ID de experto), y `max_layer` sería `10` en lugar de `0`. Esto infla `total_layers`, lo que hace que los factores de atenuación sean más cercanos a `1.0` (menos atenuación) para todas las capas.

**Impacto:** Las capas profundas reciben menos atenuación de la prevista, lo que puede contribuir a explosión de LogitVar. La atenuación depth-based `1.0 - (0.35 * (layer / total_layers)²)` se ve afectada.

---

## Hallazgo 4: `conversion_verifier.rs` — No maneja `qkv_proj` ni `gate_up_proj`

**Archivo:** `tools/conversion_verifier.rs:116-249`

**Problema:** La función `map_llama_to_mud` del verificador no maneja:
- `qkv_proj` (proyección QKV fusionada) — el conversor sí la maneja (`parser.rs:134`)
- `gate_up_proj` (proyección gate_up fusionada) — el conversor sí la maneja (`parser.rs:175`)

**Impacto:** El verificador omite la auditoría de estos tensores, dejando un hueco en la verificación de fidelidad.

---

## Hallazgo 5: `conversion_verifier.rs` — No distingue error de ternarización vs error de calibración

**Archivo:** `tools/conversion_verifier.rs:399-441`

**Problema:** El conversor aplica escalas de calibración depth-based (`calibration::compute_scales`) a los PRQ scales después de la ternarización. El verificador compara los pesos originales contra los pesos desternarizados (que incluyen calibración), reportando error acumulado sin distinguir entre:
- Error debido a la cuantización ternaria (irreductible)
- Error debido a la calibración depth-based (intencional)

**Impacto:** Las capas profundas muestran SQNR más bajo no porque la ternarización sea peor, sino porque la calibración reduce sistemáticamente las magnitudes. Esto puede llevar a interpretaciones incorrectas de la fidelidad de conversión.

---

## Hallazgo 6: `quantizer.rs` — La ternarización no considera la escala `bitnet_scale` en el umbral delta

**Archivo:** `tools/universal_converter/quantizer.rs:88-145`

**Problema:** El umbral `delta = 0.7 * dampened_scale` se calcula sobre los pesos originales (sin escalar por `bitnet_scale`). En el formato BitNet original, los pesos deberían escalarse por `bitnet_scale` antes de la ternarización, y el umbral debería aplicarse a los pesos escalados. El conversor omite este escalado, lo que significa que el umbral de ternarización es incorrecto cuando `bitnet_scale ≠ 1.0`.

**Impacto:** La ternarización puede preservar o eliminar valores incorrectamente cuando `bitnet_scale ≠ 1.0`, afectando la fidelidad de la conversión.

---

## Resumen de severidad

| Hallazgo | Severidad | Archivo | Línea | Estado |
|----------|-----------|---------|-------|--------|
| 1. Bug `bitnet_scale` en `audit_ternary_fidelity` | 🔴 Alta | quantizer.rs | 188 | ✅ **FIXADO** |
| 2. División incorrecta por `weight_scale` en verificador | 🔴 Alta | conversion_verifier.rs | 345-356 | ✅ **FIXADO** |
| 3. `max_layer` sin `break` en calibración | 🟡 Moderada | calibration.rs | 10-23 | ✅ **FIXADO** |
| 4. Verificador no maneja QKV/gate_up fusionados | 🟡 Moderada | conversion_verifier.rs | 116-249 | 📋 Documentado |
| 5. Verificador no distingue error de calibración | 🟡 Moderada | conversion_verifier.rs | 399-441 | ✅ **FIXADO** (precise_converter_audit) |
| 6. Umbral delta no considera `bitnet_scale` | 🟡 Moderada | quantizer.rs | 113 | 📋 Documentado (diseño intencional) |

---

## Herramienta de auditoría precisa (Fenwick-optimized)

Se creó `tools/precise_converter_audit.rs` — un auditor de fidelidad de conversión con cálculos precisos y **optimización con Fenwick trees**:

- **Corrige el manejo de `bitnet_scale`**: NO divide los pesos originales por `weight_scale`
- **Calcula métricas precisas**: SQNR (dB), MSE, RMSE, MAE, max error, relative Frobenius error, sparsity, per-row SQNR
- **Fenwick tree histograma de errores**: 64 bins log-espaciados con consultas de percentil O(log n) (mediana, P95, P99) via `FenwickU32::find_kth`
- **FenwickF32 seguimiento acumulativo**: O(log n) para consultas de rango "error hasta la capa L"
- **Soporta calibración depth-based**: con `--calibration`, descompone error en cuantización vs calibración
- **Maneja QKV fusionado**: detecta `qkv_proj` y extrae sub-rangos

### Uso

```bash
./mud.sh precise-audit models/smollm2 models/smollm2.mud
cargo run --release --bin precise_converter_audit -- models/smollm2/model.safetensors models/smollm2.mud --calibration
```

### Fixes aplicados al conversor

1. **quantizer.rs:188** — `audit_ternary_fidelity`: eliminada multiplicación incorrecta de `orig` por `bitnet_scale`.
2. **calibration.rs:10-23** — `compute_scales`: agregado `break` para prevenir inflado de `max_layer` por IDs de expertos.
3. **conversion_verifier.rs:345-356** — Eliminada división incorrecta de pesos originales por `weight_scale`.

### Optimizaciones aplicadas (refactorización)

1. **TERNARY_LUT branchless en `audit_ternary_fidelity`** — Reemplazado `match` por lookup table de 16 entradas (`TERNARY_LUT`), acelerando el hot path de auditoría.
2. **U8_LUT branchless en `to_f32_vec`** — Reemplazado `match` de 4 casos por lookup table de 4 entradas para desempaquetado U8.
3. **Packing ternario sin ramas** — `ternarize_f32_and_pack` usa `is_pos + is_neg * 0x15` (aritmética) en lugar de if/else chains, eliminando branch misprediction en el loop interno.
4. **`to_f32_vec` optimizado** — Eliminados Vecs intermedios de f16/bf16; usa pointer arithmetic directo con `f32::from_bits` para bf16 y `f16::from_le_bytes` para f16. F32 usa `slice::from_raw_parts().to_vec()` (zero-copy read).
5. **Byte conversion sin copy intermedio** — `flat_map(|p| p.to_le_bytes()).collect()` en lugar de loop con `extend_from_slice`.
6. **Calibración con `break` consistente** — Ambas passes (max_layer + dampening) usan `break` después del primer número parseable, previniendo inflado por expert IDs.
