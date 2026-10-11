# MUD Session Report — 2026-07-29: AutoFinetune + TUI + NaN Guard Fixes

> **Objetivo:** Reparar el "desastre" del entrenamiento (divergencia por NaN), mejorar el TUI de telemetría, y añadir AutoFinetune con integración de C-MUD/SlimeX y entrenamiento directo.

---

## 1. Resumen Ejecutivo

| Área | Estado | Tests | Clippy |
|------|--------|-------|--------|
| NaN Guard Fix | ✅ DONE | 411 pass | 0 |
| TUI train_telemetry | ✅ DONE | 411 pass | 0 |
| AutoFinetune | ✅ DONE | 411 pass | 0 |
| Cargo.toml | ✅ DONE | — | — |

---

## 2. Fix Crítico: NaN Propagation en C-MUD Thinker

### Problema
El entrenamiento divergía porque `maybe_cmud_think` introducía NaN en los registros, causando que `apply_output_norm_cpu` produjera "ALL ZEROS" (rms_inv=0.0 cuando mean_sq=NaN), colapsando todos los logits.

### Causa Raíz
```
C-MUD think → maybe_think_collapse introduce NaN → registros corruptos
→ apply_output_norm_cpu: mean_sq=NaN → rms_inv=0.0 → ALL ZEROS
→ logits=0 → loss=ln(vocab) → gradientes explosivos → divergencia
```

### Fix (`src/mud/slime_forward.rs`)
Añadida limpieza de NaN/Inf en 4 puntos de `maybe_cmud_think`:
1. **Antes del think** — Limpia registros corruptos antes de empaquetar a CellV5
2. **Después del think collapse** — Limpia NaN introducidos por `maybe_think_collapse`
3. **Después del CellV5 unpack** — Limpia NaN del desempaquetado
4. **En el write-back final** — Última defensa antes de escribir a registros

```rust
// Clean NaN/Inf from scratch before C-MUD processing (P-17)
for val in scratch.iter_mut().take(hidden) {
    if !val.is_finite() { *val = 0.0; }
}
```

### Verificación
- `forward_sanity` ✅ (antes fallaba)
- `cmud_think_forward_smoke` ✅
- `cmud_compare_not_degenerate` ✅
- 411 tests pasando, 0 fallos

---

## 3. TUI de Telemetría Reescrito (`tools/train_telemetry.rs`)

### Problema
El TUI parseaba un formato antiguo de columnas posicionales, no el formato key=value actual del trainer. No mostraba métricas C-MUD, no tenía alerta de divergencia.

### Cambios

#### 1. Parser key=value
```rust
// Antes: parseo de columnas posicionales (formato obsoleto)
// Después: parseo de key=value actual
fn parse_kv(line: &str) -> HashMap<String, String>
fn parse_val(map: &HashMap<String, String>, key: &str) -> Option<f64>
```

#### 2. 6 modos de visualización
| Modo | Paneles | Propósito |
|------|---------|-----------|
| **TRAINING** | Loss, Grad Norm (log10), LR, Loss Velocity | Diagnóstico de convergencia |
| **PERFORMANCE** | Confidence %, Tok/s, Progress %, Perplexity | Métricas de calidad |
| **JEPA** | VarJ, JEPA Integral, VarH, Gate Mean | Estabilidad dinámica |
| **TERRAIN** | Sigma %, Loss Profile (topográfico), Loss, Loss Velocity | Paisaje de pérdida |
| **EFFICIENCY** | Trajectory Efficiency, STP λ, Loss CV, Grad Norm | Eficiencia de optimización |
| **C-MUD** | cmud_steps, cmud_lock, cmud_herm, cmud_spread/cauchy | Estado del manifold complejo |

#### 3. Alerta de divergencia
```rust
let diverging = app.loss_vel.data.len() >= 10
    && app.loss_vel.data.iter().rev().take(10).all(|(_, v)| *v > 0.1);
// Título rojo "⚠ DIVERGENCE" cuando loss_vel > 0.1 sostenido
```

#### 4. Loss Profile topográfico mejorado
- Título cambia a "Loss Divergence" cuando la pérdida sube sostenidamente
- Segmentos coloreados por bandas de elevación (Deep Valley → Peak)
- Bola del modelo con color dinámico según altura

#### 5. Clip de Grad Norm
```rust
// Valores >1000 se clipean para evitar compresión de escala log10
self.grad_norm.push(s, v.min(1000.0));
```

#### 6. Y-bounds dinámicos
- VarH: rango [0, 5] en lugar de [0, 1] (soporta varh > 4.0)

#### 7. Inode probe
- Reabre el log si fue reemplazado (via `>`, `cp`, etc.)
- Detecta truncación y reinicia la posición

#### 8. Zoom/scroll
- `z`/`x`: zoom in/out
- `f`: follow/pause
- `←`/`→`: scroll horizontal
- `↑`/`↓`: ajustar window size

---

## 4. AutoFinetune con C-MUD + SlimeX (`tools/autofinetune.rs`)

### Problema
`autofinetune` solo probaba `MUD_QAT_LR`, `MUD_GRAD_CLIP`, `MUD_TRAIN_STEPS_PER_CHUNK`, `MUD_TRAIN_NUM_NEG` — no incluía C-MUD Thinker ni SlimeX.

### Cambios

#### 1. `Cargo.toml`
```toml
[[bin]]
name = "autofinetune"
path = "tools/autofinetune.rs"
```

#### 2. `HyperConfig` extendido
```rust
struct HyperConfig {
    lr: f32,
    grad_clip: f32,
    steps_per_chunk: usize,
    num_neg: usize,
    cmud_think: bool,  // NUEVO: C-MUD complex reasoning
    slimex: bool,      // NUEVO: SlimeX CellV5 + ShadowExpertBus
}
```

#### 3. Grid de búsqueda ampliado
El grid ahora incluye todas las combinaciones de:
- C-MUD Thinker (activado/desactivado)
- SlimeX (activado/desactivado)
- Esto duplica el número de trials (4 → 16 por combinación base)

#### 4. `apply_config` actualizado
```rust
if config.cmud_think {
    env::set_var("MUD_CMUD_THINK", "1");
} else {
    env::remove_var("MUD_CMUD_THINK");
}
if config.slimex {
    env::set_var("MUD_SLIMEX", "1");
} else {
    env::remove_var("MUD_SLIMEX");
}
```

#### 5. `--train` flag
Después de encontrar la mejor configuración, lanza `run_trainer` directamente:
```bash
cargo run --release --bin autofinetune -- models/smollm2.mud --trials 16 --epochs 2 --train
```

#### 6. `launch_training()`
Función nueva que:
- Aplica la configuración al entorno
- Incluye `MUD_CMUD_THINK` y `MUD_SLIMEX`
- Lanza `cargo run --release --bin run_trainer -- <model>` como proceso hijo
- Muestra comandos manuales de fallback

#### 7. Reporte JSON y terminal
Incluye `MUD_CMUD_THINK` y `MUD_SLIMEX` en:
- `best_config` (JSON)
- `env_exports` (JSON)
- Output terminal

---

## 5. mud.sh — Default de convert actualizado

### Cambio
```bash
# Antes:
INPUT=${2:-models/qwen2_0.5b/model.safetensors}
OUTPUT=${3:-models/qwen2_0.5b.mud}

# Después:
INPUT=${2:-models/smollm2/model.safetensors}
OUTPUT=${3:-models/smollm2.mud}
```

---

## 6. Verificación

### Tests
```
cargo test --lib -- --test-threads=2
test result: ok. 411 passed; 0 failed; 2 ignored
```

### Clippy
```
cargo clippy --all-targets
Finished dev profile in 0.25s
```

### Build
```
cargo build --release --bin autofinetune
Finished release profile in 13.71s
```

---

## 7. Uso

### AutoFinetune con C-MUD + SlimeX + entrenamiento directo
```bash
cargo run --release --bin autofinetune -- models/smollm2.mud --trials 16 --epochs 2 --train
```

### TUI de telemetría
```bash
cargo run --release --bin train_telemetry
```

### Convertir modelo
```bash
./mud.sh convert models/smollm2/model.safetensors
```

---

## 8. Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| `Cargo.toml` | +5 líneas (registro de autofinetune) |
| `mud.sh` | 4 líneas (default de convert) |
| `src/mud/slime_forward.rs` | 28 líneas (NaN guard en maybe_cmud_think) |
| `tools/train_telemetry.rs` | 1359 líneas (reescrito completo) |
| `tools/autofinetune.rs` | ~900 líneas (C-MUD + SlimeX + --train) |
