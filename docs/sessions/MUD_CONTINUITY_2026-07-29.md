# Continuidad — Mañana (2026-07-29)

**Último estado:** Entrenamiento activo, step 104/3325 (2.7%, epoch 1/25, ~27 min)

---

## Estado del Entrenamiento

| Campo | Valor |
|-------|-------|
| **Proceso** | `run_trainer` PID 122476 (activo) |
| **Step** | 104/3325 |
| **Epoch** | 1/25 |
| **Block** | 104/152 |
| **Progress** | 2.7% |
| **Elapsed** | 1621.6s (~27 min) |
| **ETA (full)** | ~14.4 horas (3325 × 15.8s) |

### Métricas latest (step 104)
```
loss=7.26  ppl=1420.8  lr=0.000500  tok/s=4.1
grad_norm=2935.22  varh=4.905  varj=0.034  jepa=-0.007  σ=49.8%  cog=369.2
conf=0.07%  prog=2.7%
```

### Salud ✅
- **grad_norm:** ~2900-3000 (estable, fix funcionando)
- **varh:** 4.90 (activaciones saludables, no colapsado)
- **JEPA:** σ≈50% (neutro, equilibrio)
- **Loss:** oscilante 5.48-7.77 (esperado en modelo recién convertido)
- **tok/s:** 4.1 (estable)

---

## Comandos para Continuar Mañana

### 1. Ver métricas en vivo
```bash
tail -f /home/ale/proyectos/forge_llm/mud_train_metrics.log
```

### 2. Ver progreso del TUI (si está corriendo)
```bash
# En otra terminal:
cargo run --release --bin train_telemetry
```

### 3. Ver logs del circuito (si aplica)
```bash
tail -f /home/ale/proyectos/forge_llm/logs/circuit.log
```

### 4. Verificar que el proceso sigue activo
```bash
ps aux | grep run_trainer | grep -v grep
```

### 5. Si el entrenamiento se detiene, reanudar
```bash
cd /home/ale/proyectos/forge_llm
./mud.sh train models/smollm2.mud --epochs 25
```
(El trainer reanuda automáticamente desde el checkpoint si existe)

---

## Próximos Pasos (para mañana)

### A. Monitorear convergencia (~500 steps)
A los ~500 steps (1.5 horas desde ahora), verificar:
- ¿El loss muestra tendencia de descenso?
- ¿El grad_norm se mantiene estable (< 5000)?
- ¿varh se mantiene > 3.0?

**Si loss no desciende a los 500 steps:**
```bash
# Matar proceso actual y reiniciar con LR más alto
kill $(pgrep -f run_trainer)
MUD_QAT_LR=0.03 ./mud.sh train models/smollm2.mud --epochs 25
```

### B. Verificar checkpoint
```bash
# Verificar que los pesos se están guardando
ls -la weights/checkpoints/
# Ver hash de cambio
grep "hash\|ΔW\|DW" mud_train_metrics.log | tail -5
```

### C. Análisis de loss trend
```bash
# Extraer loss promedio de ventanas
awk '/\[TELEM\]/ {print $0}' mud_train_metrics.log | \
  awk '{sum+=$5; n++} END {print "avg loss:", sum/n, "over", n, "steps"}'
```

---

## Cambios Aplicados Hoy

| Archivo | Cambio | Estado |
|---------|--------|--------|
| `mud.sh` línea 277-279 | `MAX_CHUNKS=0` → `MAX_CHUNKS=unlimited` en display | ✅ Aplicado |
| `src/mud/corpus_trainer/train_step.rs` | Sanitización + clipping de gradientes | ✅ Aplicado (462 tests pass) |
| `src/mud/constants.rs` | `DEFAULT_GRAD_CLIP=1.0` + `grad_clip_threshold()` | ✅ Aplicado |
| `docs/sessions/MUD_SESSION_REPORT_2026-07-28_VERIFICATION_PLAN.md` | Reporte de verificación completo | ✅ Creado |

---

## Comandos Útiles de Referencia

```bash
# Health check del modelo
./mud.sh health models/smollm2.mud

# Scale audit (detecta vocab collapse)
./mud.sh scale-audit models/smollm2.mud

# Full audit
cargo run --release --bin mud_full_audit -- models/smollm2.mud

# Loss certification (fast)
./mud.sh cert-loss models/smollm2.mud

# Tests
cargo test --lib -- --test-threads=2
cargo clippy --all-targets
```

---

## Recordatorios

1. **No interrumpir** el entrenamiento a menos que el loss se dispara (>15) o grad_norm > 10000
2. El loss oscilante es NORMAL para STE QAT en modelo recién convertido
3. El `MAX_CHUNKS=unlimited` (antes `0`) significa "procesar todo el corpus" — es correcto
4. El optimizer es `adam` (MUD_OPT=adam), no Muon/GaLore aunque el banner lo mencione
5. Si se quiere probar Muon: `MUD_OPT=muon ./mud.sh train models/smollm2.mud --epochs 25`
