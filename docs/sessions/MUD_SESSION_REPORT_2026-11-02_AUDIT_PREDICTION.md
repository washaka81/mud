# MUD — Session Report 2026-11-02 (auditoría, atención y predicción)

**Estado:** sesión cerrada.  
**Alcance:** auditoría documental, correcciones P0, evaluación next-token, paralelización y atención del transformer.  
**Regla:** todo valor de esta sesión se ejecutó y se reporta con comando/salida.  
**Protocolo de trazas:** [`docs/architecture/MUD_TOKEN_PREDICTION_TRACE_PROTOCOL.md`](../architecture/MUD_TOKEN_PREDICTION_TRACE_PROTOCOL.md).

---

## 1. Ground truth final

| Verificación | Comando | Resultado |
|---|---|---|
| Tests lib release | `cargo test --lib --release -- --test-threads=2` | **475 passed / 0 failed / 0 ignored** (0.40 s) |
| Proptest | `cargo test --lib --release --features proptest -- --test-threads=2` | **482 passed / 0 failed / 0 ignored** |
| Clippy | `cargo clippy --all-targets --all-features -- -D warnings` | **0 warnings / 0 errors** |
| Env SSOT | `bash mud.sh env-audit` | **128 vars código · 51 shell · 134 docs · 6 retiradas · 0 divergencias** |
| Unsafe | `bash mud.sh unsafe-audit` | **536 unsafe · 0 sin justificar** |
| Batería CI | `bash mud.sh ci` | **`[L-12] Health battery complete.`** |
| Git target | `git ls-files forge_autograd/target` | **0** trackeados; 178 `D` staged |

---

## 2. Correcciones aplicadas

| ID | Cambio | Evidencia |
|---|---|---|
| C-01 | `SmartTokenizer` detecta `mud:weight_type` (canónico) y `weight_type` (legacy), casing insensible, precedencia canónica | 5 tests dedicados |
| C-02 | `MUD_DEBUG_LAYERS` y `MUD_WORM_HOTPLUG` documentadas; `env-audit` verde | `env-audit` exit 0 |
| C-03 | `.gitignore` cubre `**/target/`; 178 artefactos de `forge_autograd/target` purgados del índice | `git ls-files` 0 |
| C-04 | `prompts.txt` movido a `tests/data/`; P-18 con excepción explícita para el plan de raíz | `GEMINI.md` §2/§7 |
| C-05 | Comentario fantasma `mud_autotrainer` eliminado; R-61 parcial (11 tools) | `mud.sh` |
| C-06 | Tablas R-73 sincronizadas (plan maestro §12, GEMINI §0.3, `PLAN_EXEC`) | docs |
| — | `mHC` lookup acepta `blk.N.mhc_alpha` y `blk.N.mhc_alpha.weight` | `inference.rs` |
| — | **K gradiente = 0 corregido**: `tape.q_f32` se rellena post-RoPE | K grad `0 → 2.7e-2`; test de regresión |
| — | Flags de diagnóstico: bypass de capas, cero ramas, escalas, sub-norm, grad scale, deadzone, debug grad | `MUD_ENV_REFERENCE.md` |
| — | Herramientas nuevas: `next_token_eval`, `gemv_parity` | `Cargo.toml` |

---

## 3. Cálculos y paralelización

### 3.1 GEMV ASM vs escalar

`gemv_parity` sobre 7 tensores reales (Q/K/V/O y FFN):

```text
cosine = 1.000000
max_abs ≤ 1.7e-6
```

**No hay desviación del kernel GEMV.**

### 3.2 Paralelización

Misma prompt/modelo con `MUD_PCORE_THREADS=1/8/16`:

| Hilos | rank | NLL | entropía |
|---:|---:|---:|---:|
| 1 | 53410 | 15.650557 | 10.752252 |
| 8 | 53410 | 15.650557 | 10.752252 |
| 16 | 53410 | 15.650557 | 10.752252 |

**Resultados idénticos; sin carrera en el forward.**

### 3.3 Atención

- Pos0: `scores[0] = 1.000000`, `o_act[0..4] == v[0..4]`.
- **Bug encontrado:** `tape.q_f32` nunca se rellenaba en el forward; el backward de K lo usa (`grad_k = grad_o · Q`).
- **Fix:** copiar Q/K/V post-RoPE al tape (`slime_forward.rs:1069`).
- Evidencia: `[GRAD] L0 k max=0.000e0` → `2.706e-2`.
- Regresión: `test_forward_tape_populates_qkv_for_attention_backward` (forward + backward, exige `k_w_grad != 0`).

---

## 4. Valores predictivos

Herramienta: `next_token_eval` (top-1/5/10, MRR, NLL, PPL, p(correcto), rank, entropía).

### 4.1 Set factual (`tests/data/next_token_eval.txt`, 8 prompts)

| Modelo | top-1 | MRR | NLL | PPL | rank medio | entropía |
|---|---:|---:|---:|---:|---:|---:|
| `ale.mud` | 0/8 | 0.000372 | 12.0286 | 167,471.9 | 75,919 | 16.914 |
| `ale` QAT fuerte | 0/8 | 0.000053 | 11.8492 | 139,967.3 | 57,134 | 17.127 |
| `qwen_coder_0.5b` base | 0/8 | 0.000226 | 17.1201 | 27,236,038 | 54,280 | 1.815 |
| `qwen` QAT 64 pasos | 0/8 | 0.000016 | 15.3685 | 4,725,659 | 88,437 | 12.395 |

### 4.2 Copy task (`/tmp/copy_eval.txt`, 6 prompts)

| Modelo | top-1 | NLL | p(correcto) | entropía |
|---|---:|---:|---:|---:|
| `ale.mud` base | **0/6 (0 %)** | 12.081 | 7.85e-6 | 16.915 |
| `ale_qat_strong` (64 pasos, LR 0.1, deadzone 0) | **2/6 (33.33 %)** | 11.076 | 4.37e-5 | 17.127 |
| `ale_qat_512` (512 pasos, deadzone 0.5) | **2/6 (33.33 %)** | **9.814** | **3.35e-3** | 16.857 |

Aciertos concretos:

```text
[01] ctx=2 target=" Paris" rank=1 p=1.943e-4 margin=+1.8604 ✓
[02] ctx=2 target=" hello" rank=1 p=3.894e-5 margin=+0.1219 ✓
```

### 4.3 Movimiento real de pesos

| Modelo | Tensores cambiados | `.weight` ternarios | Nota |
|---|---:|---:|---|
| `ale_qat_strong` vs base | 105 | **53** | copy 2/6 |
| `ale_qat_512` vs base | 69 | 14 | copy 2/6 |
| `qwen_qat_strong` vs base | 198 | **30** | PPL 5.8× mejor |

**El warning `ΔW: ternary 0/N — no change this sync` es sync-level**, no un no-op de entrenamiento: el optimizer ya empaquetó en el tensor y el sync vuelve a empaquetar el mismo shadow. Para medir movimiento real se comparan hashes antes/después.

---

## 5. Conclusión

- Los cálculos ASM, la paralelización y la atención pos0 son correctos.
- Se corrigió un bug real de atención: K no recibía gradiente.
- El pipeline **sí puede predecir** en una tarea controlada de copia: **2/6 top-1 (33.33 %)** frente a 0/6 del base, con NLL 12.08→9.81.
- El modelo real `qwen_coder_0.5b` mejora PPL 5.8× y calibración con 64 pasos, pero sigue 0/8 factual y 0/6 copy; necesita QAT largo o mayor precisión.
- El set factual de `ale.mud` sigue 0/8 por capacidad del forge `nano` (6 capas, hidden 128), no por ASM.

---

## 6. Pendiente

1. **Fix del reporte ΔW**: que mida el delta contra el checkpoint anterior, no contra el pack in-place del optimizer.
2. **QAT largo** sobre `qwen_coder_0.5b` (miles de pasos) o modelo base de mayor precisión.
3. **R-61**: 11 tools huérfanos.
4. **Commit** de la sesión (cambios unstaged).
5. R-53 (F32 emb/head), R-50 (parity), R-52 (rotación), R-72 (bench-infer).

**Fin de sesión:** 2026-11-02.
