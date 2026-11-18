# MUD — Session Report 2026-11-07 (auditoría, REA/Ghidra, ΔW de sesión)

**Estado:** sesión ejecutada y validada.
**Próxima sesión:** [`docs/planning/PLAN_NEXT_SESSION_2026-11-07.md`](../planning/PLAN_NEXT_SESSION_2026-11-07.md).
**Alcance:** auditoría del árbol, verificación ASM con REA/Ghidra, corrección de
la medición de ΔW, cierre de R-61/R-74 y saneamiento de tools/gates.

---

## 1. Ground truth al abrir

| Verificación | Comando | Resultado |
|---|---|---|
| Tests lib release | `cargo test --lib --release -- --test-threads=2` | **475 passed / 0 failed / 0 ignored** |
| Clippy | `cargo clippy --all-targets --all-features -- -D warnings` | **0 warnings / 0 errors** |
| Unsafe | `./mud.sh unsafe-audit` | **536 unsafe · 0 sin justificar** |
| Env SSOT | `./mud.sh env-audit` | **0 divergencias** |
| CI | `./mud.sh ci` | `[L-12] Health battery complete.` |

---

## 2. Hallazgos y correcciones

### 2.1 ΔW real de sesión (F-01/F-02/F-03, R-74)

El optimizer empaqueta los trits actualizados **in-place** en el tensor del
modelo; por eso `sync_shadow_to_mud` compara contra bytes que ya contienen el
resultado y su ΔW por sync tiende a 0. Además, `hash_trained_weights`:
- iteraba un `HashMap` (orden no determinista entre procesos);
- no incluía `output.weight`, `output_norm.weight` ni normas/mHC;
- calculaba el tamaño mmap de un `Ternary2Bit` como
  `rows * cols * cols.div_ceil(2) * 4` (OOB read latente).

**Corrección aplicada:**
- `WeightSnapshot` con hash por tensor + hashes de bloques de 1 KiB.
- `diff_trainable_weights` → `changed_tensors`, `changed_bytes` (1 KiB),
  `total_bytes`.
- `[DW] session tensors=X/Y bytes=... total=... moved=...` en
  `mud_train_metrics.log`.
- `hash_trained_weights` = digest ordenado del snapshot (determinista e incluye
  cabeza/normas/mHC).
- `tensor_bytes` usa `checked_data_size()`.

**Tests nuevos:**
- `test_weight_snapshot_detects_change_and_is_deterministic`
- `test_tensor_bytes_uses_checked_data_size`

**Evidencia QAT corto (copia de `ale.mud`, 1 chunk × 8 pasos, LR 0.1,
`LAST_N=all`, `FREEZE_EMB=0`, `DEADZONE=0`):**

```text
[DW] bytes=8913006 total=20153728 prq=47.3384 moved=1
[DW] session tensors=110/119 bytes=11779480 total=21410416 moved=1
[DW] bytes=0 total=20159872 prq=0.0000 moved=0
[DW] session tensors=110/119 bytes=11779480 total=21410416 moved=1
✓ final checkpoint persisted (weights hash 0xe5c428a76558bc4e → 0x02d5ceef20631ae7)
```

El primer sync detectó 44.2 % de bytes ternarios movidos; el segundo, 0 % (no
hubo más pasos). El ΔW de sesión es 110/119 tensores y ~11.8 MB de 21.4 MB.
El hash final confirma que el `.mud` cambió.

### 2.2 R-61 — tools huérfanos / bins fantasma

- Los 11 tools que R-61 listaba como huérfanos ya están declarados en
  `Cargo.toml`.
- `mud.sh tools` ya no lista `variance_inspector` (borrado) y ahora incluye
  `next_token_eval`, `gemv_parity`, `parity_gate`, `inference_coherence`,
  `training_healthcheck`, `converter_auditor`, `precise_converter_audit`,
  `pipeline_overlap_bench`, `gemv_auto_bench`, `mhc_stress_bench`,
  `autofinetune` y los validadores/benches declarados.
- Nuevo `./mud.sh tools-audit`:
  - cada `tools/*.rs` y `tools/*/main.rs` debe aparecer como `path = "..."`
    en `Cargo.toml`;
  - cada `cargo run ... --bin NAME` de `mud.sh` debe existir en `Cargo.toml`.

**Evidencia:**

```text
$ ./mud.sh tools-audit
[TOOLS-AUDIT] OK — 0 tools huérfanos, 0 bins fantasma (R-61)
exit=0
```

### 2.3 Gates de documentación y entorno

- `MUD_GEMV_ROWS`: doc corregida (default real `8`; `MUD_GEMV_ROWS=4` fuerza
  el kernel de 4 filas).
- `env-audit` y `tools-audit` añadidos a `./mud.sh ci` y a
  `.github/workflows/ci.yml`.
- `env-audit` se invoca vía `bash "$0" env-audit` (subshell) para no terminar
  la batería con su `exit`.

### 2.4 P-13 en herramientas de evaluación

- `next_token_eval`: vocab desde `vocab_size` o shape de
  `token_embd.weight`/`output.weight`; error claro si no existe.
- `inference_coherence`: idem.
- `inference_coherence::load_prompts`: solo líneas 100 % numéricas se tratan
  como IDs (antes “Hello 42 world 7” se convertía en `[42, 7]`).
- `inference_coherence` parsea posicionales sin usar el path por defecto como
  centinela.

### 2.5 Suite ASM huérfana (F-11) y paridad de escalas (F-12)

- `src/asm/tests.rs` (1488 LOC, 45 `#[test]`) **no estaba declarada** en
  `src/asm/mod.rs`: `cargo test` nunca ejecutaba un solo kernel ASM.
  Se añadió `#[cfg(test)] mod tests;` y se eliminaron 2 tests muertos
  (`test_hadamard_transform_avx2`, `test_sgemm_avx2`; sus kernels ya no
  existen). Resultado: **43 ASM tests passed, 0 failed, 0 ignored**.
- Divergencia ASM vs escalar con escalas PRQ no finitas:
  `apply_row_scales_8_avx2` hace NaN/±Inf → 0; el fallback escalar clampeaba
  primero y multiplicaba por ±1e8. Se unificó con `finite_scale` (no finito → 0,
  finito fuera de banda → ±1e8) y se añadió
  `test_apply_row_scales_8_finite_guard`.

### 2.6 Comandos nuevos

| Comando | Bin | Uso |
|---|---|---|
| `./mud.sh predict-eval [model] [prompts]` | `next_token_eval` | top-1/5/10, MRR, NLL, PPL, entropía |
| `./mud.sh gemv-parity [model] [tensor]` | `gemv_parity` | ASM vs referencia escalar dequantizada |

---

## 3. Verificación ASM con REA/Ghidra

**Target:** `target/release/forge_llm` reconstruido con
`CARGO_PROFILE_RELEASE_STRIP=false RUSTFLAGS="-C target-cpu=native"`;
`sha256 72cda711422d43b1158959f609f191f2c0f10ce1da9f97ab7fc93bb3152801e0`.

**Proveedor:** Ghidra 12.1.2 (REA `analysis-provider-binding: ghidra`).

### 3.1 `ternary_gemv_8rows_avx2`

| Hecho | Evidencia |
|---|---|
| Cuerpo | 1803 bytes, 383 instrucciones, 22 bloques básicos, **0 callees** |
| Call sites | 2 en `forge_llm::mud::gemv_gpu::ternary_gemv_rowwise_submit` + 2 en el shim de closure del `PCorePool` |
| Decodificación | `vpsrlvd` + `vpand` + 2×`vpcmpeqd` por `u32` (8 nibbles); `+1`=0x1, `-1`=0xF, `0`=resto |
| Acumulación | `vpand` + `vaddps` / `vsubps` por fila |
| Seguridad numérica | guard NaN/Inf (exponente todo-1 → 0.0) por fila; `vzeroupper` al retorno |
| Resto | el wrapper Rust llama al ASM con `n & !7`; `ternary_gemv_tail` cubre 1..7 |
| Tamaño de loop | `loop16` 817 bytes/160 instr. para 128 MACs; `loop8` 80 instr. |

**Runtime (GDB, `next_token_eval models/ale.mud tests/data/next_token_eval.txt --max 1`):**

```text
KERNEL n=384 stride_words=48 scale=1.000000 out=... x=... w=...
MIXED-WEIGHT KERNEL n=128 stride_words=16 scale=1.000000
0x...: 0xf11fff1f 0x11ffff01 0x1f000111 0x01f1f01f
out[0..8]: 0.509056 9.561947 -4.737835 26.371538 17.566809 -15.792547 -6.308182 5.134826
```

La llamada mixta ejercita las ramas `+1` y `-1`; `stride_words = n/8` exacto;
`scale=1.0` (las escalas PRQ se aplican después con
`apply_row_scales_8_avx2`).

**Paridad independiente:**

```text
$ ./mud.sh gemv-parity models/ale.mud blk.0.attn_q.weight
asm1_vs_ref  max_abs=9.536743e-7 rms=2.062201e-7 rel_rms=1.643347e-7 cosine=1.000000000
asm8_vs_ref  max_abs=9.536743e-7 rms=2.164522e-7 rel_rms=1.724885e-7 cosine=1.000000000
asm1_vs_asm8 cosine=1.000000000
```

### 3.2 Optimización — estado medido (no “hecho”)

`asm_kernel_bench` (size=4096, 200 iteraciones):

| Kernel | ASM/AVX2 | Scalar | Speedup |
|---|---:|---:|---:|
| `ternary_gemv` | 730 ns | 3 038 ns | 4.16× |
| `ternary_gemv_4rows` | 1 820 ns | 13 012 ns | 7.15× |
| `ternary_gemv_8rows` | 4 229 ns | 24 897 ns | 5.89× |
| `adam_step` | 2 303 ns | 2 143 ns | **0.93×** |
| `lm_head` (v=1024,h=4096) | 945 021 ns | 3 324 586 ns | 3.52× |

Observaciones:
- El 8-row es correcto, pero el coste por fila en `n>=2048` es peor que dos
  llamadas al 4-row en este host; el benchmark global es ruidoso (frecuencia /
  carga), así que **no se cambia el default** en esta sesión.
- `adam_step_avx2` es 0.93× vs el escalar auto-vectorizado: candidato a
  desenrollar 2× o a usar una aproximación recíproca; no se toca sin A/B
  estable y validación numérica.
- `ternary_gemv` y `lm_head` mantienen ganancias claras.

---

## 3.3 Prueba de inferencia (post-correcciones)

| Modelo | Prueba | Resultado |
|---|---|---|
| `models/ale.mud` (ternary, 22 MB) | `next_token_eval … tests/data/next_token_eval.txt --max 8` | **0/8 top-1**, NLL 12.0286, PPL **167,471.9**, entropía 16.914 bits |
| `models/ale.mud` | generación casi-greedy `"La inteligencia artificial es"`, 16 tokens (`MUD_INFER_TEMPERATURE=0.01`) | corre sin crash; salida incoherente (`folderOWS awe unanim`) |
| `models/qwen_coder_0.5b_trained.mud` (F32, 738 MB) | `next_token_eval … --max 2` | **0/2 top-1**, NLL 9.7863, PPL **17,788**, entropía 7.084 bits; 2 prompts en **2.17 s** |
| `models/qwen_coder_0.5b_trained.mud` | generación casi-greedy `"def fibonacci(n):"`, 8 tokens (`MUD_INFER_TEMPERATURE=0.01`) | corre sin crash; **4.30 s**; salida incoherente (`. 1,....`) |

**Hallazgo F-13 (documentado, no corregido en esta sesión):**
`MUD_INFER_TEMPERATURE=0` **no** activa greedy: el código hace
`.filter(|t| *t > 0.0).unwrap_or(0.8)`, así que `0` cae al default `0.8`.
Para una prueba determinista usar `MUD_INFER_TEMPERATURE=0.01` (o corregir el
filtro en la próxima sesión).

Cabeceras del forward (prueba de que el path de inferencia se inicializa):

```text
ale:  [Inference] emb_rms=6.3905e-1 scale_up=1.000 logit_scale=0.0884 tied=false ternary=true
qwen: [Inference] emb_rms=1.4064e-2 scale_up=8.000 logit_scale=0.0334 tied=true  ternary=false
```

**Lectura:** el motor ejecuta inferencia ternary y F32 sin crash y con logits
finitos; la paridad ASM ya está certificada (cosine 1.0). La calidad de
predicción sigue siendo el problema abierto (0 top-1), coherente con el
diagnóstico de Ola 4/QAT largo, no con una regresión de esta sesión.

---

## 4. Verificación final

| Gate | Comando | Resultado |
|---|---|---|
| Tests lib release | `cargo test --lib --release -- --test-threads=2` | **520 passed / 0 failed / 0 ignored** (43 ASM recuperados) |
| Proptest | `cargo test --lib --release --features proptest -- --test-threads=2` | **527 passed / 0 failed / 0 ignored** |
| Clippy | `cargo clippy --all-targets --all-features -- -D warnings` | **0** |
| Unsafe | `./mud.sh unsafe-audit` | **536 · 0 sin justificar** |
| Env | `./mud.sh env-audit` | **0 divergencias** |
| Tools | `./mud.sh tools-audit` | **0 huérfanos / 0 fantasmas** |
| CI | `./mud.sh ci` | **[L-12] Health battery complete.** |

---

## 5. Fórmulas de calidad (2026-11-07)

Documento de investigación y cálculo:
[`docs/research/MUD_QUALITY_FORMULAS_2026-11-07.md`](../research/MUD_QUALITY_FORMULAS_2026-11-07.md).
Plan de ejecución:
[`docs/planning/PLAN_QUALITY_2026-11-07.md`](../planning/PLAN_QUALITY_2026-11-07.md).

**Cálculo principal:** el conversor usa \(s=\gamma/\sqrt2\) con
\(\delta=0.7s\); BitNet b1.58 usa \(s=\gamma,\delta=0.5\gamma\); el óptimo
least-squares para los códigos es \(s^*=\sum T W/\sum|T|\). Medido en 256
filas de `token_embd` de Qwen:

```text
s_cur=0.00948  s*=0.01791  ratio=1.889
mse_cur=1.128e-4  mse*=6.281e-5
nmse_cur=0.3908  nmse*=0.2177
snr_cur=4.08 dB  snr*=6.62 dB
```

La predicción analítica gaussiana (\(s^*/s_c=1.8879\), NMSE 0.3878→0.2139,
SNR 4.11→6.70 dB) coincide con el modelo real. BitNet exacto baja NMSE ~32 %.
**Pero** el reescalado post-hoc de `prq_scale` empeora PPL
(`ale` 167,471→244,336; Qwen 17,788→20,697): la corrección debe hacerse en
conversión + QAT, no en inferencia.

**Tools añadidos:** `./mud.sh quant-audit` (`quant_formula_audit`) y
`./mud.sh scale-rescale` (`scale_rescale`).

**Calibración medida** (`next_token_eval --temperature-scan`):

| Modelo | \(T^*\) | PPL(\(T^*\)) | ECE(1) | ECE(\(T^*\)) | Top-1 |
|---|---:|---:|---:|---:|---:|
| `ale.mud` | 2.0 | 151,396 (vs 167,472) | 0.000103 | 0.000027 | 0/8 |
| `qwen_coder_0.5b_trained` | 1.4 | **9,055** (vs 17,788) | 0.264 | **0.046** | 0/2 |

La temperatura mejora NLL/ECE de Qwen (~2× PPL, 5.7× ECE) pero **no el
ranking**; el top-1 sigue 0. `ale` es casi uniforme y no se beneficia.

---

## 6. Cierre

**Cerrado en esta sesión:** R-61, R-74 y la deuda de medición ΔW.
**Abierto / siguiente:** Ola 4 (R-50 paridad, R-52 rotación Hadamard, R-53
emb/head F32 y ablación), QAT largo de `qwen_coder_0.5b` (horas de CPU) y
optimización ASM 8-row/Adam con benchmark estable. El motor ASM está
verificado numéricamente correcto (Ghidra + GDB + `gemv_parity`); lo que falta
es rendimiento, no corrección.
