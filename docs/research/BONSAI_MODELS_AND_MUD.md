# Investigación: modelos tipo **Bonsai** y encaje con MUD

**Fecha:** 2026-07-16 (estado runtime actualizado 2026-07-17)  
**Contexto:** El path BitNet “extremo” (MS 1.58b-2B nativo + repack vertical) es válido pero pesado en ops/import. Aquí se investigan **Bonsai** y vecinos: compactación *nativa* low-bit para edge, más cercanos al lema MUD *maximum intelligence / minimum footprint*.

**No confundir con:**
- *bonsai* workspace de visual analytics (arXiv 2604.19247) — no es un LLM.
- BONSAI reasoning multimodal (AAAI) — no es un peso ternario de chat.

---

## 1. Qué es la familia Bonsai (PrismML)

### 1.1 1-bit Bonsai (lanzamiento ~mar 2026)

PrismML salió de stealth con **Bonsai** como familia de LLMs **entrenados end-to-end en 1 bit** (no solo PTQ de un Qwen BF16):

| Variante | Params (aprox.) | Footprint claim | Rol |
|----------|-----------------|-----------------|-----|
| Bonsai 1.7B | ~1.7B | ~240–290 MB | Teléfono / browser / WebGPU |
| Bonsai 4B | ~4B | ~0.5 GB | Laptop ligero |
| Bonsai 8B | ~8.2B | ~1.15–1.28 GB | Flagship edge |
| Bonsai 27B | ~27B | ~3.9 GB (1-bit) | “Phone-class” 27B |

Claims públicos (PrismML / prensa): ~**14×** más pequeño que FP16 8B, ~**4–5×** menos energía/token (p.ej. RTX 4090 / Apple Silicon), demos de **~44 tok/s en iPhone**, Apache 2.0.

Arquitectura base reportada (8B): **Qwen3-8B dense** — GQA, SwiGLU, RoPE, RMSNorm, ctx largo (65k en docs HF).

Punto crítico vs BitNet MS: la comunidad (LocalLLaMA) suele decir que **Bonsai 1-bit es usable en chat real**, mientras BitNet MS era más “research / afásico” sin seating pesado.

### 1.2 Ternary Bonsai (1.58-bit, ~abr 2026)

Misma marca, **otro punto de la curva**:

- Pesos en **{-1, 0, +1}** con **escala FP16 por grupo de 128** (`g128`).
- Footprint ~**9×** menor que FP16 (menos extremo que 1-bit, más calidad).
- Tamaños: **1.7B / 4B / 8B** (+ GGUF **27B** ternary en HF).
- Mensaje oficial: *true ternary end-to-end* — emb, attn, MLP y LM head en 1.58-bit (sin “escape hatch” FP en el anuncio).
- Formatos: **MLX 2-bit**, **GGUF Q2_0 g128** (llama.cpp fork / kernels custom), safetensors “unpacked” para tooling HF.

Blog: <https://prismml.com/news/ternary-bonsai>  
Colección HF: <https://huggingface.co/collections/prism-ml/ternary-bonsai>  
GGUF 8B: `prism-ml/Ternary-Bonsai-8B-gguf`

### 1.3 deepgrove **Bonsai 500M** (otro linaje)

[deepgrove-ai/Bonsai](https://github.com/deepgrove-ai/Bonsai):

- **~500M** params, **pesos ternarios**, arch Llama-like, tokenizer Mistral/Danube-3.
- Entrenado con **&lt;5B tokens** (DCLM-Pro + Fineweb-Edu).
- Mensaje: efficiency *from training*, no solo quant post-hoc.

Escala más cercana a **SmolLM2-135M/360M** que a BitNet-2B o Bonsai-8B.

---

## 2. Bonsai vs BitNet vs PTQ-SmolLM (para MUD)

| Dimensión | BitNet MS 1.58b | Prism **Ternary Bonsai** | Prism **1-bit Bonsai** | SmolLM2 + PTQ MUD |
|-----------|-----------------|---------------------------|-------------------------|-------------------|
| Origen del ternario | Entrenado 1.58 | Entrenado 1.58 | Entrenado 1-bit | BF16 → PTQ (+ STE seating) |
| Formato pack | U8 vertical + scales | g128 scale + Q2_0/MLX | g128 1-bit | ELUT nibble + PRQ MUD |
| Emb | Suele ternario | Claim: todo 1.58 | Todo 1-bit | Híbrido (FP32 o ternary) |
| Calidad chat “out of box” | Históricamente frágil sin seating | Claim: fuerte en clase | Claim: usable edge | Afasia hasta seating largo |
| Tamaño cómodo MUD CPU | 2B–… pesado | **1.7B** ideal; 8B posible | 1.7B ideal | 135M ya en repo |
| Import a `.mud` | Converter vertical (V12) | **Nuevo path** g128/GGUF | **Nuevo path** 1-bit | Ya existe |
| Licencia | Check MS | Apache 2.0 (HF) | Apache 2.0 | Apache/HF |

**Lección paradigmática MUD (V06/V12/UCP):**  
PTQ de un dense FP a ELUT **siempre** produce Ternary Shock.  
Modelos **nativos low-bit** (BitNet *o* Bonsai) + seating STE son el camino de calidad; Bonsai se vende como el punto **comercialmente usable** de esa curva.

---

## 3. Encaje con el paradigma Forge/MUD

### Compatible (alto)

| Idea Bonsai | Hook MUD |
|-------------|----------|
| Pesos {-1,0,+1} + scale por grupo | ELUT 4-bit + PRQ (grupo 128 vs per-row — adaptar) |
| Edge / CPU / low RAM | i7-1260P, zero-alloc, PCorePool |
| Qwen3-like dense (GQA, SwiGLU, RMSNorm) | Ya soportado vía metadata + converter Llama/Qwen map |
| STE / seating post-import | `corpus_trainer`, AWAKE, restore-iq |
| No depender de PyTorch en runtime | Import una vez → `.mud` → solo Rust |
| 1.7B footprint | Más realista que 8B en laptop + menos extremo que “solo BitNet 2B” |

### Fricciones (hay que diseñar)

| Fricción | Detalle |
|----------|---------|
| **Group scale g128** | MUD hoy es **per-row PRQ**; Bonsai usa **scale compartida cada 128 pesos**. Hace falta layout en converter o expandir a per-row al import. |
| **GGUF Q2_0 / MLX** | No es ELUT nibble MUD; necesita unpacker (como el vertical MS de V12). |
| **1-bit vs 1.58** | MUD ISA y STE asumen ternario; 1-bit Bonsai es binario ±1 — otro kernel o mapa a ternario con ceros ralos. |
| **Ctx 65k** | L-13 HCA/CSA; no freír denso 65k en 1260P. |
| **Vocab Qwen (~152k)** | Emb grande; con emb FP32 el RAM sube (paradigma híbrido ya aceptado en battery). |
| **Fork llama.cpp** | No es runtime MUD; solo referencia de kernels / validación de calidad. |

### No compatible / baja prioridad

- Usar Bonsai solo vía API cloud PrismML.  
- Depender del fork llama.cpp como motor de producción (rompe P-07 / stack propio).  
- Tratar Bonsai Image (diffusion FLUX) como path de chat LLM.

---

## 4. Opciones operativas para Forge (sin “BitNet extremo”)

Ordenadas por **fidelidad MUD × esfuerzo × probabilidad de texto coherente**.

### Opción B1 — **Ternary Bonsai 1.7B → `.mud`** (recomendada)

1. Descargar safetensors “unpacked” o MLX/GGUF ternary de `prism-ml/Ternary-Bonsai-1.7B*`.  
2. Extender `universal_converter`:
   - Parse de **group scale 128** → o bien PRQ per-row al reempaquetar, o nuevo tensor `*.prq_scale` group-strided.  
   - Map Qwen3 nombres → `blk.N.attn_*` / `expert.0.w{1,2,3}`.  
3. Metadata: `rope_theta`, GQA heads, `max_pos` (capear a 4k–8k en train).  
4. **UCP:** health → AWAKE → STE seating corto (FREEZE_EMB si emb ya ternario nativo, o emb FP32 si se importa unpacked).  
5. Infer MUD con `scale_up=1`, sin peak-boost en logits muertos.

**Por qué no es “BitNet extremo”:** tamaño 1.7B, licencia Apache, nacido para edge; calidad claim superior a MS BitNet “research”.

### Opción B2 — **deepgrove Bonsai 500M**

- Escala cercana a experimentos actuales (Smol-class).  
- Entrenado ternario desde el principio con pocos tokens → seating MUD más barato.  
- Menos “wow” de benchmarks 8B, ideal para **CI e2e** y desarrollo del converter g128.

### Opción B3 — **Híbrido actual mejorado (SmolLM + UCP)** sin cambiar de familia

Seguir con SmolLM2 pero **no** como si fuera nativo ternario:

1. Convert capas ternary + emb FP32 tied (ya).  
2. **recalibration_projector ± boost** (Tier 2 UCP — casi no usado en battery).  
3. `training_healthcheck` (σ / sparsity).  
4. restore-iq / align **LAST_N=0** + FREEZE_EMB + corpus stash.  

Menos glamour; 100 % tools ya en repo.

### Opción B4 — **1-bit Bonsai solo como referencia de calidad**

- No mapear 1-bit a ELUT ternario a ciegas (rompe STE “{-1,0,+1}”).  
- Usar demos MLX/llama.cpp Prism **como oráculo** de “qué se siente un low-bit usable”.  
- Si se importa: capa de adaptación (binario → ternario con 0) o kernel binario aparte (fuera del core actual).

### Opción B5 — **Ternary Bonsai 8B** (fase 2 hardware)

- Solo si hay RAM + interés en stress L-13/HCA.  
- No es el primer target del i7-1260P de desarrollo diario.

---

## 5. Diseño técnico de import (sketch)

```
Ternary Bonsai (g128)
        │
        ▼
  unpack group: W_i ∈ {-1,0,+1}, s_g (FP16)
        │
        ├─► path A (rápido): expand s_g a scale por fila (absmean de reconstrucción)
        │         → ternarize_f32_and_pack / o pack directo ELUT
        │
        └─► path B (fiel): nuevo layout group-scale en .mud
                  (requiere GEMV consciente de g128 — más trabajo ISA)
        │
        ▼
  .mud + P-13 metadata + tokenizer Qwen
        │
        ▼
  UCP seating (STE) + TELEM VarH/VarJ
```

**Path A** es el más compatible *hoy* con `ternary_gemv` + PRQ sin tocar ASM.

---

## 6. Criterios de éxito (paradigm gates)

| Gate | Criterio MUD |
|------|----------------|
| Health | σ en rango ~0.5–0.8; no BUG-6 (σ&lt;0.3 ∧ zeros&gt;70%) |
| TELEM | VarH no DEAD_ACT; loss no pegada a ln(N) |
| Infer | Logits peak &gt; 0.05 natural o boost controlado; texto no-bucle |
| Runtime | Solo Rust; sin Python en hot path |
| Size | Preferir ≤2B para dev; 8B opcional |

---

## 7. Conclusión

- **Bonsai (PrismML)** no es un “mini-BitNet de juguete”: es la línea **comercial low-bit nativo** (1-bit y **1.58 ternary**) sobre base **Qwen3**, orientada a edge — alineada con el *footprint* y el stack MUD.  
- **deepgrove Bonsai 500M** es el hermano **pequeño y de investigación** para prototipar converter.  
- Frente a **BitNet MS extremo**, Ternary Bonsai 1.7B es el mejor **primer import de calidad** sin abandonar el paradigma ternario.  
- Frente a **SmolLM PTQ**, Bonsai evita el peaje más caro: *aprender* el manifold 1.58 desde un dense FP con afasia.

**Recomendación de producto (actualizada tras run real 1.7B):**

1. **Hecho (2026-07-16/17):** import GGUF Q2_0 → `.mud` + seating soft B/A en i7-1260P (ver §9).  
2. Corto plazo: seating más profundo (full FWD scales, o soft A más largo) *sin* OOM; medir prompts.  
3. Medio plazo: UCP completo (recalib + health) sobre el mud Bonsai; opcional emb/q-k norm polish.  
4. Largo plazo: soporte nativo g128 en GEMV si se adopta Bonsai 8B/27B.

---

## 8. RAM / hardware budget (extremo) — 15 GiB host

Host de desarrollo: **i7-1260P** (8 P-cores efectivos en PCorePool) + **Intel Iris Xe UMA** + **15 GiB** RAM.

| Prohibido | Por qué |
|-----------|---------|
| HF **unpacked** safetensors (~3.4 GB 1.7B) | RSS + page cache → OOM kill |
| Expandir todo el modelo a FP32 (1.7B×4 ≈ 6.8 GB) | + shadows QAT → muerte |
| `materialize_writable()` full en 1.7B | Duplica payload ELUT (~860 MiB → ~1.7 GiB owned) |
| `MUD_GPU_GEMV=1` / auto en este host | Bench: GPU **20–100×** más lento que AVX2 (UMA upload+readback) |

| Permitido / default | Footprint |
|---------------------|-----------|
| GGUF **Q2_0 g128** packed (~442 MiB) + `mmap` | OS page cache; no heap copy |
| `bonsai_gguf_to_mud` stream row→ELUT | peak ≈ mmap pages + 1 fila scratch |
| `.mud` ELUT+PRQ ≈ **~827 MiB** on disk (1.7B) | zero-copy load en infer |
| `materialize_for_ste_train(first, FREEZE_EMB)` | solo last-N weights+scales owned (~24–48 MiB) |
| Shadows FP32 solo last-N capas | p.ej. 2×7 matrices; no full 1.7B |
| Emb **lazy** (`FREEZE_EMB=1`) | evita ~1.2 GiB emb FP32; unpack fila on-the-fly |
| `MUD_GPU_GEMV=0` + `MUD_PCORE_THREADS=8` | hot path real = AVX2 × 8 P-cores |

**Tool:**  
`cargo run --release --bin bonsai_gguf_to_mud -- models/ternary_bonsai_1.7b/Ternary-Bonsai-1.7B-Q2_0.gguf models/ternary_bonsai_1.7b.mud`

**Train defaults (RAM, 1260P):**

```bash
export MUD_PCORE_THREADS=8
export MUD_GPU_GEMV=0          # force CPU — matches gemv_auto_bench
export MUD_USE_VULKAN=1        # opcional: heartbeat; GEMV sigue CPU
export MUD_TRAIN_FREEZE_EMB=1
export MUD_TRAIN_LAST_N_LAYERS=2
# MUD_TRAIN_FWD_LAST_N=auto    # seating speed (approx residual)
# MUD_TRAIN_FWD_LAST_N=full    # calidad (soft A)
# MUD_TRAIN_SCALES_ONLY=1      # B: no reescribe ELUT; **skips EZOP VRAM**
# MUD_TRAIN_EZOP=0             # force skip ash QAT buffers (Iris Xe UMA OOM shield)
# no MUD_TRAIN_MATERIALIZE_FULL salvo debug
```

**EZOP OOM:** en Iris Xe UMA, `ensure_buffers` reserva shadow+grad HOST_VISIBLE ≈ 2×FP32 del last-N. Si falla → fallback CPU STE (no panic). `SCALES_ONLY` y `MUD_TRAIN_EZOP=0` ni intentan alloc.

**TELEM:** `tok/s` en align ≈ **chunks/s × steps/chunk** (no chat tokens/s).  
**nvtop iGPU:** clocks/busy suelen ser heartbeat ash + compositor, **no** el FWD STE.

Detalle bottleneck: `docs/architecture/ASM_VULKAN_BOTTLENECK_2026-07.md`.

---

## 9. Estado runtime — Ternary Bonsai 1.7B en MUD (2026-07-16/17)

### 9.1 Import

| Item | Valor |
|------|--------|
| Fuente | GGUF **Q2_0 g128** packed (no unpacked) |
| Converter | `tools/bonsai_gguf_to_mud.rs` → ELUT 4-bit + PRQ per-row (path A g128→ELUT) |
| Arch | Qwen3-like: h=2048, L=28, heads=16/8, ffn=6144, vocab≈151669 |
| Fixes import | `ffn_norm\|norm`; dense FFN `ffn_up/gate/down`; specials/vocab desde `added_tokens` |
| Qwen3 | `q_norm` / `k_norm` en `SlimeLayer` + FWD |
| `.mud` | ~**827 MiB** en disco |

### 9.2 Checkpoints en `models/`

| Archivo | Rol |
|---------|-----|
| `ternary_bonsai_1.7b.mud.pre_seat` | Post-import, pre seating |
| `ternary_bonsai_1.7b.mud.post_seat_damaged` | STE agresivo (LR~5e-4) → colapso (no usar) |
| `ternary_bonsai_1.7b.mud.after_b` | Soft B corto |
| `ternary_bonsai_1.7b.mud.after_ab` | A+B temprano |
| `ternary_bonsai_1.7b.mud.after_b_long` | B intermedio |
| `ternary_bonsai_1.7b.mud.after_b_alive` | B largo SCALES_ONLY 256 chunks (keep-alive) |
| `ternary_bonsai_1.7b.mud.after_a_soft` | Soft A @1e-5 sobre `after_b_alive` |
| `ternary_bonsai_1.7b.mud.after_ab_soft` | **Head actual:** soft A + más B (128 chunks) |

Cadena final:

```
pre_seat → … → after_b_alive → after_a_soft → after_ab_soft
```

### 9.3 Seating corrido (resumen)

| Fase | Env clave | Resultado |
|------|-----------|-----------|
| Soft B | `SCALES_ONLY=1` LR=1e-4 LAST_N=2 FREEZE_EMB | VarH estable; trit-safe |
| Soft A | STE LR=**1e-5** LAST_N=2 **FWD full** FREEZE_EMB 32×8 | VarH **~43** conf alta; ~1.8 steps/s; RSS ~1.8 GB |
| Más B | SCALES_ONLY LR=1e-4 **FWD_LAST_N=auto** 128×8 | VarH~2.2 (stack parcial); ~2.0 steps/s; RSS ~1.9 GB |
| B keep-alive | SCALES_ONLY + FWD_LAST_N 256×8 | Completó sin OOM; free~9–12 GiB |

**Lecciones:**

- LR STE **≥5e-4** en last-N → VarH explode / bot collapse → restaurar `pre_seat`.  
- Soft A @ **1e-5** es el régimen seguro observado.  
- `SCALES_ONLY` no reescribe códigos ELUT (PRQ `s=ΣW·T/ΣT²`); ash pack desactivado en ese modo.  
- `FWD_LAST_N=auto` multiplica steps/s en seating B pero **aproxima** residual (calidad &lt; full FWD).  
- Inferencia post-seat: `emb_rms≈7.26e-3` (sano). **Afasia** de stopwords sigue (techo del seating superficial + pre-seat débil en MUD path).  
- Prompts re-eval: `/tmp/bonsai_prompt_eval_alive.log`, `/tmp/bonsai_prompt_eval_ab_soft.log`.

### 9.4 ASM / train stack (T10/T11 + RAM)

| ID | Cambio | Estado |
|----|--------|--------|
| — | `ternary_gemv_4rows` 16-col loop | done |
| T10 | Prefetch 1260P (`ternary_gemv.s` + 4rows): T0/T1 x, NTA W | done |
| T11 | `ternary_gemv_8rows.s` + tests; submit default **4-row**, opt-in `MUD_GEMV_ROWS=8` | done |
| — | `MUD_TRAIN_FWD_LAST_N` + emb lazy FREEZE | done |
| — | PCorePool 8, `MUD_GPU_GEMV=0` | default host |

Microbench (n=2048, warm L2): **4-row ≥ 8-row** en media → no forzar 8-row en default.

### 9.5 Cómo reanudar

```bash
# Infer con head actual
cp -f models/ternary_bonsai_1.7b.mud.after_ab_soft models/ternary_bonsai_1.7b.mud
MUD_PCORE_THREADS=8 MUD_GPU_GEMV=0 MUD_USE_VULKAN=1 \
  ./target/release/forge_llm models/ternary_bonsai_1.7b.mud

# Soft A extra (calidad FWD full)
MUD_PCORE_THREADS=8 MUD_GPU_GEMV=0 MUD_TRAIN_FREEZE_EMB=1 \
MUD_TRAIN_LAST_N_LAYERS=2 MUD_TRAIN_FWD_LAST_N=full \
MUD_TRAIN_SCALES_ONLY=0 MUD_QAT_LR=0.00001 \
MUD_TRAIN_MAX_CHUNKS=64 MUD_TRAIN_STEPS_PER_CHUNK=8 MUD_TRAIN_SKIP_AWAKE=1 \
  ./target/release/run_trainer --align --epochs 1 --batch 8 models/ternary_bonsai_1.7b.mud

# Más B scales (speed seating)
MUD_TRAIN_SCALES_ONLY=1 MUD_QAT_LR=0.0001 MUD_TRAIN_FWD_LAST_N=auto \
MUD_TRAIN_MAX_CHUNKS=256 ...  # resto igual FREEZE/LAST_N/PCORE
```

### 9.6 Gaps abiertos

1. Calidad chat: seating superficial no elimina afasia; hace falta corpus/align más largo o full-FWD B.  
2. TELEM naming dual (steps/s vs chat tok/s) para no confundir con nvtop.  
3. g128 nativo en GEMV (opcional, 8B+).  
4. No usar `post_seat_damaged` ni LR STE alto en last-N.

---

## 10. Referencias

- PrismML 1-bit Bonsai: https://prismml.com/news/bonsai-8b  
- PrismML Ternary Bonsai: https://prismml.com/news/ternary-bonsai  
- HF Ternary Bonsai collection: https://huggingface.co/collections/prism-ml/ternary-bonsai  
- HF GGUF 8B: https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf  
- deepgrove Bonsai 500M: https://github.com/deepgrove-ai/Bonsai  
- MUD UCP: `docs/manuals/MUD_CALIBRATION_PROTOCOL.md`  
- MUD training: `docs/manuals/MUD_TRAINING_PROTOCOLS.md`  
- BitNet restore audit: `docs/audits/MUD_AUDIT_REPORT_V12_BITNET_RESTORATION.md`  
- ASM / Vulkan bottleneck (1260P): `docs/architecture/ASM_VULKAN_BOTTLENECK_2026-07.md`
