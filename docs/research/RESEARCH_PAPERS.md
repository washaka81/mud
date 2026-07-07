# 📚 Forge LLM (MUD) — Índice de Papers de Investigación
**Estado:** Compilado el 2026-06-04 · 53 papers · 5 dominios de investigación  
Cada paper incluye enlace directo, autores, año y relevancia específica para el engine MUD.

---

## 🗺️ Mapa de Dominios

| # | Dominio | Papers | Prioridad |
|---|---------|--------|-----------|
| I | [Cuantización Ternaria & BitNet](#i-cuantización-ternaria--bitnet) | 18 | 🔴 CRÍTICO |
| II | [Mamba SSM & Arquitecturas Híbridas](#ii-mamba-ssm--arquitecturas-híbridas) | 10 | 🔴 CRÍTICO |
| III | [Mixture of Experts & Atención Eficiente](#iii-mixture-of-experts--atención-eficiente) | 10 | 🟠 ALTO |
| IV | [Razonamiento Recursivo & Test-Time Training](#iv-razonamiento-recursivo--test-time-training) | 7 | 🟡 MEDIO |
| V | [Edge AI & Optimización SIMD](#v-edge-ai--optimización-simd) | 8 | 🟠 ALTO |

---

## I. Cuantización Ternaria & BitNet

> Papers que fundamentan el formato `.mud` (pesos `{-1, 0, +1}`), el pipeline QAT/STE, la optimización multiplication-free, y el PRQ Per-Row.

---

### 1. BitNet: Scaling 1-bit Transformers for Large Language Models
- **Autores:** Hongyu Wang, Shuming Ma, Li Dong et al. (Microsoft Research)
- **Año:** 2023
- **arXiv:** https://arxiv.org/abs/2310.11453

**Relevancia para MUD:**
Introduce la capa `BitLinear` como reemplazo directo de `nn.Linear` con pesos binarizados a `{-1, +1}` usando absmean por tensor. Es la prueba fundacional de que los Transformers 1-bit entrenados nativamente (no PTQ) siguen las scaling laws. El PRQ Per-Row del formato `.mud` extiende directamente el absmean por tensor de este paper a granularidad por fila — la evolución natural de `BitNet → BitNet b1.58`.

---

### 2. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58) ⭐
- **Autores:** Shuming Ma, Hongyu Wang, Lingxiao Ma et al. (Microsoft Research)
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2402.17764

**Relevancia para MUD:**
**El paper de especificación primaria del formato `.mud`.** Introduce formalmente el espacio ternario `{-1, 0, +1}` (log₂3 ≈ 1.58 bits), el absmean PRQ para pesos y el absmax por token para activaciones. La paradoja Sigma (σ=0.86 como límite real de varianza ternaria, documentada en Audit V9) tiene sus raíces en el análisis de sparsity de este paper. El target de 26% de sparsity en MUD viene directamente de las distribuciones estadísticas reportadas aquí.

---

### 3. BitNet b1.58 2B4T Technical Report (2025)
- **Autores:** Shuming Ma, Hongyu Wang et al. (Microsoft Research)
- **Año:** 2025
- **arXiv:** https://arxiv.org/abs/2504.12285

**Relevancia para MUD:**
El primer LLM 1-bit open-source a escala de producción (2B params, 4T tokens), con paridad frente a FP16/BF16 equivalente en benchmarks de lenguaje, código y matemáticas. Valida la dirección arquitectural de MUD (pesos ternarios + QAT/STE desde cero) a escala de producción. El pipeline multi-etapa (pre-training → SFT → DPO) es un mapa de referencia directo para el pipeline `restore-iq` → `iteration_validator` de MUD.

---

### 4. bitnet.cpp — 1-bit AI Infra Part 1.1
- **Autores:** Jinheng Wang, Hansong Zhou, Ting Song et al. (Microsoft Research)
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2410.16144

**Relevancia para MUD:**
El análogo industrial más cercano al engine de inferencia de MUD. Documenta kernels basados en LUT que eluden la multiplicación matricial para pesos ternarios, logrando speedups de **2.37×–6.17× en x86** con 72–82% de ahorro energético. Comparación directa con `src/asm/ternary_gemv_4rows_avx2`. La versión enero 2026 añade kernel paralelo + tiling para 1.15×–2.1× adicional. Referencia de auditoría para `BIT-01` (Roadmap Phase 14).

---

### 5. QAT: Quantization and Training of Neural Networks (Google CVPR 2018) ⭐
- **Autores:** Benoit Jacob, Skirmantas Kligys, Bo Chen et al. (Google)
- **Año:** 2018
- **arXiv:** https://arxiv.org/abs/1712.05877

**Relevancia para MUD:**
Establece el principio QAT fundamental que implementa el training loop STE de MUD: inyectar "fake quantization" (clamps de cuantización simulados) en el **forward pass** mientras se mantienen shadow weights FP32 para el backward. Resuelve el "Ternary Shock" / semantic aphasia identificado en Audits V3/V5 — el PTQ causa degradación irreversible, pero QAT permite que el modelo se adapte estructuralmente a las fronteras ternarias. El `Forced Hot PRQ` de MUD (FP32 shadow → round/clamp a `[-1,0,1]` → pack) es una aplicación directa de "fake quantization" para 1.58-bit.

---

### 6. Straight-Through Estimator (STE) — Bengio et al. ⭐
- **Autores:** Yoshua Bengio, Nicholas Léonard, Aaron Courville (Université de Montréal)
- **Año:** 2013
- **arXiv:** https://arxiv.org/abs/1308.3432

**Relevancia para MUD:**
**El backbone matemático del training loop QAT de MUD** (resolución Audits V6/V7). La función de redondeo ternario `round(x)` es no-diferenciable (gradiente cero casi en todas partes), lo que haría imposible el backprop. El STE resuelve esto tratando el cuantizador como función identidad durante el backward pass: `∂L/∂x_float ≈ ∂L/∂x_quantized`. Permite que los gradientes fluyan a través del paso de truncación `[-1, 0, 1] * scale` y actualicen los shadow weights FP32.

---

### 7. FairyFuse: Multiplication-Free LLM Inference via Fused Ternary Kernels 🔥
- **Autores:** Yu-Zhen Xuan et al.
- **Año:** 2026 (Abril)
- **arXiv:** https://arxiv.org/abs/2604.20913

**Relevancia para MUD:**
FairyFuse propone fusionar sub-GEMVs de modelos ternarios en loops SIMD usando solo adiciones y substracciones enmascaradas, eliminando completamente LUT overheads y dequantización. Fusiona el `scale` directamente en el acumulador. Para MUD con AVX2, esto es una referencia para el item `BIT-01` (optimizaciones en `src/asm/ternary_gemv_4rows_avx2.s`) para fusionar la escala PRQ dentro del acumulador para evitar el overhead de carga/almacenamiento por fila.

---

### 8. Scalable MatMul-free Language Modeling 🔥
- **Autores:** Chao Zhou et al.
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2406.02528

**Relevancia para MUD:**
Demuestra que los LLMs pueden eliminar **completamente** la multiplicación matricial en todas las capas usando activaciones ternarias y pesos ternarios, reemplazando el GEMM por operaciones bitwise XNOR y POPCNT. El modelo de 2.7B parámetros mantiene una perplexidad competitiva. Para MUD, valida la viabilidad de prescindir de instrucciones de multiplicación no solo para pesos sino también para activaciones en futuras fases del motor.

---

### 9. Sparse-BitNet: Semi-Structured Sparsity for 1.58-bit LLMs
- **Autores:** Microsoft Research
- **Año:** 2026
- **Link:** Disponible en el repo `microsoft/BitNet`

**Relevancia para MUD:**
Empíricamente, demuestra que el **42% de los pesos ternarios son naturalmente cero** en modelos BitNet b1.58 entrenados. Sparse-BitNet explota esta sparsity semi-estructurada (N:M) combinada con pesos ternarios para lograr speedups adicionales de 1.5-2×. Para MUD, esto justifica añadir un bitmask de sparsity en el formato `.mud` para realizar skip automático de pesos cero directamente en el kernel AVX2 sin cargarlos de memoria.

---

### 10. ITQ3_S — Interleaved Ternary Quantization con FWHT
- **Autores:** Investigación de Cuantización Académica
- **Año:** 2026

**Relevancia para MUD:**
Aplica la Fast Walsh-Hadamard Transform (FWHT) para rotar el espacio de pesos antes de la cuantización ternaria. Esto redistribuye los outliers en distribuciones casi Gaussianas, haciendo que la cuantización PRQ sea mucho más fiel y reduciendo drásticamente el error de cuantización en modelos con outliers altos (como Qwen3-4B). Útil para el `universal_converter`.

---

### 11. TWN: Ternary Weight Networks
- **Autores:** Fengfu Li, Bo Zhang, Bin Liu, Xiaolin Hu (Tsinghua University)
- **Año:** 2016
- **arXiv:** https://arxiv.org/abs/1605.04711

**Relevancia para MUD:**
Paper fundacional de redes ternarias previo a BitNet. Define el threshold de cuantización ternaria `|w| > Δ` donde `Δ = 0.7 × E[|w|]`. El factor `0.7` de TWN es el precursor matemático del factor de dampening `0.707` (1/√2) utilizado en MUD para contrarrestar la inflación de la varianza durante la cuantización.

---

### 12. XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks
- **Autores:** Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, Ali Farhadi
- **Año:** 2016
- **arXiv:** https://arxiv.org/abs/1603.05279

**Relevancia para MUD:**
Paper histórico que introduce el cálculo binario `XNOR + POPCNT` y el escalado local por canal. Relevante para entender cómo emular eficientemente popcounts y sumas vectoriales acumuladas en registros SIMD en `src/asm/`.

---

### 13. DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks
- **Autores:** Shuchang Zhou, Yuxin Wu, Zekun Ni, Xinyu Zhou, Hewen He, Yuheng Jia
- **Año:** 2016
- **arXiv:** https://arxiv.org/abs/1606.06160

**Relevancia para MUD:**
Establece métodos para cuantizar tanto activaciones como pesos y gradientes a baja precisión (1-bit, 2-bit, etc.). Útil como referencia teórica para la cuantización de activaciones a 8-bit (DoReFa aplicado a redes ternarias) y para los gradientes en el entrenamiento QAT nativo en `forge_autograd`.

---

### 14. AWQ: Activation-aware Weight Quantization
- **Autores:** Ji Lin, Jiaming Tang, Haotian Tang, et al. (MIT/NVIDIA)
- **Año:** 2023
- **arXiv:** https://arxiv.org/abs/2306.00978

**Relevancia para MUD:**
Demuestra que proteger solo el 1% de los pesos con magnitudes salientes (outliers en activaciones) reduce el error de cuantización drásticamente sin necesidad de alta precisión en el 99% restante. Es la base teórica del Per-Row Quantization (PRQ) de MUD, validando el escalado fila a fila.

---

### 15. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- **Autores:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
- **Año:** 2022
- **arXiv:** https://arxiv.org/abs/2210.17323

**Relevancia para MUD:**
Aplica la inversa de la matriz Hessiana para minimizar el error de cuantización. Es la base de los formatos GGUF. Muestra los límites de la cuantización PTQ y explica por qué la aproximación de MUD (PRQ dampening $O(n)$ más QAT) es más óptima para arquitecturas ultra-cuantizadas que el PTQ tradicional.

---

### 16. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- **Autores:** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
- **Año:** 2022
- **arXiv:** https://arxiv.org/abs/2208.07339

**Relevancia para MUD:**
Introduce la descomposición de la multiplicación matricial en FP16 y enteros INT8 para manejar outliers sistemáticos en las activaciones. Justifica la necesidad del floor numérico y RMS stabilization de `1e-8` (Audit V9) para evitar la explosión de logits en MUD.

---

### 17. ShiftAddNet: A Hardware-Inspired Framework for Fast and Energy-Efficient Neural Networks
- **Autores:** Haoran You et al.
- **Año:** 2020
- **arXiv:** https://arxiv.org/abs/2010.12785

**Relevancia para MUD:**
Reemplaza multiplicaciones con operaciones de bit-shift y adiciones. Valida matemáticamente el kernel `ternary_gemv_4rows_avx2` de MUD, donde multiplicar por pesos ternarios en $\{-1, 0, 1\}$ se reduce a operaciones de negación (complemento a dos), identidades (noops) o saltos (skip).

---

### 18. AdderNet: Do We Really Need Multiplications in Deep Learning?
- **Autores:** Hanting Chen et al.
- **Año:** 2019
- **arXiv:** https://arxiv.org/abs/1912.13200

**Relevancia para MUD:**
Propone utilizar distancias L1 (solo restas y sumas) en lugar de multiplicaciones matriciales para capas convolucionales y de atención. Inspiración para la fase `LDT-01` del Roadmap (capas de deducción en retículos matemáticos).

---

## II. Mamba SSM & Arquitecturas Híbridas

> Fundamentos matemáticos del O(1) context scaling y el hybrid Transformer+Mamba engine.

---

### 19. S4: Efficiently Modeling Long Sequences with Structured State Spaces ⭐
- **Autores:** Albert Gu, Karan Goel, Christopher Ré (Stanford University)
- **Año:** 2021
- **arXiv:** https://arxiv.org/abs/2111.00396

**Relevancia para MUD:**
Establece la representación dual convolucional/recurrente de los SSMs — paralela para training y recurrente para inferencia. Es el origen del "fixed-state SSM scan" para capas Mamba mandatado en `GEMINI.md`.

---

### 20. HiPPO: Recurrent Memory with Optimal Polynomial Projections ⭐
- **Autores:** Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, Christopher Ré
- **Año:** 2020
- **arXiv:** https://arxiv.org/abs/2008.07669

**Relevancia para MUD:**
Define la inicialización de la matriz de transición **A** basada en polinomios ortogonales. La variante `HiPPO-LegS` asegura **eigenvalores con parte real estrictamente negativa**, que es la base teórica directa para la verificación de estabilidad del UCP v2 (Paso 2) implementado en `conversion_verifier`.

---

### 21. Mamba: Linear-Time Sequence Modeling with Selective State Spaces ⭐
- **Autores:** Albert Gu (CMU), Tri Dao (Princeton University)
- **Año:** 2023
- **arXiv:** https://arxiv.org/abs/2312.00752

**Relevancia para MUD:**
**El paper de especificación primaria del SSM selectivo.** Introduce las matrices dependientes de los tokens ($\Delta$, $B$, $C$) y el escaneo recurrente. El kernel secuencial en MUD se basa directamente en esta arquitectura para lograr escalado de contexto $O(1)$.

---

### 22. Mamba-2 / SSD: Transformers are SSMs (State Space Duality) ⭐
- **Autores:** Tri Dao (Princeton), Albert Gu (CMU)
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2405.21060

**Relevancia para MUD:**
Formaliza el framework **SSD**, conectando matemáticamente SSMs con Attention. Permite el diseño de capas Mamba agrupadas por cabezas (multi-head). La actualización al scan paralelo por bloques de Mamba-2 (2-8× más rápido) es la base de la optimización del Roadmap para MUD.

---

### 23. Mamba-3: Improved Sequence Modeling using State Space Principles 🆕
- **Autores:** Aakash Lahoti, Kevin Y. Li, Berlin Chen, et al. (incluyendo Tri Dao, Albert Gu)
- **Año:** 2026 (Marzo)
- **arXiv:** https://arxiv.org/abs/2603.15569
- **Venue:** **ICLR 2026 (Oral)**

**Relevancia para MUD:**
Propone tres innovaciones cruciales para el Roadmap `MATH-03`:
1. *Discretización Exponencial-Trapezoidal:* Reemplaza Euler con un método de segundo orden, estabilizando numéricamente el escaneo en pesos cuantizados.
2. *Estados Complejos:* Captura dinámicas oscilatorias sin necesidad de codificación posicional.
3. *Formulación MIMO:* Procesa vectores de embedding en paralelo, aumentando la intensidad aritmética del kernel SIMD/AVX2.

---

### 24. Jamba: A Hybrid Transformer-Mamba Language Model ⭐
- **Autores:** Opher Lieber, Barak Lenz, Hofit Bata et al. (AI21 Labs)
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2403.19887

**Relevancia para MUD:**
**El plano de arquitectura híbrida de MUD.** Demuestra empíricamente que intercalar bloques Attention y Mamba SSM (+ expertos MoE) ofrece mejores métricas de calidad y throughput que modelos puros. Sus análisis de ratio de capas se usan para configurar `src/mud/inference.rs`.

---

### 25. iFairy: 2-bit Complex LLM con Parámetros en {±1, ±i}
- **Autores:** Yuxuan Zhang et al.
- **Año:** 2025 (Octubre)
- **arXiv:** https://arxiv.org/abs/2510.08865

**Relevancia para MUD:**
Usa un espacio de cuantización complejo `{±1, ±i}` en lugar del ternario tradicional. La multiplicación por `i` se reduce a swap + cambio de signo, lo cual es de costo cero con instrucciones vectoriales. Relevante para la integración de estados complejos de Mamba-3 en MUD.

---

### 26. Jamba-1.5: ExpertsInt8 Quantization para MoE+Mamba
- **Autores:** AI21 Labs
- **Año:** 2024 (Agosto)
- **arXiv:** https://arxiv.org/abs/2408.12570

**Relevancia para MUD:**
Introduce la cuantización híbrida: pesos de expertos MoE en INT8 mientras mantiene la lógica de ruteo y atención en precisión completa. Valida la filosofía de MUD sobre la protección diferenciada de parámetros por componente para contextos ultra-largos (256k tokens).

---

### 27. Slender-Mamba: Fully Quantized Mamba in 1.58 Bits From Head to Toe 🔥
- **Autores:** Yu, Kojima, Matsuo, Iwasawa (Universidad de Tokyo)
- **Año:** 2025 (Enero)
- **Venue:** COLING 2025
- **GitHub:** https://github.com/YU-ZHENXUAN-ucllm/Slender-Mamba

**Relevancia para MUD:**
**La validación científica más cercana a MUD.** Aplica QAT ternario (1.58-bit) con STE a un modelo Mamba-2 completo (incluyendo embeddings y proyecciones). Logra un 90% de reducción en el almacenamiento de parámetros con mínima pérdida de perplexity. Demuestra científicamente que el esquema QAT en Mamba es estable y coherente.

---

### 28. Quamba2: Quantization Framework for Mamba-1 and Mamba-2
- **Autores:** ICML Quantization Group
- **Año:** 2024 (ICML 2025)

**Relevancia para MUD:**
Demuestra que la cuantización PTQ (Post-Training) naive falla catastróficamente en modelos Mamba porque las matrices A, B y C dinámicas requieren cuantizarse con esquemas separados. Valida de forma independiente la decisión de MUD (Audit V3/V5) de pivotar de PTQ directo a QAT con STE.

---

## III. Mixture of Experts & Atención Eficiente

> Routing, load balancing, Flash Attention y kernels de atención cache-eficientes.

---

### 29. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- **Autores:** William Fedus, Barret Zoph, Noam Shazeer (Google Brain)
- **Año:** 2021
- **arXiv:** https://arxiv.org/abs/2101.03961

**Relevancia para MUD:**
Define el ruteo Top-1 y la penalización de balanceo de carga (Auxiliary Loss). Sus técnicas de estabilidad en precisión baja (dropout, clipping de gradientes) inspiran la sanitización de gradientes de shadow weights en `forge_autograd`.

---

### 30. GShard: Scaling Giant Models with Conditional Computation
- **Autores:** Dmitry Lepikhin, HyoukJoong Lee et al. (Google)
- **Año:** 2020
- **arXiv:** https://arxiv.org/abs/2006.16668

**Relevancia para MUD:**
Pionero en el ruteo Top-2 y la gestión de dispatch local de tokens hacia expertos. El fallback a ruteo random para expertos sobrecargados es aplicable en MUD para evitar el colapso del MoE con pocos parámetros.

---

### 31. Mixtral of Experts ⭐
- **Autores:** Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux et al. (Mistral AI)
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2401.04088

**Relevancia para MUD:**
Prueba que el routing Top-2 sobre 8 expertos segmentados alcanza calidad de modelo denso con menos cómputo activo por token. Su ruteador se implementa como referencia de lógica sin asignación en el `InferenceWorkspace` de MUD.

---

### 32. Hash Layers For Large Sparse Models
- **Autores:** Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston (Meta AI)
- **Año:** 2021
- **arXiv:** https://arxiv.org/abs/2106.04426

**Relevancia para MUD:**
El ruteo determinista basado en hash del token elimina parámetros del router y la pérdida auxiliar de balanceo. Es ideal para MUD porque tiene costo de memoria y asignación cero ($O(1)$ lookup estático), compatible con la Zero-Allocation Policy.

---

### 33. Expert Choice Routing
- **Autores:** Yanqi Zhou, Tao Lei, Hanxiao Liu et al. (Google)
- **Año:** 2022
- **arXiv:** https://arxiv.org/abs/2202.09368

**Relevancia para MUD:**
Invierte el ruteo: los expertos eligen tokens. Garantiza un balance de carga perfecto por diseño, eliminando la necesidad de balanceo auxiliar, crítico para evitar la inactividad de expertos en modelos pequeños.

---

### 34. DeepSeekMoE: Towards Ultimate Expert Specialization
- **Autores:** Damai Dai, Chengqi Deng, Harvey Jia, et al. (DeepSeek AI)
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2401.06066

**Relevancia para MUD:**
Propone segmentación fina de expertos y el uso de **expertos compartidos** (siempre activos). Los expertos compartidos ternarios absorben el conocimiento común redundante, reduciendo la entropía del router. La granularidad fina mejora las escalas PRQ.

---

### 35. ST-MoE: Designing Stable and Transferable Sparse Expert Models ⭐
- **Autores:** Barret Zoph, Irwan Bello, Sameer Kumar et al. (Google Brain)
- **Año:** 2022
- **arXiv:** https://arxiv.org/abs/2202.08906

**Relevancia para MUD:**
Introduce la pérdida **Router z-loss**:
$$L_z = \beta \cdot (\log \sum e^{logits})^2$$
Esta pérdida penaliza magnitudes grandes en los logits de salida del router, previniendo overflows numéricos en softmax durante el entrenamiento QAT/STE de MUD. **Es el ítem EDGE-03 del Roadmap.**

---

### 36. FlashAttention: Fast and Memory-Efficient Exact Attention
- **Autores:** Tri Dao, Daniel Y. Fu, Stefano Ermon, et al. (Stanford University)
- **Año:** 2022
- **arXiv:** https://arxiv.org/abs/2205.14135

### 37. FlashAttention-2: Faster Attention with Better Parallelism
- **Autores:** Tri Dao (Princeton)
- **Año:** 2023
- **arXiv:** https://arxiv.org/abs/2307.08691

### 38. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
- **Autores:** Jay Shah, Ganesh Bikshandi, Ying Zhang, et al.
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2407.08608

**Relevancia para MUD (FA-1/2/3):**
El principio de segmentación y tiling (cache-tiling) se aplica a las capas de atención del motor en CPU para mantenerse dentro de L2/L3 cache. Las técnicas de escalado y estabilidad matemática en precisión baja sirven para prevenir la degradación de precisión en la dequantización ternaria durante la acumulación intermedia.

---

## IV. Razonamiento Recursivo & Test-Time Training

> Deliberación iterativa, razonamiento latente y el framework RRM/LDT de la Fase 14.

---

### 39. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- **Autores:** Jason Wei, Xuezhi Wang, Dale Schuurmans et al. (Google)
- **Año:** 2022
- **arXiv:** https://arxiv.org/abs/2201.11903

**Relevancia para MUD:**
Demuestra que los tokens intermedios de razonamiento mejoran la precisión lógica. Fundamento del sistema `<thinking>` en la UI/CLI de MUD.

---

### 40. Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking ⭐
- **Autores:** Eric Zelikman, Georges Hasson, Yann Dubois, Noah D. Goodman (Stanford University)
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2403.09629

**Relevancia para MUD:**
Genera "thoughts" latentes en paralelo para cada token. Se conecta con `RRM-02` (Latent Imagination Asíncrona): los cores E de la CPU generan exploraciones especulativas mientras los cores P validan el flujo determinista.

---

### 41. Test-Time Training on Video (TTT) — Meta-Learning for Fast Adaptation
- **Autores:** Yossi Gandelsman, Yu Sun, Xinlei Chen, Alexei Efros
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2407.04468

**Relevancia para MUD:**
Actualiza una sub-red interna (pesos rápidos) durante la inferencia para capturar contexto ultra-largo. Referencia para `ALIGN-02: TTT Layers`. El buffer de entrenamiento en inferencia debe apegarse a la Zero-Allocation Policy.

---

### 42. Speculative Decoding: Accelerating LLM Inference via Speculative Execution ⭐
- **Autores:** Yaniv Leviathan, Matan Kalman, Yossi Matias (Google)
- **Año:** 2023
- **arXiv:** https://arxiv.org/abs/2211.17192

**Relevancia para MUD:**
Usa un modelo draft pequeño y rápido para proponer tokens, validados por el modelo maestro en un paso paralelo. En MUD, esto permite que un modelo ternario ultra-pequeño (ej. 135M) corra en hilos paralelos para guiar la inferencia de un modelo 4B de forma asíncrona.

---

### 43. DeepSeek-R1: Incentivizing Reasoning Capability via RL
- **Autores:** DeepSeek-AI Team
- **Año:** 2025
- **arXiv:** https://arxiv.org/abs/2501.12948

**Relevancia para MUD:**
Prueba que el comportamiento de "slow thinking" puede emerger por RL sin SFT. Valida la dirección del Roadmap Phase 14: es más viable implementar bucles de razonamiento recursivo latente en modelos de escala modesta que depender únicamente de la escala bruta de parámetros.

---

### 44. EAGLE: Speculative Sampling via Draft Model
- **Autores:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang (Microsoft Research)
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2401.15077

**Relevancia para MUD:**
Especulación a nivel de características vectoriales (hidden states) en lugar de tokens. En MUD, esto es extremadamente eficiente porque evita la tokenización y búsquedas en embeddings repetitivas, operando directo en la memoria pre-asignada del `InferenceWorkspace`.

---

### 45. Lattice-Based Deduction (LDT) — Referencia Teórica
- **Davey & Priestley:** *Introduction to Lattices and Order*, Cambridge University Press (2002).
- **Dave Jaffar & Michael Maher:** *Constraint Logic Programming* (1987).

**Relevancia para MUD:**
Establece los cimientos algebraicos para mapear hidden states continuos a elementos de un retículo lógico ordenado. Referencia para `LDT-01: Lattice Constraint Projections` para lograr deducciones 100% deterministas sin alucinación en modelos de sub-2M parámetros.

---

## V. Edge AI & Optimización SIMD

> Inferencia eficiente en CPU consumer y optimización del tamaño del KV Cache y la memoria de tokens.

---

### 46. LLM in a Flash: Efficient LLM Inference with Limited Memory ⭐
- **Autores:** Keivan Alizadeh, Iman Mirzadeh et al. (Apple)
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2312.11514

**Relevancia para MUD:**
Propone transferencias secuenciales óptimas de pesos desde flash usando mmap alineado. Justifica la pre-asignación y el mmap del `.mud` file, garantizando lecturas alineadas a 32 bytes para carga directa en registros vectoriales `ymm`.

---

### 47. PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU ⭐
- **Autores:** Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen (Shanghai Jiao Tong University)
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2312.12456

**Relevancia para MUD:**
Explotación de localidad de activación (hot/cold neurons). Las neuronas calientes se pre-cargan en memoria rápida (L3 caché en MUD), mientras las frías se cargan bajo demanda o se saltan por sparsity (26% de pesos en cero).

---

### 48. Efficient Streaming LLMs with Attention Sinks ⭐
- **Autores:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
- **Año:** 2023
- **arXiv:** https://arxiv.org/abs/2309.17453

**Relevancia para MUD:**
Establece que retener permanentemente los primeros 4 tokens en el KV cache circular (Attention Sinks) evita el colapso semántico de la atención softmax en secuencias largas. **Es el ítem EDGE-01 del Roadmap.**

---

### 49. RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE) ⭐
- **Autores:** Jianlin Su, Yu Lu, Shengfeng Pan et al.
- **Año:** 2021
- **arXiv:** https://arxiv.org/abs/2104.09864

**Relevancia para MUD:**
Aplica embeddings posicionales in-place sobre los vectores Q y K de forma directa sin dependencias dinámicas. Implementado nativamente en `src/asm/rope.s` con intrínsecos de rotación AVX2.

---

### 50. ALiBi: Train Short, Test Long — Attention with Linear Biases
- **Autores:** Ofir Press, Noah A. Smith, Mike Lewis
- **Año:** 2022
- **arXiv:** https://arxiv.org/abs/2108.12409

**Relevancia para MUD:**
Elimina las tablas de embedding posicionales añadiendo un sesgo escalar estático a los logits de atención. Es ideal para arquitecturas de inferencia zero-allocation porque evita indexación y memoria dinámica.

---

### 51. GQA: Training Generalized Multi-Query Transformer Models
- **Autores:** Joshua Ainslie, James Lee-Thorp et al. (Google Research)
- **Año:** 2023
- **arXiv:** https://arxiv.org/abs/2305.13245

**Relevancia para MUD:**
Agrupa las cabezas K y V reduciendo el tamaño físico del KV cache de manera lineal (típicamente 4-8×). Implementado en `InferenceWorkspace` para mantener el límite de memoria del buffer fijo.

---

### 52. BPE: Neural Machine Translation of Rare Words with Subword Units (Sennrich 2016)
- **Autores:** Rico Sennrich, Barry Haddow, Alexandra Birch
- **Año:** 2016
- **arXiv:** https://arxiv.org/abs/1508.07909

**Relevancia para MUD:**
Establece las bases lógicas de los tokenizadores de par de bytes (BPE). Utilizado para el diseño y corrección de la complejidad $O(n^2)$ del tokenizador nativo en `src/model/tokenizer.rs` para migrarlo a $O(n \log n)$ (`EDGE-05`).

---

### 53. Small Language Models Survey (2024)
- **Autores:** Survey Group
- **Año:** 2024
- **arXiv:** https://arxiv.org/abs/2409.15790

**Relevancia para MUD:**
Estudio de viabilidad de modelos sub-2B parámetros (Phi, SmolLM) en arquitecturas de hardware embebido. Valida la factibilidad de desplegar modelos de 0.5B a 4B en hardware modesto de manera eficiente.

---

*Índice Maestro Unificado de Papers de Forge LLM · 2026-06-04*
