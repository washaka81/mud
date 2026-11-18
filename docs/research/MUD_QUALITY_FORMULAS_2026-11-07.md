# MUD — Fórmulas para mejorar la calidad de predicción

**Fecha:** 2026-11-07
**Rol:** documento de investigación y cálculo. No sustituye al SSOT de ejecución
(`R-##`); alimenta el plan [`PLAN_QUALITY_2026-11-07.md`](../planning/PLAN_QUALITY_2026-11-07.md).
**Regla:** cada número de este documento se ejecutó o se deriva de una fórmula
explícita; las fuentes externas se citan al final.

---

## 0. Resumen ejecutivo

La auditoría de inferencia del 2026-11-07 dejó dos diagnósticos distintos:

| Modelo | Top-1 | PPL | Entropía | Lectura |
|---|---:|---:|---:|---|
| `ale.mud` (forge nano, 6L/128h) | 0/8 | 167,471.9 | 16.914 bits | **Casi uniforme** (`log2(151643)=17.210`; solo 0.296 bits por debajo) |
| `qwen_coder_0.5b_trained` (0.5B) | 0/2 | 17,788.0 | 7.084 bits | **Confiado pero incorrecto** (10.13 bits por debajo del uniforme) |

Por tanto, “subir el entropy” no es el objetivo: `ale` necesita capacidad/datos;
`qwen` necesita **ranking + calibración + cuantización/recuperación**. Las
fórmulas de este documento atacan las causas medibles:

1. **La escala ternaria actual está sesgada a la baja.** El conversor usa
   \(s=\mathrm{absmean}/\sqrt2\); BitNet b1.58 usa \(s=\mathrm{absmean}\) y el
   óptimo de mínimos cuadrados es aún mayor. En `token_embd` de Qwen medimos
   \(s^*/s=1.889\), NMSE \(0.391\to0.218\) y SNR \(4.08\to6.62\) dB.
2. **El umbral no es independiente de la escala.** El código actual usa
   \(\delta=0.7s\); BitNet usa \(\delta=0.5s_{\text{BitNet}}\). El umbral efectivo
   actual (\(0.495\,\mathrm{absmean}\)) coincide casi con BitNet
   (\(0.5\,\mathrm{absmean}\)), pero la escala de reconstrucción no.
3. **El MSE de pesos no es la métrica final.** Un deadzone mayor baja el MSE
   de reconstrucción en Gaussianas (0.9 da 0.196 vs 0.214 en 0.7), pero
   sobre-sparsifica. La decisión debe ser PPL/top-1, no solo MSE.
4. **El escalado post-hoc no arregla modelos ya entrenados.** Multiplicar los
   `prq_scale` por \(\sqrt2\) empeoró PPL: `ale` 167k→244k y Qwen 17.8k→20.7k.
   La corrección debe hacerse al **convertir/re-entrenar**, no al inferir.

---

## 1. Métricas y lo que realmente miden

Para \(N\) tokens, con logits \(z\in\mathbb R^V\) y objetivo \(y\):

\[
\mathrm{PPL}=\exp\!\left(-\frac1N\sum_{t=1}^N \log p_\theta(x_t\mid x_{<t})\right),
\qquad
H(p)=-\sum_{v=1}^V p_v\log_2 p_v .
\]

\[
\mathrm{rank}=\;1+\#\{v:z_v>z_y\},
\qquad
\mathrm{margin}=z_y-\max_{v\ne y}z_v,
\qquad
\mathrm{MRR}=\frac1N\sum_{i=1}^N\frac1{\mathrm{rank}_i}.
\]

**Caveat de reproducibilidad (F-13):** `MUD_INFER_TEMPERATURE=0` no significa
greedy en el código actual: `src/main.rs` filtra `>0` y cae a `0.8`. Para una
generación determinista usar `0.01`, o corregir el filtro (tarea de la próxima
sesión).

La **entropía máxima** es \(\log_2 V\). Para `ale` (\(V=151643\)),
\(\log_2 V=17.2103\) bits: una salida casi uniforme tiene PPL cercana a
\(V/\text{constante}\). Para `qwen` (\(V=151936\)), la misma cota. La distancia
\(H_{\max}-H\) es una medida de cuánta “decisión” toma el modelo, pero no de si
la decisión es correcta. Por eso medimos top-1, rank y margin junto a PPL.

**Calibración (a añadir al evaluador).** Expected Calibration Error:

\[
\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{N}
\left|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\right|,
\]

donde \(B_m\) agrupa predicciones por confianza. La temperatura óptima:

\[
T^*=\arg\min_{T>0}\;-\frac1N\sum_{i=1}^N
\log \mathrm{softmax}(z_i/T)_{y_i}.
\]

Temperatura no cambia el argmax; mejora NLL/ECE, no top-1. Útil para saber si
el modelo está “seguro y equivocado” (qwen) o “plano” (ale).

**Medido con `next_token_eval --temperature-scan` (2026-11-07):**

| Modelo | \(T^*\) | NLL(\(T^*\)) | PPL(\(T^*\)) | ECE(\(T=1\)) | ECE(\(T^*\)) |
|---|---:|---:|---:|---:|---:|
| `ale.mud` (8 prompts) | 2.0 | 11.928 | 151,396 | 0.000103 | 0.000027 |
| `qwen_coder_0.5b_trained` (2 prompts) | 1.4 | 9.111 | **9,055** | 0.264423 | **0.046** |

Lectura: Qwen mejora PPL 17,788→9,055 (≈2×) y ECE 0.264→0.046 al calibrar
con \(T=1.4\); **pero el top-1 sigue 0/2**. La temperatura no arregla el
ranking, solo la probabilidad. `ale` es casi uniforme: \(T=2\) apenas cambia
NLL (12.03→11.93), confirmando que el problema es señal, no calibración.

---

## 2. Cuantización ternaria: fórmulas y sesgo de escala

### 2.1 Definiciones

Cuantización simétrica con escala \(s>0\), umbral \(\delta>0\) y códigos
\(T_i\in\{-1,0,+1\}\):

\[
T_i=\begin{cases}
+1,& W_i>\delta,\\
0,& |W_i|\le\delta,\\
-1,& W_i<-\delta,
\end{cases}
\qquad
\hat W_i=s\,T_i .
\]

Error de reconstrucción y su versión normalizada:

\[
\mathcal E_w=\|W-\hat W\|_2^2=\sum_i(W_i-sT_i)^2,
\qquad
\mathrm{NMSE}=\frac{\mathcal E_w}{\sum_i W_i^2},
\qquad
\mathrm{SNR_{dB}}=10\log_{10}\frac{\sum_iW_i^2}{\mathcal E_w}.
\]

### 2.2 Fórmulas en juego

| Fuente | Escala | Umbral | Nota |
|---|---|---|---|
| **MUD actual** (`quantizer.rs`, `checkpoint.rs`) | \(s_c=\gamma/\sqrt2\), \(\gamma=\frac1n\sum|W_i|\) | \(\delta_c=0.7s_c=0.495\gamma\) | Implementación real |
| **BitNet b1.58** | \(s_B=\gamma\) | \(\delta_B=0.5s_B=0.5\gamma\) | `RoundClip(W/(γ+ε),-1,1)` |
| **TWN / LSQ óptimo para \(T\) fijo** | \(s^*=\dfrac{\sum_iT_iW_i}{\sum_i|T_i|}\) | el que fija \(T\) | Mínimos cuadrados |
| **TWN aproximado** | \(\alpha=\dfrac{\sum T_iW_i}{\sum|T_i|}\) | \(\Delta\approx0.75\gamma\) | Artículo original |
| **CAT-Q / PT²-LLM** | aprendida | aprendida | Sección 6 |

BitNet b1.58 define exactamente
\(\tilde W=\mathrm{RoundClip}(W/(\gamma+\epsilon),-1,1)\), con
\(\gamma=\frac1{nm}\sum_{ij}|W_{ij}|\). No hay división por \(\sqrt2\).
Nuestro conversor **sí** divide por \(\sqrt2\) y luego usa deadzone 0.7.

### 2.3 Derivación del óptimo para \(T\) fijo

Con \(T\) fijo, \(\mathcal E_w(s)=\sum_i(W_i-sT_i)^2\). Derivando:

\[
\frac{\partial\mathcal E_w}{\partial s}
=-2\sum_iT_i(W_i-sT_i)=0
\;\Longrightarrow\;
s^*=\frac{\sum_iT_iW_i}{\sum_iT_i^2}
=\frac{\sum_iT_iW_i}{\sum_i|T_i|}.
\]

Es la fórmula que usan TWN/LSQ/CAT-Q/PT²-LLM para reajustar la escala. Nuestro
código usa \(\gamma/\sqrt2\) en vez de \(s^*\), y nunca reajusta la escala al
conjunto de códigos.

### 2.4 Cálculo sobre modelos reales

Herramienta: `quant_formula_audit` (nueva, `./mud.sh quant-audit`).
Muestreo de 256 filas de `token_embd` F32 de
`qwen_coder_0.5b_trained.mud` (896 columnas/fila):

```text
GLOBAL F32 sample rows=256 kurt=0.42
s_cur=0.00948  s*_cur=0.01791  ratio=1.889
mse_cur=1.128e-4  mse*_cur=6.281e-5
mse_bitnet=7.705e-5  mse*_bitnet=6.256e-5
nmse_cur=0.3908  nmse*_cur=0.2177
snr_cur=4.08 dB  snr*_cur=6.62 dB
nz_cur=0.69  nz_bitnet=0.69
```

Interpretación:

- La escala actual \(s_c=\gamma/\sqrt2\) es **1.889× menor** que el óptimo
  least-squares para los mismos códigos.
- El NMSE baja de **0.391 a 0.218** (44 % menos error); el SNR sube
  **2.54 dB**.
- La fórmula BitNet exacta (\(s=\gamma\), \(\delta=0.5\gamma\)) da
  `mse_bitnet=7.7e-5`, mejor que la actual (`1.128e-4`) pero peor que el
  reajuste LSQ (`6.28e-5`).

**Validación analítica (Gaussiana \(N(0,1)\)).** Con
\(\gamma=\sqrt{2/\pi}=0.7979\), \(s_c=\gamma/\sqrt2=0.5642\),
\(\delta=0.7s_c=0.3949\):

\[
s^*=\frac{\mathbb E[|W|\mathbf 1_{|W|>\delta}]}{\mathbb P(|W|>\delta)}
=\frac{\phi(\delta)}{1-\Phi(\delta)}
=1.0651,
\qquad
\frac{s^*}{s_c}=1.8879.
\]

El cálculo numérico da NMSE actual \(0.3878\), NMSE óptimo \(0.2139\),
SNR \(4.11\to6.70\) dB. Coincide con lo medido en Qwen (1.889, 0.391,
0.218, 4.08, 6.62). Es decir, **las filas de `token_embd` se comportan como
Gaussianas y la fórmula predice el error real**.

### 2.5 BitNet vs LSQ vs actual (Gaussiana)

| Fórmula | \(s\) | \(\delta\) | NMSE | SNR |
|---|---:|---:|---:|---:|
| MUD actual | \(0.5642\) | \(0.3949\) | 0.3878 | 4.11 dB |
| BitNet b1.58 | \(0.7979\) | \(0.3989\) | 0.2634 | 5.79 dB |
| Actual + LSQ re-fit | \(1.0651\) | \(0.3949\) | 0.2139 | 6.70 dB |
| BitNet + LSQ re-fit | \(1.0680\) | \(0.3989\) | 0.2130 | 6.72 dB |

Conclusión: **cambiar a BitNet exacto ya reduce el MSE ~32 %**; reajustar la
escala al conjunto de códigos reduce ~45 %. Pero…

### 2.6 El MSE no es la calidad final

El deadzone barrido con reajuste de escala en Gaussiana:

| deadzone | \(\delta\) | \(s^*\) | no-cero | NMSE | SNR |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.2821 | 0.9857 | 0.778 | 0.2442 | 6.12 dB |
| 0.7 | 0.3949 | 1.0651 | 0.693 | 0.2139 | 6.70 dB |
| 0.9 | 0.5078 | 1.1468 | 0.612 | 0.1957 | 7.08 dB |

El MSE baja monótonamente al aumentar el deadzone porque la escala crece para
ajustar menos pesos. Eso **no** significa que el modelo sea mejor: se pierde
información y se sobre-sparsifica. La decisión correcta es output-aware
(PT²-LLM AGA / CAT-Q) o PPL/top-1, no MSE puro.

### 2.7 Prueba post-hoc (por qué no basta cambiar scales al inferir)

Se multiplicaron todos los `prq_scale` de modelos ya entrenados por
\(\sqrt2\) y \(1.889\) (`scale_rescale`), sin tocar los trits:

| Modelo | Factor | Top-1 | PPL | Entropía |
|---|---:|---:|---:|---:|
| `ale.mud` | 1.0 | 0/8 | 167,471.9 | 16.914 |
| `ale.mud` | 1.414 | 0/8 | 244,336.6 | 16.618 |
| `ale.mud` | 1.889 | 0/8 | 381,500.4 | 16.151 |
| `qwen_coder_0.5b_trained` | 1.0 | 0/2 | 17,788.0 | 7.084 |
| `qwen_coder_0.5b_trained` | 1.414 | 0/2 | 20,697.6 | 6.678 |

**Lección:** la escala actual está absorbida por el entrenamiento; cambiarla
post-hoc rompe la calibración de activaciones. La corrección de escala debe
hacerse **en la conversión/forja + QAT**, no en inferencia.

---

## 3. STE, deadzone y QAT

### 3.1 Straight-Through Estimator

Con \(\hat W=s\,T\) y \(T\) no diferenciable:

\[
\frac{\partial\mathcal L}{\partial W}
\approx
\frac{\partial\mathcal L}{\partial\hat W}
\;\cdot\;
\frac{\partial\hat W}{\partial W},
\qquad
\frac{\partial\hat W}{\partial W}\approx 1
\;\;(\text{STE}).
\]

Nuestro `pack_ternary_into` aplica `delta = deadzone * s`; el gradiente del
shadow se usa directamente (STE). El deadzone 0.7 crea una zona muerta extra:
pesos con \(|W|\in(0.5s,0.7s]\) no reciben código ±1 aunque el gradiente los
empuje. Esto puede ralentizar la recuperación.

### 3.2 LSQ: aprender la escala

LSQ hace \(s\) entrenable y usa:

\[
\frac{\partial \hat W_i}{\partial s}
=
\begin{cases}
\mathrm{round}(W_i/s)-W_i/s, & |W_i/s|\le Q,\\
\mathrm{sign}(W_i), & |W_i/s|>Q,
\end{cases}
\]

escalado por \(1/\sqrt{NQ}\) para estabilizar. En ternario \(Q=1\), el umbral
es \(s/2\). Aprender \(s\) por fila (o por bloque) es la forma más directa de
corregir el sesgo de la sección 2 sin cambiar la arquitectura.

### 3.3 CAT-Q: ternarización suavizada

CAT-Q (ICML 2026) propone una transición diferenciable:

\[
f(W;s,\Delta)=
\frac{\tanh\!\big(s(W-\Delta)\big)+\tanh\!\big(s(W+\Delta)\big)}
{2\tanh(s)},
\]

con \(s\to0\) identidad y \(s\to\infty\) ternario duro. Además modula

\[
\hat W=\frac{W-\mu}{\alpha},\qquad
\mu=\mu_0+\delta_\mu\alpha_0,\qquad
\alpha=\delta_\alpha\alpha_0,
\]

y \(\Delta=\delta_\Delta\Delta_0\), con \(\Delta_0=0.5\). El schedule usa
\(\gamma=0.8\) (80 % de épocas diferenciables), \(s_0=30\), 512 muestras de
calibración y reconstrucción por ventana deslizante de capas. Es la receta PTQ
más directamente aplicable a MUD si no se quiere QAT largo.

### 3.4 Annealing de deadzone

Nuestra SSOT es `TERNARY_DEADZONE=0.7`. Una política razonable es **annealing**
durante QAT: empezar en 0.5 (más señal, como BitNet) y subir a 0.7 al final
(menos ruido, más sparsity), o al revés según la tarea. Medir PPL/top-1 en cada
fase; no fijar 0.7 por defecto sin evidencia downstream.

---

## 4. Rotaciones e outliers

### 4.1 Hadamard / QuaRot / SpinQuant

Una rotación ortogonal \(R\) (\(R^\top R=I\)) preserva el producto:

\[
Y=XW^\top=(XR)(R^\top W^\top).
\]

Si \(R\) es Hadamard normalizada \(H\), mezcla columnas y reduce outliers
(„incoherencia”). QuaRot inserta 4 rotaciones \(R_1..R_4\); SpinQuant aprende
\(R\) con una parametrización ortogonal (Cayley). BitNet v2 aplica una Hadamard
online antes de cuantizar activaciones (H-BitLinear).

**Métrica de outlier:** exceso de kurtosis

\[
\mathrm{Kurt}(x)=\frac{\mathbb E[(x-\mu)^4]}{\sigma^4}-3.
\]

Gaussiana = 0; colas pesadas > 0. En `token_embd` de Qwen medimos
`kurt=0.42` (256 filas), no es un caso extremo. La rotación suele ayudar más a
**activaciones** que a embeddings; hay que medir kurtosis de activaciones por
capa antes de invertir en R-52.

### 4.2 BitNet v2

H-BitLinear:

\[
X' = \mathrm{Hadamard}(X),\qquad
Q_{4bit}(X') = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{X'}{s}\right),-8,7\right),
\]

con \(s\) absmax por token. La Hadamard “suaviza” la distribución para que
4-bit sea viable. Para MUD, el equivalente es aplicar Hadamard antes de
ternarizar pesos (R-52) y/o antes de cuantizar activaciones (futuro).

---

## 5. Destilación y recuperación

### 5.1 BitNet Distillation (BitDistill)

Tres etapas: SubLN, continued pre-training, y destilación:

\[
\mathcal L=\mathcal L_{CE}+\lambda\mathcal L_{LD}+\gamma\mathcal L_{AD},
\]

\[
\mathcal L_{LD}=\frac1N\sum_i
D_{KL}\!\left(P_{\text{teacher}}^{\tau}(\cdot\mid x_i)
\,\|\,P_{\text{student}}^{\tau}(\cdot\mid x_i)\right),
\qquad
P^\tau(y)=\frac{e^{z_y/\tau}}{\sum_{y'}e^{z_{y'}/\tau}},
\]

\[
\mathcal L_{AD}=\frac1{|\Upsilon|}\sum_{i\in\Upsilon}\sum_{j\in\{Q,K,V\}}
\alpha_i\frac1{A_r|\mathbf x|}
\sum_{a,t}D_{KL}\!\left(R^{T}_{i,j,a,t}\,\|\,R^{S}_{i,j,a,t}\right),
\]

\[
R_{i,j,a,t}=\mathrm{softmax}\!\left(\frac{A_{i,j,a,t}A_{i,j,a,t}^\top}{\sqrt{d_r}}\right).
\]

Valores reportados: \(\tau=5.0\); clasificación \(\lambda=10,\gamma=10^5\);
summarization \(\lambda=1,\gamma=10^3\); una sola capa para \(\mathcal L_{AD}\)
(las últimas capas funcionan mejor); 10B tokens de continued pre-training.
El resultado es que un 1.58-bit iguala a FP16 en tareas downstream.

### 5.2 Token-Scaled Logit Distillation (TSLD)

Para modelos generativos, escalar la pérdida por token (en vez de media)
mejora la destilación ternaria; es la referencia de TernaryLLM. La idea:
\(\mathcal L_{LD}\) debe ponderar tokens difíciles, no promediar todos por
igual.

### 5.3 Coste en MUD

No tenemos GPU dedicada; BitDistill completo es caro. Pero **podemos hacer una
versión mínima**:
- teacher = `qwen_coder_0.5b.mud` F32 (mismo tokenizer),
- student = ternario,
- logit KD sobre corpus project (no 10B tokens),
- \(\tau=2..5\), \(\lambda=0.5..2\),
- medir PPL/top-1 cada 250 pasos.
Es la vía más prometedora para qwen; para `ale` (forge 6L/128h) no hay teacher
equivalente y la capacidad es el límite.

---

## 6. Embeddings y cabeza (R-53)

Para una capa \(Y=XW^\top\), el error de salida se acota por

\[
\|\Delta Y\|_2\le \|W\|_2\|\Delta X\|_2+\|\Delta W\|_2\|X\|_2.
\]

En el **LM head**, un error \(\Delta W_{\text{head}}\) produce
\(\Delta z=\Delta W_{\text{head}}h\); el argmax es muy sensible porque el
espacio de logits es de 150k dimensiones y el margen entre tokens suele ser
pequeño. En los **embeddings**, cada token de entrada pasa por la misma matriz;
un error de cuantización se amplifica en todas las posiciones.

Por eso BitNet y la práctica estándar dejan embeddings/head en mayor precisión.
`universal_converter` ya deja `token_embd`/`output` en F32 por defecto;
`mud_forge` los hace ternarios. Medido en Qwen: `token_embd` F32, pero si se
ternarizara con la fórmula actual el NMSE sería **0.375** (SNR 4.27 dB), es
decir, se perdería más de un tercio de la energía de la señal. **No ternarizar
emb/head es una decisión correcta**; el problema de `ale` es que el forge los
fuerza a ternary.

---

## 7. PT²-LLM: ternarización post-entrenamiento

PT²-LLM (2025) hace ternarización asimétrica:

\[
\hat W=\alpha T+\mu,\qquad
\mu=\frac1m\sum_j W_{:,j},
\]

y resuelve la grilla óptima por fila:

\[
\alpha^*=\frac{m\sum(W\circ T)-(\sum T)(\sum W)}
{m\sum T^2-(\sum T)^2},
\qquad
\mu^*=\frac{(\sum T^2)(\sum W)-(\sum T)\sum(WT)}
{m\sum T^2-(\sum T)^2}.
\]

Luego actualiza \(T\) por redondeo flexible:

\[
T_{ij}^*=\arg\min_{t\in\{-1,0,1\}}\left|\frac{W_{ij}-\mu_i^*}{\alpha_i^*}-t\right|.
\]

La fase **AGA** reemplaza la covarianza de pesos por \(C=\sum X X^\top\) y
resuelve el mismo sistema para minimizar \(\|WX-\hat WX\|_F^2\). La fase
**SSR** reordena columnas por similitud coseno:

\[
S_{ij}=\frac{W_{:,i}^\top W_{:,j}}{\|W_{:,i}\|_2\|W_{:,j}\|_2}.
\]

Reporta PPL WikiText2 de 11.56 en LLaMA-2-7B (vs GPTQ 2-bit 129.19, AWQ
2.6e5) y 32.19 en LLaMA-3-8B. Es la receta PTQ más fuerte para aplicar si no
se puede hacer QAT largo.

---

## 8. Mapa fórmula → acción (prioridad)

| Prioridad | Acción | Fórmula | Coste | Impacto esperado |
|---|---|---|---|---|
| **P0** | Cambiar escala de conversión a BitNet exacto o LSQ re-fit | \(s=\gamma\), \(\delta=0.5\gamma\) o \(s^*=\sum TW/\sum|T|\) | Bajo (converter) | NMSE −32…−45 %; PPL a medir |
| **P0** | No cambiar scales post-hoc en inferencia | — | Cero | Evita regresión (medido: PPL empeora) |
| **P0** | Medir ECE/T* y kurtosis de activaciones | ECE, \(T^*\), Kurt | Bajo (tools) | Diagnóstico fino |
| **P1** | R-53: emb/head F32 en forge y modelos reales | \(\|\Delta Y\|\le\|W\|\|\Delta X\|+\|\Delta W\|\|X\|\) | Bajo-Medio | Recupera resolución léxica |
| **P1** | QAT largo + KD (BitDistill mínimo) | \(\mathcal L=\mathcal L_{CE}+\lambda\mathcal L_{LD}+\gamma\mathcal L_{AD}\) | Alto (CPU) | La vía real para qwen |
| **P1** | Annealing de deadzone 0.5↔0.7 | STE + schedule | Medio | Recuperación más estable |
| **P2** | Hadamard/FWHT antes de ternarizar (R-52) | \(Y=(XR)(R^\top W^\top)\) | Medio | Outliers/activaciones |
| **P2** | CAT-Q/PT²-LLM ITF+AGA+SSR | §3.3/§7 | Medio-Alto | Mejor PTQ sin QAT |
| **P3** | Temperature scaling/ECE en eval | \(T^*\), ECE | Bajo | Honestidad probabilística |
| **P3** | 5 trits/byte (R-42) | — | Alto | Eficiencia, no calidad |

---

## 9. Referencias

- Ma et al., **BitNet b1.58 2B4T Technical Report**, arXiv:2504.12285 (2025). Absmean: \( \tilde W=\mathrm{RoundClip}(W/\gamma,-1,1)\), \(\gamma=\frac1{nm}\sum|W_{ij}|\).
- Ma et al., **The Era of 1-bit LLMs: All LLMs are in 1.58 Bits**, arXiv:2402.17764 (2024). Fórmula absmean original.
- Yan et al., **PT²-LLM: Post-Training Ternarization for LLMs**, arXiv:2510.03267 (2025). ATQ/ITF/AGA/SSR.
- Wang et al., **CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs**, arXiv:2606.26650 (ICML 2026). LM/ST/sliding-layer.
- Wu et al., **BitNet Distillation**, arXiv:2510.13998 (2025). SubLN + continued pretraining + logit/attention KD.
- Esser et al., **Learned Step Size Quantization**, arXiv:1902.08153 (ICLR 2020). Gradiente de \(s\).
- Ashkboos et al., **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs**, arXiv:2404.00456 (2024). Rotaciones Hadamard.
- Liu et al., **SpinQuant: LLM Quantization with Learned Rotations**, arXiv:2405.16406 (2024). Rotaciones aprendidas.
- Wang et al., **BitNet v2: Native 4-bit Activations with Hadamard Transformation**, arXiv:2504.18415 (2025). H-BitLinear.
- Kim et al., **Token-Scaled Logit Distillation for Ternary Weight Generative LMs**, NeurIPS 2023. TSLD.
- Guo et al., **On Calibration of Modern Neural Networks**, ICML 2017. Temperature scaling/ECE.
