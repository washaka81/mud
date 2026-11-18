# Research — Calibración de tokens con Laya (convaiinnovations/laya)

## 1. Qué es Laya
- Modelo no-autoregresivo (bidirectional encoder, ModernBERT/mmBERT) + decision head.
- Devuelve probabilidades calibradas por opción en un solo forward pass.
- Entrenado con **RLCD** (Reinforcement Learning against strictly proper scoring rules): el reward es la log-verosimilitud del target, por lo que reportar la probabilidad verdadera es la única estrategia óptima (proper scoring rule).
- Ships uncalibrated: `temperature = [1.0,1.0,1.0]`. Refit de temperatura por (tipo de pregunta, número de opciones) reduce ECE 0.314 → 0.106 (multilingual) y 0.466 → 0.081 (english).

## 2. Matemática relevante para Forge LLM

### 2.1 Proper scoring rules (RLCD)
Para un target y ∈ {1..V} y distribución predicha p, el score propio (log score) es
  S(p, y) = log p_y.
Maximizar E[S] equivale a minimizar la entropía cruzada (NLL):
  L = - (1/N) Σ log p_{y_i}.
Por tanto el entrenamiento STE QAT ya optimiza un proper scoring rule. Laya aporta la idea de **recompensar la honestidad probabilística**, no sólo el argmax.

### 2.2 Temperature scaling
p_i(T) = softmax(z_i / T).
T se ajusta minimizando NLL sobre un conjunto de validación:
  T* = argmin_T  -Σ log p_{y_i}(T).
Con T>1 se reduce la confianza (reduce overconfidence). Laya usa buckets por (tipo, #opciones) porque la calibración depende del tamaño del espacio de decisión.

### 2.3 Expected Calibration Error (ECE)
ECE = Σ_b (n_b/N) |acc(b) − conf(b)|,
donde se agrupan predicciones por confianza en bins. Mide la discrepancia entre confianza y precisión. Laya reporta ECE macro; nuestro `inference_coherence` ya reporta entropía y margen, pero no ECE.

### 2.4 Aplicación a calibración de tokens en Forge LLM
- Nuestro modelo emite logits z ∈ R^V sobre vocabulario V. La probabilidad del siguiente token es p = softmax(z/T).
- Hoy T=1 (implícito). El baseline muestra entropía ~11.72 bits con V=151643 (max ~17.2 bits), es decir distribución casi uniforme pero con overconfidence en el top-1 incorrecto (margen +2.7 a +3.5).
- Calibrar T sobre un corpus de validación puede:
  (a) reducir la confianza del top-1 incorrecto (evitar alucinación),
  (b) mejorar la coherencia de sampling (nucleus),
  (c) dar una señal honesta para reranking/self-play (D1).
- Calibración por bucket: distintos T para prompts cortos vs largos, o por dominio (código vs texto), análogo a (tipo, #opciones) de Laya.

### 2.5 Protocolo propuesto (P-Laya)
1. Recolectar un corpus de validación (prompt, target next-token).
2. Calcular logits z para cada prompt (reusar forward_last_logits).
3. Optimizar T (escalar) minimizando NLL sobre el corpus (1D, convexo, bisección/Newton).
4. Calcular ECE antes/después para medir mejora.
5. Opcional: T por bucket (longitud del prompt, dominio) si hay suficientes muestras.
6. Exponer `MUD_TEMPERATURE` (default 1.0) en inference y training (sampling).

### 2.6 Límites
- Laya es no-generativo; no aporta logits de siguiente token. Aporta la metodología de calibración (temperature scaling + ECE + proper scoring rules), no pesos.
- La calibración no corrige un modelo que no sabe (0% top-1). Primero mejorar la calidad (R-50/R-52/R-53/QAT); la calibración es la capa final para honestidad probabilística.

## 3. Acción sugerida
- Implementar `MUD_TEMPERATURE` en `inference.rs`/`main.rs` (dividir logits por T antes de softmax).
- Añadir bin `calibrate_temperature` que optimiza T sobre un corpus y reporta ECE.
- Añadir ECE al reporte de `inference_coherence`.
EOF
echo done