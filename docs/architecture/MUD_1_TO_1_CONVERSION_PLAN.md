# Plan Maestro de Conversión 1:1 a Formato Ternario (.mud)

Este plan establece el procedimiento arquitectónico definitivo para convertir cualquier modelo LLM (FP16/BF16) al formato discreto 1.58-bit (Ternario) de M.U.D., garantizando una retención semántica y lógica lo más cercana posible al 1:1 original, mitigando por completo el "Ternary Shock".

## FASE 1: Cuantización Per-Row (PRQ) con Amortiguación de Profundidad
La conversión estática estricta (PTQ) genera "Afasia Semántica". Para evitar el colapso inicial de la matriz matricial, la extracción de pesos se manipula mediante amortiguación.

1. **Extracción de Pesos Base**: Leer tensores en formato original (GGUF/Safetensors).
2. **Aplicación del Factor de Amortiguación (`0.7071`)**: Durante la cuantización PRQ (Per-Row Quantization), se debe multiplicar el `absmean` por `0.7071` (1/sqrt(2)). Esto previene la paradoja del "Target Sigma" y empuja el modelo a la frontera óptima de esparsidad del 26.0%.
3. **Fijación de Epsilon (`1e-8`)**: Asegurar que todas las divisiones de caché KV y capas RMS Normalization respeten un suelo estricto de `1e-8` para prevenir el colapso de divisores en coma flotante.
4. **Salida Inicial**: Empaquetado de pesos sombra iniciales en el formato `.mud`.

## FASE 2: Alineación por Destilación de Onda Holográfica (Holographic Wave Distillation)
Aquí se recupera el delta de inteligencia perdido en el redondeo numérico, obligando al modelo ternario a "imitar la onda" del modelo maestro continuo.

1. **Extracción de la Fase Sinusoidal Maestra**: Congelar el modelo FP16 original en memoria y extraer su tensor de activación continuo (la onda) para una distribución representativa de tokens.
2. **Inyección de Sub-LayerNorm (BitDistill)**: Insertar operaciones Sub-LayerNorm automáticamente antes de `W_MHSA_out` y `W_FFN_down` para preparar el espacio latente.
3. **Medición de Similitud del Coseno**: Calcular el error de fase (KL-Divergence) entre la onda de activación continua (Master) y la onda ternaria escalonada (Student).
4. **STE Backpropagation (Derivadas)**: Propagar las derivadas de este error de fase usando el *Straight-Through Estimator* (STE) de vuelta a las escalas globales ($\gamma$) del modelo ternario, forzando a los límites discretos (+1, 0, -1) a acoplarse con la geometría original de la onda.

## FASE 3: Asentamiento Termodinámico (QAT Deep Cycle)
Para grabar los cambios de la Destilación Holográfica permanentemente en el modelo:

1. **QAT de Ciclo Profundo (`--full-qat`)**: A diferencia de las conversiones antiguas que usaban PTQ o L-QAT rápido, ejecutar un ciclo profundo de QAT usando el *Native Corpus Aligner* para asentar matemáticamente los embeddings BPE sobre la cuadrícula ternaria.
2. **Inyección de Jitter (`NEURAL_KICK_JITTER = 1e-5`)**: Añadir micro-ruido durante el asiento para evitar que el modelo colapse en un atractor determinista negativo.
3. **Decaimiento Dinámico de Pesos ($\lambda$)**: Utilizar el estimador de carga de trabajo (`training_estimator`) para calcular pasos SGD y $\lambda$ óptimos antes de aplicar el gradiente.
4. **Sanitización de Gradientes**: Todo gradiente antes de aplicarse a los pesos sombra (FP32) debe someterse a `is_finite()` y un clampeo duro a `[-1.0, 0.0, 1.0]`.

## FASE 4: Verificación Matemática y Endoso (UCP v2)
Ningún modelo puede considerarse convertido exitosamente sin certificar su homeostasis matemática.

1. **Validación de Estabilidad de Eigenvalores (HiPPO)**: Ejecutar `conversion_verifier` para confirmar que los eigenvalores mantengan una parte real estrictamente negativa (evitando divergencia infinita en el escaneo SSM/Mamba).
2. **Aserción de SQNR**: Garantizar que el modelo mantenga un Ratio Señal-Ruido de Cuantización $\ge 10.5$ dB.
3. **Seguridad de Frontera (`boundary_validator`)**: Verificar que no existan pesos fraccionales filtrados y que la covarianza de escala (COV) sea menor a 0.12.
4. **Acreditación Final (`iteration_validator`)**: Comprobar que el puntaje compuesto sea $\ge 96\%$ de similitud geométrica con el modelo original.

---
**Nota sobre Transformadas Avanzadas (Investigación HBLLM):** 
Aunque la versión actual se apoya 100% en *Holographic Wave Distillation*, la incorporación teórica de **Transformadas Wavelet de Haar** (referenciada en la investigación del proyecto) durante la Fase 1 podría implementarse a futuro para ayudar a descomponer las frecuencias paramétricas de manera más fina antes del empaquetado, reduciendo el trabajo de retropropagación requerido en la Fase 2.
