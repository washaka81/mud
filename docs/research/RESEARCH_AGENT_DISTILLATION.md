# Investigación: Agentic Workflow Distillation (The Subterranean Agent)

## 1. El Paradigma Actual vs. Propuesta
Actualmente, los frameworks de agentes (como LangChain, AutoGen) ejecutan un ciclo de orquestación en Python sobre un modelo base, pagando el costo de latencia, contexto y I/O de red en cada iteración del ciclo de razonamiento (ej: planificar -> herramienta -> error -> replanificar).
El documento *"Compiling Agentic Workflows into Weights"* propone destilar (compilar) toda esa lógica iterativa directamente en los parámetros del modelo. Esto convierte al orquestador en un "agente subterráneo", ejecutándose en una sola pasada hacia adelante con un costo de cómputo 100x menor.

## 2. Compatibilidad con MUD y QAT (Quantization-Aware Training)
En **Forge LLM**, ya entrenamos los modelos usando STE QAT (Straight-Through Estimator Quantization-Aware Training) de 1.58-bits. Podemos reutilizar este loop de propagación de error no solo para enseñar tokens del corpus, sino para forzar a la red a imitar "trazas de razonamiento" (reasoning traces).

### Cómo lograrlo en MUD:
1. **Recolección de Trazas (The Orchestrator Trace):**
   Registramos las interacciones exitosas de un modelo superior (Frontier Model) o un script heurístico resolviendo un problema iterativo. El registro incluye el uso de herramientas, análisis de fallos y *scratchpads*.
2. **Entrenamiento de Destilación (Distill-QAT):**
   Cargamos estas trazas en el `MudCorpusTrainer`. El modelo de 1.58-bits es obligado a predecir las acciones correctas y generar los bloques lógicos bajo el estricto clamp `[-1, 0, 1]`.
3. **Internalización del Routing:**
   En nuestro caso, el `MudRouter` (el orquestador de expertos que acabamos de optimizar) servirá también para enrutar el pensamiento: algunos expertos se especializarán en "LLamada a función" y otros en "Síntesis", de forma emergente a medida que la entropía (Delta-Sigma) decrece durante la destilación.

## 3. Hoja de Ruta de Implementación
- [ ] Implementar `MudCorpusTrainer::distill_workflow()` para cargar un JSONL con trazas agentiles y forzar STE QAT sobre ellas.
- [ ] Ajustar el *Weight Decay* (Epsilon/Lambda) dinámico en este proceso, ya que la destilación de flujos requiere de más plasticidad estructural que la memorización de datos del corpus común.
- [ ] Probar compilando un loop de "Self-Correction" simple en los pesos del Mamba Hybrid.
