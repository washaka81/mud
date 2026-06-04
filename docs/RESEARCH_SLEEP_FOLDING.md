# Investigación: SSM Context Consolidation (Sleep & Fold)

## 1. El Problema: El Muro de Memoria (Memory Wall) del KV-Cache
Actualmente, los modelos Transformers tradicionales requieren almacenar todo el historial de la conversación (Key-Value Cache) en RAM para evitar recalcular la atención desde cero en cada token. A medida que la conversación se extiende, el costo de memoria y computo crece de forma cuadrática ($O(N^2)$), lo que vuelve inviable tener agentes autónomos operando perpetuamente.

## 2. La Solución "Language Models Need Sleep"
Inspirado en los últimos avances de investigación (May 2026), proponemos implementar un mecanismo de "sueño" para el motor de **Forge LLM (MUD)**.

La arquitectura *Jamba Hybrid* de MUD (ratio 1:5 de capas Transformer y Mamba) es la estructura perfecta para esto. Mientras que la Atención sufre el costo del historial infinito, el modelo **Mamba SSM** comprime la información en un estado recurrente fijo (Fast Weights) operando en $O(1)$.

### El Ciclo Wake / Sleep:
1. **Wake Phase (Fase de Vigilia):**
   - El modelo atiende instrucciones, genera tokens y almacena el KV-cache localmente para un razonamiento nítido a corto plazo.
   - El desempeño es súper rápido gracias a la política Zero-Allocation.
2. **Sleep Phase (Fase de Sueño / Consolidación):**
   - Cuando el tamaño del KV-cache excede un umbral (o el agente queda inactivo), se invoca `engine.sleep()`.
   - **Context Folding:** El motor realiza una serie de pasadas recurrentes en *offline* sobre el KV-cache acumulado para "plegar" y destilar esa información dentro de los estados ocultos (`mamba_conv_state`, `mamba_a_bar`, `mamba_b_bar`) de las capas Mamba.
   - **Cache Flush:** Una vez finalizada la consolidación (absorción por Fast Weights locales), el KV-cache de las capas de Atención se borra de la memoria (`kv_cache.clear()`).
   - El agente se despierta sin memoria RAM ocupada, pero reteniendo el "recuerdo" estructural de la conversación en su estado de Mamba.

## 3. Hoja de Ruta de Implementación en MUD (`src/mud/inference.rs`)
Para materializar esta tecnología, requeriremos las siguientes modificaciones:

* **Paso 1: Modificar `InferenceWorkspace`**
  - Implementar rutinas seguras (Zero-Allocation) para limpiar (`flush()`) selectivamente los búferes de KV de Atención sin reasignar memoria.
* **Paso 2: Desarrollar `mud::sleep_routine`**
  - Crear una rutina iterativa donde los tensores de KV se re-proyectan exclusivamente a través de los bloques `MudMambaLayer` con una regla de aprendizaje local (Delta-Rule) u optimización STE para forzar el ajuste del estado SSM sin backpropagation real.
* **Paso 3: Exponer API de Consolidación**
  - Proveer herramientas para que el orquestador externo o el usuario invoque el comando de Sleep cuando no haya urgencia en la latencia, similar a una defragmentación en segundo plano.

## 4. Impacto Esperado
Implementar *Sleep Folding* convertirá a Forge LLM en uno de los primeros motores de inferencia capaces de operar en un hardware restringido de manera verdaderamente **perpetua**, eliminando el colapso por "Out of Memory" que sufren otros agentes en tareas de larga duración.
