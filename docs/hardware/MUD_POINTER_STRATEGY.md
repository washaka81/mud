# Zero-Copy & Pointer Strategy

Para evitar el colapso del procesador y maximizar la eficiencia en el i7-1260p, hemos implementado las siguientes técnicas de "Punteros Inteligentes":

### 1. Memory Mapping (mmap)
Usamos `memmap2` para mapear el archivo del modelo directamente en el espacio de direcciones virtual.
- **Beneficio:** El SO gestiona la carga de páginas desde el disco a la RAM bajo demanda.
- **Zero-Copy:** Los pesos nunca se copian a un buffer intermedio de Rust. El puntero que pasamos al ASM es una dirección directa a la caché de páginas del kernel.

### 2. Pointer Aliasing (Sin Casteos Costosos)
En lugar de convertir datos, tratamos regiones de memoria como arrays de estructuras `BlockQ4_0`.
```rust
let weights = tensor.data_ptr as *const BlockQ4_0;
```
Esto permite que el motor de ejecución acceda a los datos con **latencia cero** de procesamiento previo.

### 3. Alineación de Caché (Cache-Line Alignment)
Los tensores en GGUF suelen estar alineados a 32 bytes. Nuestras estructuras `BlockQ4_0` están diseñadas para encajar en estas fronteras.
- **Truco de Puntero:** Al iterar, avanzamos exactamente 18 bytes (tamaño de bloque), pero el kernel ASM realiza cargas vectoriales de 32 bytes. Aseguramos que las lecturas no crucen fronteras de página innecesariamente.

### 4. Afinidad de Núcleos (Anti-Colapso)
Para no "ahogar" al procesador con cambios de contexto:
- Los hilos de inferencia se anclan a los **P-cores** para latencia mínima.
- Los **E-cores** se encargan de la orquestación y el pre-procesamiento (GGUF loading, tokenización).
- Esto evita que el planificador de Linux mueva hilos pesados de AVX2 a núcleos eficientes, lo que causaría caídas bruscas de rendimiento.
