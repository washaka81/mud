# SLIME Engine — Selective Latent Integral Model Engine

Motor experimental de lenguaje continuo basado en Neural ODEs + SSM (Mamba), implementado en C++17 con Eigen3.

## Estado Actual

| Componente | Estado |
|-----------|--------|
| BPE Tokenizer (32K vocab) | ✅ Funcional, optimizado |
| EmbeddingMatrix (32K×128D) | ✅ Entrenado (10 épocas, loss 2.09) |
| word_to_vec sincronizado | ✅ 139K vectores desde embeddings entrenados |
| ODE weight matrix W | ❌ Sin entrenar (produce sopa de palabras) |
| Entrenamiento ODE | ⏳ Bloqueado por velocidad (~10 min/época) |
| Integración llama.cpp | 📅 Próxima fase |

## Compilación

```bash
mkdir build && cd build
cmake ..
make -j4
```

## Ejecutables

- `calculus_llm` — Motor principal (chat interactivo)
- `embed_trainer` — Entrenamiento de embeddings (CBOW + negative sampling)
- `ode_trainer` — Entrenamiento de la matriz ODE
- `trainer` — Entrenador legacy (16D, dataset.csv)

## Entrenamiento

```bash
# Embeddings (ya entrenado, re-entrenar con corpus más grande):
./embed_trainer corpus_es.txt 10 4 0.01

# ODE (pendiente de optimización):
./ode_trainer corpus_es.txt 20 0.01
```

## Chat

```bash
./calculus_llm
```

Ver `ROADMAP.md` para detalles del plan de desarrollo.
