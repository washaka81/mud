cmake_minimum_required(VERSION 3.10)
project(CalculusLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Directorios de inclusión
find_package(Eigen3 REQUIRED)
include_directories(src ${EIGEN3_INCLUDE_DIRS} /usr/include/eigen3)

# Copiar recursos al directorio de build
configure_file(${CMAKE_SOURCE_DIR}/vocabulario_es.txt ${CMAKE_BINARY_DIR}/vocabulario_es.txt COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/build/bpe_vocab.txt ${CMAKE_BINARY_DIR}/bpe_vocab.txt COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/build/bpe_merges.txt ${CMAKE_BINARY_DIR}/bpe_merges.txt COPYONLY)

# Fuentes del núcleo (lógica)
set(SOURCES
    src/math/calculus.cpp
    src/math/integrators.cpp
    src/math/positronic.cpp
    src/nlp/tokenizer.cpp
    src/nlp/bpe_tokenizer.cpp
    src/nlp/embedding.cpp
    src/nlp/embedding_trainer.cpp
    src/models/neural_ode.cpp
    src/models/continuous_llm.cpp
    src/models/potential_model.cpp
    src/models/trainer.cpp
)


add_executable(calculus_llm src/main.cpp ${SOURCES})

# Target de entrenamiento de embeddings
add_executable(embed_trainer src/embed_train_main.cpp ${SOURCES})

# Target de entrenamiento
add_executable(trainer src/train_main.cpp ${SOURCES})

# Target de entrenamiento de ODE
add_executable(ode_trainer src/ode_train_main.cpp ${SOURCES})

# Target de benchmark
add_executable(benchmark tests/benchmark.cpp ${SOURCES})

# Target de verificación del motor
add_executable(verify_engine tests/verify_engine.cpp ${SOURCES})

# New Test targets
add_executable(test_tokenizer tests/test_tokenizer.cpp ${SOURCES})
add_executable(test_models tests/test_models.cpp ${SOURCES})
add_executable(audit tests/audit.cpp ${SOURCES})
