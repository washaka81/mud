#!/bin/bash

# =======================================================
# SLIME Engine (Positronic Brain) Command Line Interface
# =======================================================

COMMAND=$1

case $COMMAND in
    build)
        echo ">>> Compilando el Motor Tensorial SLIME (Eigen3)..."
        pushd calculus_llm > /dev/null
        mkdir -p build && pushd build > /dev/null
        cmake .. && make -j$(nproc)
        popd > /dev/null && popd > /dev/null
        ;;
    train)
        echo ">>> Iniciando Entrenamiento (tokenizer + pesos ODE)..."
        pushd calculus_llm > /dev/null
        if [ ! -f "build/trainer" ]; then
            echo "Error: Primero debes compilar ejecutando: ./slime.sh build"
            popd > /dev/null
            exit 1
        fi
        pushd build > /dev/null
        ./trainer
        popd > /dev/null && popd > /dev/null
        ;;
    chat)
        echo ">>> Iniciando el Cerebro Positrónico (Interfaz Interactiva)..."
        pushd calculus_llm > /dev/null
        if [ ! -f "../chat_interface.py" ]; then
            echo "Error: No se encuentra chat_interface.py en la raiz"
            popd > /dev/null
            exit 1
        fi
        pushd build > /dev/null
        python3 ../../chat_interface.py
        popd > /dev/null && popd > /dev/null
        ;;
    test)
        echo ">>> Validando Matemáticas y Leyes de Asimov..."
        pushd calculus_llm > /dev/null
        if [ ! -f "build/verify_engine" ]; then
            echo "Error: Primero debes compilar ejecutando: ./slime.sh build"
            popd > /dev/null
            exit 1
        fi
        pushd build > /dev/null && ./verify_engine && popd > /dev/null
        popd > /dev/null
        ;;
    test-all)
        echo ">>> Ejecutando todas las suites de tests..."
        pushd calculus_llm/build > /dev/null
        for suite in test_tokenizer test_models verify_engine benchmark; do
            if [ ! -f "./$suite" ]; then
                echo "Error: Falta $suite. Ejecutá ./slime.sh build primero."
                popd > /dev/null
                exit 1
            fi
        done
        echo "--- test_tokenizer ---" && ./test_tokenizer && \
        echo "--- test_models ---" && ./test_models && \
        echo "--- verify_engine ---" && ./verify_engine && \
        echo "--- benchmark ---" && ./benchmark
        popd > /dev/null
        ;;
    benchmark)
        echo ">>> Evaluando el Rendimiento Base..."
        pushd calculus_llm > /dev/null
        if [ ! -f "build/benchmark" ]; then
            echo "Error: Primero debes compilar ejecutando: ./slime.sh build"
            popd > /dev/null
            exit 1
        fi
        pushd build > /dev/null && ./benchmark && popd > /dev/null
        popd > /dev/null
        ;;
    all)
        $0 build && $0 train && $0 test
        ;;
    *)
        echo "Uso: ./slime.sh [comando]"
        echo ""
        echo "Comandos disponibles:"
        echo "  build     - Compila todo el motor desde cero usando CMake y Eigen3"
        echo "  train     - Entrena vectores del tokenizer + pesos de la Neural ODE"
        echo "  chat      - Inicia la conversación con el modelo (Generación Continua)"
        echo "  test      - Ejecuta validaciones de atractores, RK45 y repulsión Asimov"
        echo "  test-all  - Ejecuta todas las suites: tokenizer, models, engine, benchmark"
        echo "  benchmark - Mide los tiempos de procesamiento tensorial"
        echo "  all       - Ejecuta secuencialmente: build -> train -> test"
        echo ""
        ;;
esac
