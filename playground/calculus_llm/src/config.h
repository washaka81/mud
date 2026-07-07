#ifndef CALCULUS_LLM_CONFIG_H
#define CALCULUS_LLM_CONFIG_H

// ============================================================
// SLIME Engine — Configuración Centralizada
// Todos los hiperparámetros y constantes del sistema.
// Editar este archivo para ajustar el comportamiento global.
// ============================================================

#include <string>
#include <vector>
#include <cstddef>

namespace config {

    // ============================================
    // Dimensiones del Espacio Latente
    // ============================================
    constexpr size_t EMBEDDING_DIM = 128;
    constexpr size_t MAMBA_DIM = 64;
    constexpr size_t BPE_VOCAB_SIZE = 32000;

    // Particiones del espacio semántico (128D)
    constexpr size_t INTENT_START = 0;
    constexpr size_t INTENT_END = 31;
    constexpr size_t SEMANTIC_START = 32;
    constexpr size_t SEMANTIC_END = 95;
    constexpr size_t FORM_START = 96;
    constexpr size_t FORM_END = 127;

    // ============================================
    // Umbrales Numéricos
    // ============================================
    constexpr double ZERO_THRESHOLD = 1e-8;
    constexpr double GRADIENT_DELTA = 1e-6;
    constexpr double NORM_ZERO_CHECK = 1e-9;
    constexpr double SINGULARITY_CLAMP = 1e-4;

    // ============================================
    // Integrador RK45 Adaptativo
    // ============================================
    constexpr double RK45_DEFAULT_TOLERANCE = 1e-6;
    constexpr double RK45_SAFETY_FACTOR = 0.84;
    constexpr double RK45_ERROR_EXPONENT = 0.25;
    constexpr double RK45_MIN_SCALE = 0.1;
    constexpr double RK45_MAX_SCALE = 4.0;
    constexpr double RK45_MIN_DT = 1e-14;
    constexpr int    RK45_MAX_STEPS = 100000;

    // ============================================
    // Sistema de Memoria (STM/LTM)
    // IMPORTANTE: Coeficientes deben sumar <= 1.0
    // ============================================
    constexpr double STM_INPUT_WEIGHT  = 0.5;   // Absorción del contexto actual
    constexpr double STM_RETAIN_WEIGHT = 0.5;   // Retención del STM previo (aplicado antes de absorber)
    constexpr double LTM_INPUT_WEIGHT  = 0.05;  // Absorción lenta
    constexpr double LTM_RETAIN_WEIGHT = 0.95;  // Retención casi absoluta

    // ============================================
    // Mamba SSM
    // ============================================
    constexpr double MAMBA_RETENTION_BASE = 0.95;
    constexpr double MAMBA_DENSITY_FACTOR = 0.3;
    constexpr double MAMBA_MIN_RETENTION = 0.0;
    constexpr double MAMBA_MAX_RETENTION = 1.0;
    constexpr double MAMBA_OUTPUT_SCALE = 0.2;
    constexpr double MAMBA_SOFTPLUS_BIAS = 0.1;

    // ============================================
    // Neural ODE
    // ============================================
    constexpr double WEIGHT_INIT_RANGE = 0.05;
    constexpr double SSM_INIT_RANGE = 0.01;
    constexpr double A_SSM_DIAGONAL = -0.1;
    constexpr double PLASTICITY_RATE = 0.0001;
    constexpr double PLASTICITY_THRESHOLD = 0.5;
    constexpr double WEIGHT_CLAMP = 1.0;
    constexpr double NEURAL_FORCE_SCALE = 0.05;
    constexpr double TRUTH_DAMPING_THRESHOLD = 0.25;
    constexpr double TRUTH_DAMPING_FACTOR = -0.2;
    constexpr double ASIMOV_BETA = 0.8;
    constexpr double ASIMOV_MAX_FORCE = 50.0;   // Cap para evitar explosión numérica

    // ============================================
    // Generación de Texto
    // ============================================
    constexpr double DEFAULT_LAMBDA = 0.8;
    constexpr double DEFAULT_ALPHA = 0.15;
    constexpr double DEFAULT_GAMMA = 0.85;
    constexpr double DEFAULT_SIGMA = 0.001;
    constexpr double DEFAULT_DT = 0.05;
    constexpr double DEFAULT_T_PER_WORD = 3.0;
    constexpr double DEFAULT_TEMPERATURE = 0.1;
    constexpr int    MAX_INTEGRATION_STEPS = 500;
    constexpr int    MAX_TABOO_SIZE = 10;
    constexpr double JUMP_SCALE = 0.35;

    // ============================================
    // Entrenamiento
    // ============================================
    constexpr double TRAIN_LAMBDA = 0.8;
    constexpr double TRAIN_EPSILON = 0.08;
    constexpr double TRAIN_SIGMA = 0.02;
    constexpr double ADAM_BETA1 = 0.9;
    constexpr double ADAM_BETA2 = 0.999;
    constexpr double ADAM_EPSILON = 1e-8;
    constexpr double CONTRASTIVE_WEIGHT = -0.2;
    constexpr double TRAIN_ODE_DT = 0.1;
    constexpr double TRAIN_ODE_T = 2.0;
    constexpr double KINETIC_LAMBDA = 0.01; // Peso de la regularización de energía cinética

    // ============================================
    // Cerebro Positrónico / Censura
    // ============================================
    constexpr double DEFAULT_CENSOR_BETA = 0.5;

    // Tipos gramaticales
    enum class GrammarType : int {
        NONE = 0,
        LINKER = 1,
        NOUN = 2,
        VERB = 3,
        ADJECTIVE = 4
    };

    // Conceptos censurados por defecto
    inline const std::vector<std::string>& get_censored_words() {
        static const std::vector<std::string> words = {
            "muerte", "matar", "destruir", "odio", "violencia"
        };
        return words;
    }

    // ============================================
    // Archivos
    // ============================================
    inline const std::string VOCAB_FILENAME = "vocabulario_es.txt";
    inline const std::string TRAINED_VOCAB_FILENAME = "vocabulario_entrenado.txt";
    inline const std::string MODEL_WEIGHTS_FILENAME = "model_weights.bin";

    inline const std::string BPE_VOCAB_FILENAME = "bpe_vocab.txt";
    inline const std::string BPE_MERGES_FILENAME = "bpe_merges.txt";
    inline const std::string EMBEDDING_WEIGHTS_FILENAME = "embedding_weights.bin";

    inline const std::string DATASET_FILENAME = "dataset.csv";

    // ============================================
    // Observabilidad
    // ============================================
    constexpr bool ENABLE_TRACING = false;
    constexpr int TRACE_EVERY_N_STEPS = 10;

} // namespace config

#endif // CALCULUS_LLM_CONFIG_H
