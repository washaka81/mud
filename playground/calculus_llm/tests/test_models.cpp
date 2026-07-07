#include "../src/models/neural_ode.h"
#include "../src/models/continuous_llm.h"
#include "../src/models/trainer.h"
#include "../src/config.h"
#include <iostream>
#include <cassert>
#include <cmath>

static int test_failures = 0;

#define TEST_ASSERT(condition, msg) do { \
    if (!(condition)) { \
        std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        test_failures++; \
    } else { \
        std::cout << "  [OK] " << msg << std::endl; \
    } \
} while(0)

void test_neural_ode_evolve() {
    std::cout << ">>> Test: NeuralODE Evolve\n";
    models::NeuralODE ode(config::EMBEDDING_DIM);
    math::State start = math::State::Constant(config::EMBEDDING_DIM, 0.1);
    math::State end = ode.evolve(start, 0.0, 1.0, 0.1);
    
    bool all_finite = true;
    for(int i = 0; i < end.size(); i++) {
        if (!std::isfinite(end[i])) all_finite = false;
    }
    TEST_ASSERT(all_finite, "Evolved state should contain finite values.");
}

void test_neural_ode_save_load() {
    std::cout << ">>> Test: NeuralODE Save/Load\n";
    models::NeuralODE ode1(config::EMBEDDING_DIM);
    bool saved = ode1.save_weights("test_weights.bin");
    TEST_ASSERT(saved, "Should save weights successfully.");

    models::NeuralODE ode2(config::EMBEDDING_DIM);
    bool loaded = ode2.load_weights("test_weights.bin");
    TEST_ASSERT(loaded, "Should load weights successfully.");
}

void test_continuous_llm_generate() {
    std::cout << ">>> Test: ContinuousLLM Generate\n";
    models::ContinuousLLM llm;
    std::string response = llm.generate("hola mundo", 2, 5);
    TEST_ASSERT(!response.empty(), "Generated response should not be empty.");
}

void test_trainer_runs() {
    std::cout << ">>> Test: Trainer Runs\n";
    nlp::Tokenizer tok(config::EMBEDDING_DIM);
    models::ContinuousLLM llm;
    models::Trainer trainer(llm, tok, llm.get_ode());
    
    std::vector<models::TrainingPair> dataset = {
        {"hola", "mundo"}
    };
    
    // Test shouldn't crash
    trainer.train_epoch(dataset, 0.1);
    TEST_ASSERT(true, "Trainer epoch ran without crashing.");
}

void test_continuous_llm_pragmatics() {
    std::cout << ">>> Test: ContinuousLLM Pragmatics (Greeting & Register)\n";
    models::ContinuousLLM llm;
    
    // Test formal greeting
    std::string response_formal = llm.generate("Buenos días señor, ¿cómo está?", 2, 5);
    std::cout << "  Formal response: " << response_formal << "\n";
    
    // Lowercase response to make checks register-insensitive
    std::string formal_lower = response_formal;
    std::transform(formal_lower.begin(), formal_lower.end(), formal_lower.begin(), ::tolower);
    
    bool has_formal_preamble = (formal_lower.find("buenos") != std::string::npos ||
                                 formal_lower.find("estimado") != std::string::npos ||
                                 formal_lower.find("señor") != std::string::npos ||
                                 formal_lower.find("señora") != std::string::npos ||
                                 formal_lower.find("considere") != std::string::npos);
    TEST_ASSERT(has_formal_preamble, "Formal register preamble is generated when formal greeting is used.");
    
    // Test informal greeting
    std::string response_informal = llm.generate("¡Hola amigo! ¿Qué tal?", 2, 5);
    std::cout << "  Informal response: " << response_informal << "\n";
    
    std::string informal_lower = response_informal;
    std::transform(informal_lower.begin(), informal_lower.end(), informal_lower.begin(), ::tolower);
    
    bool has_informal_preamble = (informal_lower.find("hola") != std::string::npos ||
                                   informal_lower.find("qué tal") != std::string::npos ||
                                   informal_lower.find("cómo va") != std::string::npos ||
                                   informal_lower.find("oye") != std::string::npos);
    TEST_ASSERT(has_informal_preamble, "Informal register preamble is generated when horizontal greeting is used.");
    
    // Test key-value query
    std::string response_kv = llm.generate("dame las llaves y valores del sistema", 2, 5);
    std::cout << "  Key-Value response: " << response_kv << "\n";
    TEST_ASSERT(!response_kv.empty(), "Key-Value response should not be empty.");
}

int main() {
    std::cout << "========================================\n";
    std::cout << "   Test Suite: Models                   \n";
    std::cout << "========================================\n\n";

    test_neural_ode_evolve();
    test_neural_ode_save_load();
    test_continuous_llm_generate();
    test_continuous_llm_pragmatics();
    test_trainer_runs();

    if (test_failures > 0) {
        std::cerr << "\n[RESULT] " << test_failures << " tests failed!\n";
        return 1;
    }
    
    std::cout << "\n[RESULT] All tests passed.\n";
    return 0;
}
