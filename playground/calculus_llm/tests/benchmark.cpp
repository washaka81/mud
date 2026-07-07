#include "../src/models/continuous_llm.h"
#include "../src/config.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>

struct BenchmarkResult {
    std::string name;
    double lambda, sigma, delta_t, alpha;
    std::string prompt;
    std::vector<double> times_ms;
    int iterations;

    void print() const {
        if (times_ms.empty()) return;
        double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
        double mean = sum / times_ms.size();
        
        // Two-pass numerically stable stddev calculation
        double variance = 0;
        for (double t : times_ms) variance += (t - mean) * (t - mean);
        double stddev = std::sqrt(variance / times_ms.size());

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  " << name << "\n";
        std::cout << "    Lambda=" << lambda << " Sigma=" << sigma
                  << " dt=" << delta_t << " alpha=" << alpha << "\n";
        std::cout << "    Prompt: \"" << prompt << "\"\n";
        std::cout << "    Tiempo: " << mean << " ms (σ=" << stddev
                  << ") en " << iterations << " corridas\n";
    }
};

static double run_generation(double lambda, double sigma, double delta_t,
                              double alpha, const std::string& prompt) {
    models::ContinuousLLM llm;
    llm.set_params(lambda, sigma, delta_t);
    llm.set_alpha(alpha);
    auto start = std::chrono::high_resolution_clock::now();
    std::string response = llm.generate(prompt, 4, 10);
    (void)response;
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "========================================\n";
    std::cout << "   Benchmark de Convergencia (SLIME)    \n";
    std::cout << "========================================\n\n";

    // Warmup: 2 generaciones dummy para estabilizar cachés
    std::cout << "[Warmup] Cacheando...\n";
    for (int i = 0; i < 2; ++i) {
        models::ContinuousLLM llm;
        llm.generate("hola", 1, 2);
    }
    std::cout << "[Warmup] Listo.\n\n";

    const int ITER = 3;

    std::vector<BenchmarkResult> results;

    // Test 1: Convergencia rápida (Lambda alto)
    {
        BenchmarkResult r{"Test 1: Convergencia rápida", 1.0, 0.0, 0.1, config::DEFAULT_ALPHA, "hola"};
        for (int i = 0; i < ITER; ++i)
            r.times_ms.push_back(run_generation(r.lambda, r.sigma, r.delta_t, r.alpha, r.prompt));
        r.iterations = ITER;
        r.print();
        results.push_back(r);
    }

    // Test 2: Convergencia con ruido (Sigma alto)
    {
        BenchmarkResult r{"Test 2: Ruido en dinámica", 0.5, 0.2, 0.1, config::DEFAULT_ALPHA, "calculo"};
        for (int i = 0; i < ITER; ++i)
            r.times_ms.push_back(run_generation(r.lambda, r.sigma, r.delta_t, r.alpha, r.prompt));
        r.iterations = ITER;
        r.print();
        results.push_back(r);
    }

    // Test 3: Contexto fuerte
    {
        BenchmarkResult r{"Test 3: Contexto fuerte (alpha=0.3)", 1.2, 0.0, config::DEFAULT_DT, 0.3, "el amor es una sutileza"};
        for (int i = 0; i < ITER; ++i)
            r.times_ms.push_back(run_generation(r.lambda, r.sigma, r.delta_t, r.alpha, r.prompt));
        r.iterations = ITER;
        r.print();
        results.push_back(r);
    }

    // Test 4: Prompt largo (carga semántica)
    {
        BenchmarkResult r{"Test 4: Prompt largo", config::DEFAULT_LAMBDA, config::DEFAULT_SIGMA, config::DEFAULT_DT, config::DEFAULT_ALPHA,
                          "¿es la verdad el conocimiento absoluto del universo?"};
        for (int i = 0; i < ITER; ++i)
            r.times_ms.push_back(run_generation(r.lambda, r.sigma, r.delta_t, r.alpha, r.prompt));
        r.iterations = ITER;
        r.print();
        results.push_back(r);
    }

    // Resumen final
    std::cout << "\n--- Resumen ---\n";
    double total = 0;
    for (const auto& r : results) {
        double mean = std::accumulate(r.times_ms.begin(), r.times_ms.end(), 0.0) / r.times_ms.size();
        total += mean;
        std::cout << "  " << r.name << ": " << std::fixed << std::setprecision(1) << mean << " ms\n";
    }
    std::cout << "  Total: " << std::fixed << std::setprecision(1) << total << " ms\n";

    return 0;
}
