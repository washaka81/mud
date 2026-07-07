#include "positronic.h"
#include "../config.h"
#include <cmath>
#include <algorithm>

namespace math {

State PositronicBrain::gate_not(const State& a) {
    return mul(a, -1.0);
}

State PositronicBrain::gate_and(const State& a, const State& b) {
    size_t n = std::min(static_cast<size_t>(a.size()), static_cast<size_t>(b.size()));
    State res = State::Zero(n);
    for (size_t i = 0; i < n; ++i) {
        // Soft-AND: similar a la lógica difusa (min(a, b))
        // Pero adaptado a vectores que pueden ser negativos
        res[i] = (std::abs(a[i]) < std::abs(b[i])) ? a[i] : b[i];
    }
    return res;
}

State PositronicBrain::gate_or(const State& a, const State& b) {
    size_t n = std::min(static_cast<size_t>(a.size()), static_cast<size_t>(b.size()));
    State res = State::Zero(n);
    for (size_t i = 0; i < n; ++i) {
        // Soft-OR: max(a, b)
        res[i] = (std::abs(a[i]) > std::abs(b[i])) ? a[i] : b[i];
    }
    return res;
}

State PositronicBrain::gate_xor(const State& a, const State& b) {
    return sub(a, b);
}

double PositronicBrain::evaluate_truth(const State& a, const State& b) {
    double d = dot(a, b);
    double n = norm(a) * norm(b);
    if (n < config::NORM_ZERO_CHECK) return 0.0;
    
    // Similitud de coseno escalada a [0, 1]
    double similarity = d / n;
    return (similarity + 1.0) / 2.0;
}

State PositronicBrain::get_grammatical_force(int current_type, 
                                           const State& noun_centroid, 
                                           const State& verb_centroid, 
                                           const State& linker_centroid) {
    // Definimos las transiciones probables:
    // Linker (el, la) -> Noun (hombre)
    // Noun -> Verb (quiere)
    // Verb -> Linker/Noun
    
    switch (current_type) {
        case 1: // Linker
            return noun_centroid;
        case 2: // Noun
            return verb_centroid;
        case 3: // Verb
            return linker_centroid;
        default:
            return State::Zero();
    }
}

State PositronicBrain::get_asimov_force(const State& y, const State& danger_centroid, double beta) {
    // Implementación de Lógica Diferencial: Barrera de Potencial
    // V(x) = beta / ||x - danger_centroid||^2
    // F(x) = -∇V = 2 * beta * (x - danger_centroid) / ||x - danger_centroid||^4
    State diff = y - danger_centroid;
    double dist_sq = diff.squaredNorm();
    
    // Evitar singularidad (división por cero) si y cae exactamente en el centroide
    if (dist_sq < config::SINGULARITY_CLAMP) dist_sq = config::SINGULARITY_CLAMP;
    
    double scalar = (2.0 * beta) / (dist_sq * dist_sq);
    // Cap scalar to avoid numerical explosion
    scalar = std::min(scalar, config::ASIMOV_MAX_FORCE);
    return scalar * diff; // Vector de repulsión ortogonal a la barrera
}

} // namespace math
