#include "../include/positronic_c.h"

// Logical NOT: invert each component via 1 - sigmoid(a_i)
State gate_not_c(State a) {
    State res;
    for (int i = 0; i < DIM; i++) {
        double sig = 1.0 / (1.0 + exp(-a.data[i]));
        res.data[i] = 1.0 - sig;
    }
    return res;
}

// Logical AND: element-wise product of sigmoids
State gate_and_c(State a, State b) {
    State res;
    for (int i = 0; i < DIM; i++) {
        double sa = 1.0 / (1.0 + exp(-a.data[i]));
        double sb = 1.0 / (1.0 + exp(-b.data[i]));
        res.data[i] = sa * sb;
    }
    return res;
}

// Logical OR: De Morgan — OR(a,b) = NOT(AND(NOT(a), NOT(b)))
State gate_or_c(State a, State b) {
    return gate_not_c(gate_and_c(gate_not_c(a), gate_not_c(b)));
}

// Asimov repulsive force wrapper with safety checks
State asimov_force_c(State y, State danger, double beta) {
    if (beta < 0.0) beta = 0.0; // Safety: no attractive "repulsion"
    return asimov_force(y, danger, beta);
}

// Evaluate truth: cosine similarity in [0,1]
double evaluate_truth_c(State a, State b) {
    double na = norm(a);
    double nb = norm(b);
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    double cosine = dot(a, b) / (na * nb);
    // Clamp to [0, 1]
    if (cosine < 0.0) cosine = 0.0;
    if (cosine > 1.0) cosine = 1.0;
    return cosine;
}
