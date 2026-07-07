#include "../include/math_c.h"

State add(State a, State b) {
    State res;
    for (int i = 0; i < DIM; i++) res.data[i] = a.data[i] + b.data[i];
    return res;
}

State sub(State a, State b) {
    State res;
    for (int i = 0; i < DIM; i++) res.data[i] = a.data[i] - b.data[i];
    return res;
}

State mul(State a, double scalar) {
    State res;
    for (int i = 0; i < DIM; i++) res.data[i] = a.data[i] * scalar;
    return res;
}

double dot(State a, State b) {
    double res = 0;
    for (int i = 0; i < DIM; i++) res += a.data[i] * b.data[i];
    return res;
}

double norm(State a) {
    return sqrt(dot(a, a));
}

State gradient(ScalarFunc f, State x, double delta, void* user_data) {
    State grad;
    for (int i = 0; i < DIM; i++) {
        State x_plus = x;
        State x_minus = x;
        x_plus.data[i] += delta;
        x_minus.data[i] -= delta;
        grad.data[i] = (f(x_plus, user_data) - f(x_minus, user_data)) / (2.0 * delta);
    }
    return grad;
}

State rk4_step(ODEFunc f, State y, double t, double dt, void* user_data) {
    State k1 = f(y, t, user_data);
    State k2 = f(add(y, mul(k1, dt / 2.0)), t + dt / 2.0, user_data);
    State k3 = f(add(y, mul(k2, dt / 2.0)), t + dt / 2.0, user_data);
    State k4 = f(add(y, mul(k3, dt)), t + dt, user_data);

    State dy = mul(add(add(k1, mul(k2, 2.0)), add(mul(k3, 2.0), k4)), dt / 6.0);
    return add(y, dy);
}

State euler_maruyama_step(DriftFunc f, DiffusionFunc g, State y, double t, double dt, void* user_data) {
    State drift = f(y, t, user_data);
    State diffusion = g(y, t, user_data);
    
    State dy_drift = mul(drift, dt);
    State dy_diffusion = mul(diffusion, sqrt(dt));
    
    return add(y, add(dy_drift, dy_diffusion));
}

double state_norm_sq(State a) {
    return dot(a, a);
}

State asimov_force(State y, State danger, double beta) {
    State diff = sub(y, danger);
    double dist_sq = state_norm_sq(diff);
    if (dist_sq < 1e-12) dist_sq = 1e-12; // Avoid division by zero
    double scale = beta / (dist_sq * sqrt(dist_sq)); // beta / |d|^3
    return mul(diff, scale);
}
