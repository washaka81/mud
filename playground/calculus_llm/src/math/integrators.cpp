#include "integrators.h"
#include <cmath>
#include <algorithm>

namespace math {

State rk4_step(const ODEFunc& f, const State& y, double t, double dt) {
    State k1 = f(y, t);
    
    State next_y = y;
    State h_2_k1 = k1;
    mul_in(h_2_k1, dt / 2.0);
    add_in(next_y, h_2_k1);
    State k2 = f(next_y, t + dt / 2.0);

    next_y = y;
    State h_2_k2 = k2;
    mul_in(h_2_k2, dt / 2.0);
    add_in(next_y, h_2_k2);
    State k3 = f(next_y, t + dt / 2.0);

    next_y = y;
    State h_k3 = k3;
    mul_in(h_k3, dt);
    add_in(next_y, h_k3);
    State k4 = f(next_y, t + dt);

    // dy = (k1 + 2*k2 + 2*k3 + k4) * dt / 6.0
    State dy = k1;
    
    State k2_2 = k2;
    mul_in(k2_2, 2.0);
    add_in(dy, k2_2);
    
    State k3_2 = k3;
    mul_in(k3_2, 2.0);
    add_in(dy, k3_2);
    
    add_in(dy, k4);
    mul_in(dy, dt / 6.0);
    
    return add(y, dy);
}

State euler_maruyama_step(const ODEFunc& f, const DiffusionFunc& g, const State& y, double t, double dt) {
    State drift = f(y, t);
    State diffusion = g(y, t);
    
    State dy_drift = drift;
    mul_in(dy_drift, dt);
    
    State dy_diffusion = diffusion;
    mul_in(dy_diffusion, std::sqrt(dt));
    
    State next_y = y;
    add_in(next_y, dy_drift);
    add_in(next_y, dy_diffusion);
    return next_y;
}

IntegrationResult rk45_step(const ODEFunc& f, State& y, double& t, double& dt, double tolerance) {
    // Coeficientes de Runge-Kutta-Fehlberg
    State k1 = f(y, t);
    
    auto step = [&](double weight, const State& k) {
        State next = y;
        add_in(next, mul(k, dt * weight));
        return next;
    };

    State k2 = f(step(1.0/4.0, k1), t + dt/4.0);
    
    State y3 = y;
    add_in(y3, mul(k1, dt * 3.0/32.0));
    add_in(y3, mul(k2, dt * 9.0/32.0));
    State k3 = f(y3, t + dt * 3.0/8.0);

    State y4 = y;
    add_in(y4, mul(k1, dt * 1932.0/2197.0));
    add_in(y4, mul(k2, dt * -7200.0/2197.0));
    add_in(y4, mul(k3, dt * 7296.0/2197.0));
    State k4 = f(y4, t + dt * 12.0/13.0);

    State y5 = y;
    add_in(y5, mul(k1, dt * 439.0/216.0));
    add_in(y5, mul(k2, dt * -8.0));
    add_in(y5, mul(k3, dt * 3680.0/513.0));
    add_in(y5, mul(k4, dt * -845.0/4104.0));
    State k5 = f(y5, t + dt);

    State y6 = y;
    add_in(y6, mul(k1, dt * -8.0/27.0));
    add_in(y6, mul(k2, dt * 2.0));
    add_in(y6, mul(k3, dt * -3544.0/2565.0));
    add_in(y6, mul(k4, dt * 1859.0/4104.0));
    add_in(y6, mul(k5, dt * -11.0/40.0));
    State k6 = f(y6, t + dt * 1.0/2.0);

    // Orden 4 (y_next)
    State y_next = y;
    add_in(y_next, mul(k1, dt * 25.0/216.0));
    add_in(y_next, mul(k3, dt * 1408.0/2565.0));
    add_in(y_next, mul(k4, dt * 2197.0/4104.0));
    add_in(y_next, mul(k5, dt * -1.0/5.0));

    // Orden 5 (z_next)
    State z_next = y;
    add_in(z_next, mul(k1, dt * 16.0/135.0));
    add_in(z_next, mul(k3, dt * 6656.0/12825.0));
    add_in(z_next, mul(k4, dt * 28561.0/56430.0));
    add_in(z_next, mul(k5, dt * -9.0/50.0));
    add_in(z_next, mul(k6, dt * 2.0/55.0));

    // Estimación del error
    double error = norm(sub(z_next, y_next));
    
    bool accepted = (error <= tolerance);
    int steps_accepted = 0;
    if (accepted) {
        t += dt;
        y = z_next;
        steps_accepted = 1;
    }

    // Ajuste de dt
    double s = 0.84 * std::pow(tolerance / (error + 1e-15), 0.25);
    dt *= std::max(0.1, std::min(4.0, s));
    if (dt < 1e-14) dt = 1e-14;

    return IntegrationResult{y, accepted, steps_accepted};
}

} // namespace math
