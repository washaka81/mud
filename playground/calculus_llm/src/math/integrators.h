#ifndef CALCULUS_LLM_MATH_INTEGRATORS_H
#define CALCULUS_LLM_MATH_INTEGRATORS_H

#include "calculus.h"
#include "../config.h"

namespace math {

// dy/dt = f(y, t)
using ODEFunc = std::function<State(const State&, double)>;
using DiffusionFunc = std::function<State(const State&, double)>;

// Resultado de una integración con seguimiento de convergencia
struct IntegrationResult {
    State state;
    bool converged;
    int steps_taken;
};

// Realiza un paso de integración usando Runge-Kutta 4
State rk4_step(const ODEFunc& f, const State& y, double t, double dt);

// Realiza un paso de integración estocástica usando Euler-Maruyama
State euler_maruyama_step(const ODEFunc& f, const DiffusionFunc& g, const State& y, double t, double dt);


// Realiza un paso adaptativo usando Runge-Kutta-Fehlberg (RK45)
// Retorna IntegrationResult con el nuevo estado, convergencia, y pasos aceptados
IntegrationResult rk45_step(const ODEFunc& f, State& y, double& t, double& dt, double tolerance = 1e-6);

} // namespace math

#endif // CALCULUS_LLM_MATH_INTEGRATORS_H
