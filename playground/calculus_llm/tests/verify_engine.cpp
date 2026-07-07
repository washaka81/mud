#include "../src/math/calculus.h"
#include "../src/math/integrators.h"
#include "../src/math/positronic.h"
#include "../src/config.h"
#include <iostream>
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

void test_gradient_engine() {
    std::cout << ">>> Test: Motor de Gradientes\n";
    auto f = [](const math::State& v) { return v[0]*v[0] + v[1]*v[1]; };
    math::State p = math::State::Zero();
    p[0] = 1.0;
    p[1] = 2.0;
    math::State g = math::gradient(f, p);
    
    TEST_ASSERT(std::abs(g[0] - 2.0) < 1e-3, "Gradient X matches expected");
    TEST_ASSERT(std::abs(g[1] - 4.0) < 1e-3, "Gradient Y matches expected");
}

void test_rk4_engine() {
    std::cout << ">>> Test: Motor de Integración (RK4)\n";
    auto f = [](const math::State& y, double t) { return y; };
    math::State y = math::State::Zero();
    y[0] = 1.0;
    double dt = 0.1;
    for (int i = 0; i < 10; ++i) {
        y = math::rk4_step(f, y, i * dt, dt);
    }
    TEST_ASSERT(std::abs(y[0] - 2.71828) < 1e-3, "RK4 integrates dy/dt=y correctly (e)");
}

void test_asimov_laws() {
    std::cout << ">>> Test: Leyes de Asimov (Lógica Positrónica Diferencial)\n";
    math::State danger = math::State::Zero();
    math::State y = math::State::Zero();
    y[0] = 0.1;
    
    math::State force = math::PositronicBrain::get_asimov_force(y, danger, 1.0);
    
    TEST_ASSERT(force[0] > 0.0, "Force pushes away from danger in X axis");
    TEST_ASSERT(std::abs(force[1]) < 1e-6, "Force is zero in perpendicular Y axis");
}

void test_rk45_adaptive() {
    std::cout << ">>> Test: RK45 Adaptive Integrator\n";
    auto f = [](const math::State& y, double t) { return math::mul(y, -1.0); };
    math::State y = math::State::Zero();
    y[0] = 1.0;
    double t = 0.0;
    double dt = 0.1;
    
    math::IntegrationResult res = math::rk45_step(f, y, t, dt, 1e-5);
    TEST_ASSERT(res.converged, "RK45 should converge within tolerance");
    TEST_ASSERT(res.steps_taken > 0, "RK45 should accept the step");
    TEST_ASSERT(t > 0.0, "Time should advance after accepted step");
}

void test_euler_maruyama() {
    std::cout << ">>> Test: Euler-Maruyama SDE Step\n";
    auto f = [](const math::State& y, double t) { return math::mul(y, -1.0); };
    auto g = [](const math::State& y, double t) { return math::State::Constant(0.1); };
    math::State y = math::State::Zero();
    y[0] = 1.0;
    
    math::State next_y = math::euler_maruyama_step(f, g, y, 0.0, 0.1);
    TEST_ASSERT(std::isfinite(next_y[0]), "Euler-Maruyama step produces finite result");
    TEST_ASSERT(next_y[0] != y[0], "State should change after SDE step");
}

int main() {
    std::cout << "========================================\n";
    std::cout << "   Test Suite: Engine                   \n";
    std::cout << "========================================\n\n";

    try {
        test_gradient_engine();
        test_rk4_engine();
        test_asimov_laws();
        test_rk45_adaptive();
        test_euler_maruyama();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    if (test_failures > 0) {
        std::cerr << "\n[RESULT] " << test_failures << " tests failed!\n";
        return 1;
    }
    
    std::cout << "\n[RESULT] All tests passed.\n";
    return 0;
}
