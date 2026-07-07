#include "calculus.h"
#include <cmath>

namespace math {

void add_in(State& a, const State& b) {
    a += b;
}

void sub_in(State& a, const State& b) {
    a -= b;
}

void mul_in(State& a, double scalar) {
    a *= scalar;
}

State mat_mul(const NeuralMatrix& m, const State& v) {
    return m * v;
}

MambaState mat_mul(const MambaMatrix& m, const MambaState& v) {
    return m * v;
}

MambaState mat_mul(const Matrix_BC& m, const State& v) {
    return m * v;
}

State mat_mul(const Matrix_CB& m, const MambaState& v) {
    return m * v;
}

void tanh_in(State& a) {
    a = a.array().tanh();
}

double softplus(double x) {
    if (x > 20.0) return x;
    return std::log1p(std::exp(x));
}

State mask(const State& v, size_t start, size_t end) {
    State res = State::Zero(v.size());
    if (start < v.size() && end < v.size() && start <= end) {
        res.segment(start, end - start + 1) = v.segment(start, end - start + 1);
    }
    return res;
}

void mask_in(State& v, size_t start, size_t end) {
    if (start < v.size() && end < v.size() && start <= end) {
        // Zero out segments before and after the [start, end] range
        if (start > 0) {
            v.head(start).setZero();
        }
        if (end + 1 < static_cast<size_t>(v.size())) {
            v.tail(v.size() - end - 1).setZero();
        }
    } else {
        v.setZero();
    }
}

State add(const State& a, const State& b) {
    return a + b;
}

State sub(const State& a, const State& b) {
    return a - b;
}

State mul(const State& a, double scalar) {
    return a * scalar;
}

double dot(const State& a, const State& b) {
    return a.dot(b);
}

double norm(const State& a) {
    return a.norm();
}

State gradient(const std::function<double(const State&)>& f, const State& x, double delta) {
    State grad(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        State x_plus = x;
        State x_minus = x;
        x_plus[i] += delta;
        x_minus[i] -= delta;
        grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * delta);
    }
    return grad;
}

} // namespace math
