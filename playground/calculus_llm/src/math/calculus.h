#ifndef CALCULUS_LLM_MATH_CALCULUS_H
#define CALCULUS_LLM_MATH_CALCULUS_H

#include <Eigen/Dense>
#include <vector>
#include <functional>
#include "../config.h"

namespace math {

// Optimizamos usando tamaños fijos conocidos en tiempo de compilación para evitar alocación en el heap
using State = Eigen::Matrix<double, config::EMBEDDING_DIM, 1>;
using NeuralMatrix = Eigen::Matrix<double, config::EMBEDDING_DIM, config::EMBEDDING_DIM>;
using Matrix = Eigen::MatrixXd; // Dinámica para tablas de embeddings
using MambaState = Eigen::Matrix<double, config::MAMBA_DIM, 1>;
using MambaMatrix = Eigen::Matrix<double, config::MAMBA_DIM, config::MAMBA_DIM>;

// Matrices de proyección Mamba
using Matrix_BC = Eigen::Matrix<double, config::MAMBA_DIM, config::EMBEDDING_DIM>;
using Matrix_CB = Eigen::Matrix<double, config::EMBEDDING_DIM, config::MAMBA_DIM>;
using Matrix_Delta = Eigen::Matrix<double, 1, config::EMBEDDING_DIM>;

// Operaciones vectoriales básicas (Nuevas versiones in-place para eficiencia)
void add_in(State& a, const State& b);
void sub_in(State& a, const State& b);
void mul_in(State& a, double scalar);

// Operaciones matriciales para la topología neuronal
State mat_mul(const NeuralMatrix& m, const State& v);
MambaState mat_mul(const MambaMatrix& m, const MambaState& v);
MambaState mat_mul(const Matrix_BC& m, const State& v);
State mat_mul(const Matrix_CB& m, const MambaState& v);
void tanh_in(State& a);
double softplus(double x);

// Máscaras para Variedades Fractales (Jerarquía Semántica)
State mask(const State& v, size_t start, size_t end);
void mask_in(State& v, size_t start, size_t end);

// Operaciones funcionales (mantienen compatibilidad)
State add(const State& a, const State& b);
State sub(const State& a, const State& b);
State mul(const State& a, double scalar);
double dot(const State& a, const State& b);
double norm(const State& a);

// Cálculo de gradiente numérico para una función escalar f(State)
State gradient(const std::function<double(const State&)>& f, const State& x, double delta = 1e-6);

} // namespace math

#endif // CALCULUS_LLM_MATH_CALCULUS_H
