#ifndef MATH_C_H
#define MATH_C_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIM 16

typedef struct {
    double data[DIM];
} State;

// Operaciones vectoriales
State add(State a, State b);
State sub(State a, State b);
State mul(State a, double scalar);
double dot(State a, State b);
double norm(State a);

// Cálculo
typedef double (*ScalarFunc)(State, void*);
State gradient(ScalarFunc f, State x, double delta, void* user_data);

typedef State (*ODEFunc)(State, double, void*);
State rk4_step(ODEFunc f, State y, double t, double dt, void* user_data);

// Ecuaciones Diferenciales Estocásticas (SDE)
typedef State (*DriftFunc)(State, double, void*);
typedef State (*DiffusionFunc)(State, double, void*);
State euler_maruyama_step(DriftFunc f, DiffusionFunc g, State y, double t, double dt, void* user_data);

// Utilidades adicionales
double state_norm_sq(State a);
State asimov_force(State y, State danger, double beta);

#endif
