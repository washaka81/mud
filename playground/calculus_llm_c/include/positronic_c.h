#ifndef POSITRONIC_C_H
#define POSITRONIC_C_H

#include "math_c.h"

// Positronic brain gates — C port
State gate_not_c(State a);
State gate_and_c(State a, State b);
State gate_or_c(State a, State b);
State asimov_force_c(State y, State danger, double beta);
double evaluate_truth_c(State a, State b);

#endif
