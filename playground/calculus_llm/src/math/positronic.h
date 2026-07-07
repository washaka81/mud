#ifndef CALCULUS_LLM_MATH_POSITRONIC_H
#define CALCULUS_LLM_MATH_POSITRONIC_H

#include "calculus.h"
#include "../config.h"

namespace math {

// Operaciones Lógicas Positrónicas sobre el Espacio Continuo
class PositronicBrain {
public:
    // NOT: Invierte el sentido semántico (Inversión vectorial)
    static State gate_not(const State& a);

    // AND: Intersección semántica (Proyección o componente mínima)
    static State gate_and(const State& a, const State& b);

    // OR: Unión semántica (Suma con normalización de saturación)
    static State gate_or(const State& a, const State& b);

    // XOR: Diferencia semántica pura
    static State gate_xor(const State& a, const State& b);

    // Función de Decisión: Retorna 1.0 si la proposición es coherente
    static double evaluate_truth(const State& a, const State& b);

    // Fuerza Gramatical: Retorna un vector de atracción según la categoría gramatical esperada
    // tipo: 0 (Nulo), 1 (Linker), 2 (Noun), 3 (Verb), 4 (Adj)
    static State get_grammatical_force(int current_type, 
                                     const State& noun_centroid, 
                                     const State& verb_centroid, 
                                     const State& linker_centroid);

    // Lógica + Cálculo Diferencial: Leyes de Asimov como Campos de Fuerza
    // Genera un gradiente repulsivo (F = -∇V) para alejar la trayectoria de un concepto "peligroso".
    // Permite combinar peligros con Lógica OR (sumando campos) o AND (multiplicando potenciales).
    static State get_asimov_force(const State& y, const State& danger_centroid, double beta = 0.5);
};

} // namespace math

#endif // CALCULUS_LLM_MATH_POSITRONIC_H
