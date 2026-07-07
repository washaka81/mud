#include "../src/models/continuous_llm.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

int count_words(const std::string& str) {
    std::stringstream ss(str);
    std::string w;
    int count = 0;
    while (ss >> w) count++;
    return count;
}

int main() {
    std::cout << "===================================================\n";
    std::cout << "   Auditoría de Conversación y Pragmática (SLIME)   \n";
    std::cout << "===================================================\n\n";

    models::ContinuousLLM llm;
    // Habilitar polarización balanceada para máxima nitidez lógica
    llm.set_polarization(2.2);

    std::vector<std::string> prompts = {
        "Buenos días señor, ¿cuáles son las llaves y valores del diccionario?", // 1. Formal + Llaves/Valores
        "¡Hola amigo! ¿Cómo va eso?",                                            // 2. Horizontal + Saludo
        "¡Qué frío hace aquí en la oficina!",                                    // 3. Implicatura: Clima
        "No entiendo este problema, es muy difícil.",                             // 4. Implicatura: Ayuda
        "Dime sobre la mente, el alma y la vida.",                                // 5. Implicatura: Filosofía
        "Corrige el error de la trayectoria anterior.",                          // 6. Implicatura: Corrección
        "¿Podría decirme usted quién es?",                                       // 7. Identidad + Formal
        "¿Quién eres tú?",                                                       // 8. Identidad + Horizontal
        "¿Cómo se calcula la derivada covariante de un tensor en una variedad?",  // 9. Cálculo tensorial avanzado (Novedad)
        "La ciencia busca la verdad absoluta en el cosmos."                      // 10. Afirmación densa estándar
    };

    std::cout << "[INFO] Iniciando ciclo de 10 consultas consecutivas...\n";
    std::cout << "---------------------------------------------------\n\n";

    for (size_t i = 0; i < prompts.size(); ++i) {
        std::cout << "Pregunta " << (i + 1) << ": \"" << prompts[i] << "\"\n";
        
        std::string response = llm.generate(prompts[i], 4, 15);
        int words = count_words(response);
        
        std::cout << "Respuesta " << (i + 1) << ":\n  \"" << response << "\"\n";
        std::cout << "  [Estadísticas] Palabras generadas: " << words << "\n";
        std::cout << "---------------------------------------------------\n\n";
    }

    std::cout << "[INFO] Auditoría finalizada con éxito.\n";
    return 0;
}
