#include "models/continuous_llm.h"
#include <iostream>
#include <string>

int main() {
    models::ContinuousLLM llm;
    
    std::cout << "========================================================\n";
    std::cout << "   SLIME: Selective Latent Integral Model Engine (C++) \n";
    std::cout << "========================================================\n";
    std::cout << "Motor dinámico híbrido basado en Neural ODEs (RK45) y Lógica Positrónica.\n";

    std::string input;
    while (true) {
        std::cout << "Prompt > ";
        if (!std::getline(std::cin, input) || input == "salir") break;
        
        if (input.empty()) continue;

        std::string response = llm.generate(input);
        
        std::cout << "\n[LLM] Respuesta final: " << response << "\n";
        std::cout << "----------------------------------------\n";
    }

    return 0;
}
