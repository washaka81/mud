#include "models/trainer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

int main() {
    nlp::Tokenizer tokenizer(16);
    models::ContinuousLLM llm;
    models::NeuralODE& ode_ref = llm.get_ode();
    models::Trainer trainer(llm, tokenizer, ode_ref);

    std::vector<models::TrainingPair> dataset;
    std::ifstream file("../dataset.csv"); // Asumiendo ejecución desde build/
    if (!file.is_open()) file.open("dataset.csv"); // Fallback a la raíz
    
    if (file.is_open()) {
        std::string line, input, target;
        // Saltar cabecera
        std::getline(file, line);
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            if (std::getline(ss, input, ',') && std::getline(ss, target, ',')) {
                dataset.push_back({input, target});
            }
        }
        file.close();
        std::cout << ">>> Cargados " << dataset.size() << " pares desde dataset.csv\n";
    } else {
        std::cerr << ">>> Error: No se pudo abrir dataset.csv\n";
        return 1;
    }

    std::cout << ">>> Iniciando entrenamiento semántico ampliado...\n";
    
    double initial_lr = 0.3;
    double ode_lr = 0.01;
    // Entrenamos por 50 épocas con learning rate decay
    for (int i = 0; i < 50; ++i) {
        if (i % 10 == 0) std::cout << "Época " << i << ": ";
        
        double current_lr = initial_lr * (1.0 - (double)i / 50.0); // Decaimiento lineal
        trainer.train_epoch(dataset, current_lr);

        // Entrenar pesos de la Neural ODE cada 5 épocas
        if (i % 5 == 0) {
            double current_ode_lr = ode_lr * (1.0 - (double)i / 50.0);
            trainer.train_ode_epoch(dataset, current_ode_lr);
        }
    }

    std::cout << ">>> Entrenamiento completado.\n";
    
    // Guardamos los vectores entrenados
    tokenizer.save_to_file("vocabulario_entrenado.txt");
    std::cout << "[INFO] Vectores guardados en 'vocabulario_entrenado.txt'\n";

    // Guardamos los pesos de la ODE
    ode_ref.save_weights(config::MODEL_WEIGHTS_FILENAME);
    std::cout << "[INFO] Pesos de la ODE guardados en '" << config::MODEL_WEIGHTS_FILENAME << "'\n";

    return 0;
}
