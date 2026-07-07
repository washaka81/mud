#include "models/trainer.h"
#include "nlp/embedding.h"
#include "config.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

static std::string to_lower(const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (char c : s) r.push_back(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

int main(int argc, char** argv) {
    std::setbuf(stdout, NULL);
    std::setbuf(stderr, NULL);

    std::string corpus_path = argc > 1 ? argv[1] : "corpus_es.txt";
    int epochs = argc > 2 ? std::stoi(argv[2]) : 20;
    double lr = argc > 3 ? std::stod(argv[3]) : 0.01;

    std::cout << "=== ODE Trainer ===\n"
              << "Corpus: " << corpus_path << "\n"
              << "Epochs: " << epochs << "\n"
              << "LR: " << lr << "\n";

    // Load tokenizer (which also loads BPE + trained embeddings)
    nlp::Tokenizer tokenizer(config::EMBEDDING_DIM);

    // Create ODE model and trainer
    models::NeuralODE ode(config::EMBEDDING_DIM);
    models::ContinuousLLM llm_dummy;
    models::Trainer trainer(llm_dummy, tokenizer, ode);

    // Try loading existing ODE weights
    if (ode.load_weights(config::MODEL_WEIGHTS_FILENAME)) {
        std::cout << "[ODE] Loaded existing weights from " << config::MODEL_WEIGHTS_FILENAME << "\n";
    }

    // Load corpus and create training pairs
    std::ifstream file(corpus_path);
    if (!file.is_open()) {
        std::cerr << "Cannot open " << corpus_path << "\n";
        return 1;
    }

    std::vector<models::TrainingPair> pairs;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::string word;
        std::vector<std::string> words;
        for (char c : line) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!word.empty()) { words.push_back(to_lower(word)); word.clear(); }
            } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '\'' || c == '-') {
                word.push_back(c);
            } else {
                if (!word.empty()) { words.push_back(to_lower(word)); word.clear(); }
            }
        }
        if (!word.empty()) words.push_back(to_lower(word));

        for (size_t i = 0; i + 1 < words.size(); ++i) {
            pairs.push_back({words[i], words[i + 1]});
        }
    }
    file.close();

    std::cout << "[ODE] Created " << pairs.size() << " training pairs\n";

    // Train
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double decay = 1.0 - static_cast<double>(epoch) / epochs;
        trainer.train_ode_epoch(pairs, lr * decay);
    }

    // Save ODE weights
    ode.save_weights(config::MODEL_WEIGHTS_FILENAME);
    std::cout << "[ODE] Saved weights to " << config::MODEL_WEIGHTS_FILENAME << "\n";

    return 0;
}
