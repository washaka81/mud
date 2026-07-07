#include "nlp/bpe_tokenizer.h"
#include "nlp/embedding.h"
#include "nlp/embedding_trainer.h"
#include "config.h"
#include <iostream>

int main(int argc, char** argv) {
    std::setbuf(stdout, NULL);
    std::setbuf(stderr, NULL);

    std::string corpus_path = argc > 1 ? argv[1] : "corpus_es.txt";
    int epochs = argc > 2 ? std::stoi(argv[2]) : 20;
    size_t context = argc > 3 ? std::stoul(argv[3]) : 4;
    double lr = argc > 4 ? std::stod(argv[4]) : 0.01;

    std::cout << "=== Embedding Trainer ===\n";
    std::cout << "Corpus: " << corpus_path << "\n";
    std::cout << "Epochs: " << epochs << "\n";
    std::cout << "Context: " << context << "\n";
    std::cout << "LR: " << lr << std::endl;

    // Load BPE tokenizer
    std::cout << "Loading BPE tokenizer..." << std::endl;
    nlp::BpeTokenizer tokenizer;
    bool loaded = tokenizer.load(config::BPE_VOCAB_FILENAME, config::BPE_MERGES_FILENAME);
    if (!loaded) {
        loaded = tokenizer.load("../" + config::BPE_VOCAB_FILENAME,
                                 "../" + config::BPE_MERGES_FILENAME);
    }
    if (!loaded) {
        std::cerr << "Cannot load BPE files\n";
        return 1;
    }
    std::cout << "BPE loaded: " << tokenizer.vocab_size() << " tokens" << std::endl;

    // Create embedding matrix
    std::cout << "Creating embedding matrix..." << std::endl;
    nlp::EmbeddingMatrix embed(tokenizer.vocab_size(), config::EMBEDDING_DIM);
    std::cout << "Embedding matrix created" << std::endl;

    // Try loading existing embeddings
    embed.load(config::EMBEDDING_WEIGHTS_FILENAME);

    // Create trainer and train
    nlp::EmbeddingTrainer trainer(embed, tokenizer, context, lr);
    std::cout << "Starting training..." << std::endl;
    trainer.train_on_text(corpus_path, epochs);

    std::cout << "\n=== Training complete ===\n";
    std::cout << "Final loss: " << trainer.stats().total_loss / std::max(1, trainer.stats().steps) << "\n";
    std::cout << "Saved to: " << config::EMBEDDING_WEIGHTS_FILENAME << std::endl;

    return 0;
}
