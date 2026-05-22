import os

# Read the new sentiment corpus
with open('training/sentiment_corpus.py', 'r') as f:
    sentiment_content = f.read()

# Extract the list content
start_idx = sentiment_content.find('[')
end_idx = sentiment_content.rfind(']') + 1
sentiment_list_str = sentiment_content[start_idx:end_idx]

# Read the original trainer
with open('training/kaggle_trainer.py', 'r') as f:
    trainer_content = f.read()

# Replace the CORPUS list in kaggle_trainer.py
corpus_start = trainer_content.find('CORPUS = [')
corpus_end = trainer_content.find(']', corpus_start) + 1

new_trainer_content = trainer_content[:corpus_start] + "CORPUS = " + sentiment_list_str + trainer_content[corpus_end:]

# Update constants and metadata in trainer
new_trainer_content = new_trainer_content.replace('STEPS = 30000', 'STEPS = 40000') # More steps for better convergence
new_trainer_content = new_trainer_content.replace('arch", "mud-ternary-moe-v1', 'arch", "mud-ternary-moe-v36-sentiment')

with open('training/kaggle_trainer.py', 'w') as f:
    f.write(new_trainer_content)

print("Kaggle trainer updated with sentiment corpus and extended steps.")
