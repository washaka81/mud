#!/bin/bash
# Extract knowledge from MUD model by running prompts and ingesting outputs

PROMPTS_FILE=/tmp/prompts.txt
OUTPUT_FILE=/tmp/kaggle_knowledge.txt

cat > "$PROMPTS_FILE" << 'EOF'
hola
hello
como estas
what is MUD
que sabes hacer
que es inteligencia artificial
como funciona un modelo de lenguaje
gracias
adios
buenos dias
buenas tardes
buenas noches
como te llamas
eres inteligente
que idiomas hablas
de donde eres
cual es tu proposito
que significa MoE
que es una red neuronal
hablas espanol
hablas ingles
EOF

# Run engine, feed each prompt, capture responses
> "$OUTPUT_FILE"
while IFS= read -r prompt; do
  echo "Q: $prompt" >> "$OUTPUT_FILE"
  echo "$prompt" | timeout 10 ./target/release/forge_llm 2>/dev/null | grep -v "^MUD" | grep -v "^❯" | grep -v "^━━" | grep -v "^Inf" | grep -v "^Inference" | grep -v "^Times" | grep -v "^$" | head -5 >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
done < "$PROMPTS_FILE"

echo "Knowledge extracted to $OUTPUT_FILE"
wc -l "$OUTPUT_FILE"
