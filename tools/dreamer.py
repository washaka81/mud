import sqlite3
import os
import json

def generate_synthetic_dataset(db_path, output_path):
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Retrieve high-rank facts
    cursor.execute("SELECT content, source FROM facts ORDER BY rank DESC")
    rows = cursor.fetchall()
    
    dataset = []
    
    print(f"Dreaming... Processing {len(rows)} facts into training pairs.")
    
    for content, source in rows:
        # Create a "Thinking" pattern for each fact
        # This teaches the model to explain the source and context
        prompt = f"Q: What is mentioned in {source}? A: "
        thought = f"<thinking> Analyzing segment from {source}. This information relates to {content[:30]}... </thinking>"
        answer = f"<answer> {content} </answer>"
        
        full_text = prompt + thought + answer
        dataset.append(full_text)
        
        # Also create a reverse logic pair
        prompt_rev = f"Q: Explain this: {content[:40]}... A: "
        thought_rev = f"<thinking> Searching internal knowledge for concepts related to {content[:20]}... </thinking>"
        answer_rev = f"<answer> This is part of the documentation in {source}: {content} </answer>"
        
        dataset.append(prompt_rev + thought_rev + answer_rev)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in dataset:
            f.write(line + "\n")
            
    print(f"✅ Dataset generated: {output_path} ({len(dataset)} pairs)")
    conn.close()

if __name__ == "__main__":
    generate_synthetic_dataset("models/knowledge.db", "training/synthetic_knowledge.txt")
