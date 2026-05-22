import sqlite3
import os

def generate():
    if not os.path.exists('models/knowledge.db'):
        print("Error: models/knowledge.db not found.")
        return

    print("Extracting 59k facts from database...")
    conn = sqlite3.connect('models/knowledge.db')
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM facts")
    rows = cursor.fetchall()
    conn.close()

    output_path = 'training/massive_knowledge_corpus.txt'
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in rows:
            content = row[0]
            # Clean content if needed (e.g. remove "Source: ... | Content: ")
            if " | Content: " in content:
                content = content.split(" | Content: ")[1]
            
            f.write(content.strip() + "\n")
            count += 1

    print(f"Dataset generated at {output_path} with {count} facts.")

if __name__ == "__main__":
    generate()
