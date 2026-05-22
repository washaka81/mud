import sqlite3
import os
import re
import random

def apply_typos(text):
    # Diccionario de tildes para simular errores comunes
    tildes = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n',
              'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ñ': 'N'}
    
    chars = list(text)
    # 1. Simular falta de tildes (50% de probabilidad en palabras con tilde)
    for i, c in enumerate(chars):
        if c in tildes and random.random() > 0.5:
            chars[i] = tildes[c]
    
    # 2. Simular intercambio de letras adyacentes (muy baja probabilidad)
    if len(chars) > 5 and random.random() > 0.95:
        idx = random.randint(1, len(chars) - 2)
        if chars[idx].isalpha() and chars[idx+1].isalpha():
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            
    return "".join(chars)

def extract_for_v42(db_path, output_path):
    if not os.path.exists(db_path):
        print(f"Error: {db_path} no encontrado.")
        return

    print(f"🔗 Conectando a {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Extraer contenido y fuente
    cursor.execute("SELECT content, source FROM facts ORDER BY rank DESC")
    rows = cursor.fetchall()
    
    print(f"📊 Procesando {len(rows)} hechos con Robustez Ortográfica (Augmentation)...")
    def clean_text(text):
        text = re.sub(r'([.,!?()¿¡])', r' \1 ', text)
        return " ".join(text.split())

    # Preámbulos simpáticos y un poco chistosos
    prefixes = [
        "¡Hola! He buscado en mi memoria y he encontrado esto: ",
        "¡Qué buena pregunta! Mira lo que dicen mis circuitos: ",
        "He consultado mis neuronas digitales y la respuesta es: ",
        "Dato curioso y real: ",
        "¡Vaya! Esa es interesante. Aquí tienes: ",
        "Según la verdad absoluta de mis datos: ",
        "¡A ver, a ver! Deja que revise... ¡Listo!: ",
        "Mis algoritmos están felices de contarte que: "
    ]

    suffixes = [
        " ¡Espero que te sea súper útil!",
        " ¿No es fascinante cómo funciona todo?",
        " ¡Listo! Problema resuelto por tu asistente favorito.",
        " (Mis circuitos hacen un pequeño baile de victoria por encontrar esto).",
        " ¡Cualquier otra cosa, aquí me tienes!",
        " ¡Ciencia y lógica al rescate!"
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for content, source in rows:
            cleaned = clean_text(content)
            pre = random.choice(prefixes)
            suf = random.choice(suffixes)

            # Formato 1: Conocimiento Directo (Siempre Perfecto)
            f.write(f"{cleaned}\n")

            # Formato 2: Q&A Correcto (Entrada perfecta -> Respuesta Simpática)
            f.write(f"Q: Sobre {source} , ¿qué se menciona? A: {pre}{cleaned}{suf} </s>\n")

            # Formato 3: Q&A con RUIDO en la PREGUNTA (Entrada imperfecta -> Respuesta Simpática)
            noisy_q = apply_typos(f"Q: Sobre {source} , que se menciona? A: ")
            f.write(f"{noisy_q}{pre}{cleaned}{suf} </s>\n")

            
            count += 3
            if count % 15000 == 0:
                print(f"  > {count} líneas generadas (incluyendo aumentación)...")

    conn.close()
    print(f"✅ Corpus MUD-V1.5-MASTER generado: {output_path} ({count} secuencias robustas)")

if __name__ == "__main__":
    db = "models/knowledge.db"
    out = "training/massive_knowledge_corpus.txt"
    extract_for_v42(db, out)

if __name__ == "__main__":
    db = "models/knowledge.db"
    out = "training/massive_knowledge_corpus.txt"
    extract_for_v42(db, out)
