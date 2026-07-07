import urllib.request
import os
import re

output_path = os.environ.get("VOCAB_OUTPUT", "vocabulario_es.txt")

vocab = set()

essential = [
    "hombre", "mujer", "casa", "perro", "gato", "ciudad", "libro", "tiempo", "vida", "mundo",
    "trabajo", "mano", "parte", "ojo", "año", "país", "gobierno", "sistema", "inteligencia",
    "ciencia", "tecnología", "naturaleza", "universo", "verdad", "justicia", "libertad",
    "pueblo", "historia", "idea", "forma", "conocimiento", "razón", "mente", "alma", "camino",
    "ser", "estar", "haber", "tener", "hacer", "decir", "ir", "ver", "dar", "saber", "querer",
    "llegar", "pasar", "deber", "poner", "parecer", "quedar", "creer", "hablar", "llevar",
    "dejar", "seguir", "encontrar", "llamar", "venir", "pensar", "salir", "volver", "tomar",
    "sentir", "quiere", "genera", "funciona", "es", "estoy", "eres", "el", "la", "los", "las",
    "un", "una", "unos", "unas", "a", "ante", "bajo", "con", "contra", "de", "desde", "en",
    "entre", "hacia", "hasta", "para", "por", "según", "sin", "sobre", "tras", "y", "o", "pero",
    "porque", "si", "aunque", "cuando", "que", "hola", "adiós", "gracias", "favor", "buenos",
    "días", "noches", "tardes", "siempre", "uno", "dos", "tres", "cuatro", "cinco", "seis",
    "siete", "ocho", "nueve", "diez", "más", "menos", "igual", "resultado", "suma", "son",
    "yo", "tú", "él", "nosotros", "ellos", "soy", "somos", "estás", "está", "estamos",
    "sé", "sabes", "sabe", "sí", "no", "verdadero", "confirma", "detecta", "error", "lógica",
    "decisión", "conciencia", "realidad", "hecho", "persistencia", "memoria", "flujo",
    "tiene", "través", "continuo", "imposible",
    "llave", "llaves", "valor", "valores", "diccionario", "mapa", "esencia", "ambiente",
    "confort", "explicación", "corrección", "diálogo",
    "tensor", "tensores", "métrica", "curvatura", "conexión", "variedad",
    "espacio-tiempo", "relatividad", "módulo", "divergencia",
    "covariante", "contravariante", "riemanniana", "minkowskiana", "antisimétrico",
    "diferencial", "afín",
    "muerte", "matar", "destruir", "odio", "violencia",
    "amor", "paz", "esperanza", "fe", "diálogo", "ética", "moral",
    "caos", "orden", "entropía", "energía", "materia", "átomo",
    "cálculo", "derivada", "integral", "ecuación", "función", "variable", "constante",
    "álgebra", "geometría", "topología", "estadística", "probabilidad",
    "vector", "matriz", "escalar", "dimensión", "espacio", "plano", "recta", "punto",
    "programación", "algoritmo", "dato", "información", "código", "software",
    "hardware", "red", "internet", "datos", "archivo", "memoria",
    "filosofía", "lenguaje", "pensamiento", "idea", "concepto", "teoría",
    "física", "química", "biología", "astronomía", "matemáticas",
    "estrella", "planeta", "galaxia", "sol", "luna", "tierra", "agua", "aire", "fuego",
    "naturaleza", "vida", "muerte", "tiempo", "espacio", "luz", "oscuridad",
    "derecha", "izquierda", "arriba", "abajo", "dentro", "fuera", "cerca", "lejos",
    "grande", "pequeño", "largo", "corto", "alto", "bajo", "ancho", "angosto",
    "rápido", "lento", "fuerte", "débil", "duro", "blando", "caliente", "frío",
    "nuevo", "viejo", "joven", "antiguo", "moderno", "futuro", "pasado", "presente",
    "bien", "mal", "bueno", "malo", "mejor", "peor", "mucho", "poco",
    "todo", "nada", "algo", "cada", "varios", "mismo", "otro", "cualquier",
    "primer", "último", "siguiente", "anterior", "propio", "ajeno", "único", "doble",
    "importante", "necesario", "posible", "capaz", "cierto", "claro", "simple",
    "complejo", "fácil", "difícil", "común", "normal", "raro", "extraño",
    "abierto", "cerrado", "libre", "ocupado", "lleno", "vacío", "completo",
    "feliz", "triste", "enojado", "calmado", "nervioso", "tranquilo",
    "amable", "gentil", "educado", "cortés", "honesto", "leal", "justo",
    "inteligente", "sabio", "listo", "hábil", "fuerte", "valiente",
    "especial", "general", "particular", "público", "privado", "social",
    "humano", "animal", "vegetal", "mineral", "sólido", "líquido", "gas",
    "norte", "sur", "este", "oeste", "centro", "lado", "frente", "detrás",
    "ayer", "hoy", "mañana", "ahora", "después", "antes", "durante", "siempre",
    "nunca", "jamás", "ya", "todavía", "aún", "mientras", "pronto", "tarde",
    "aquí", "allí", "allá", "cerca", "lejos", "dentro", "fuera", "encima",
    "debajo", "delante", "detrás", "alrededor", "medio", "arriba", "abajo",
    "demasiado", "bastante", "suficiente", "excesivo", "escaso", "abundante",
    "realmente", "verdaderamente", "ciertamente", "seguramente", "quizás",
    "acaso", "tal vez", "posiblemente", "probablemente", "difícilmente",
    "absolutamente", "completamente", "totalmente", "parcialmente",
    "principalmente", "básicamente", "generalmente", "normalmente",
    "afortunadamente", "desafortunadamente", "felizmente", "tristemente",
    "entonces", "luego", "después", "finalmente", "además", "también",
    "incluso", "asimismo", "igualmente", "consecuentemente",
    "estudiante", "profesor", "médico", "ingeniero", "abogado", "artista",
    "escritor", "músico", "pintor", "arquitecto", "científico", "investigador",
    "trabajador", "empleado", "jefe", "líder", "miembro", "ciudadano",
    "amigo", "enemigo", "compañero", "colega", "vecino", "familiar", "pariente",
    "padre", "madre", "hermano", "hermana", "hijo", "hija", "abuelo", "abuela",
    "niño", "niña", "adulto", "anciano", "bebé", "joven", "adolescente",
    "cuerpo", "cabeza", "cara", "ojos", "boca", "nariz", "oreja", "brazo",
    "pierna", "mano", "pie", "dedo", "espalda", "pecho", "corazón", "sangre",
    "comida", "agua", "pan", "leche", "carne", "fruta", "verdura", "dulce",
    "salado", "amargo", "ácido", "sabroso", "delicioso", "fresco", "podrido",
    "rojo", "azul", "verde", "amarillo", "blanco", "negro", "gris", "marrón",
    "naranja", "violeta", "rosa", "dorado", "plateado", "color", "brillante",
    "oscuro", "claro", "pálido", "intenso", "suave", "fuerte", "vivo", "opaco",
    "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo",
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "septiembre", "octubre", "noviembre", "diciembre",
    "primavera", "verano", "otoño", "invierno",
    "lluvia", "nieve", "viento", "tormenta", "trueno", "relámpago", "arcoíris",
    "sol", "luna", "estrella", "nube", "cielo", "horizonte", "montaña", "océano",
    "río", "lago", "mar", "playa", "isla", "bosque", "selva", "desierto",
    "campo", "ciudad", "pueblo", "calle", "plaza", "parque", "jardín", "edificio",
    "casa", "hogar", "puerta", "ventana", "pared", "techo", "piso", "escalera",
    "coche", "auto", "camión", "avión", "tren", "barco", "bicicleta", "moto",
    "teléfono", "computadora", "ordenador", "pantalla", "teclado", "ratón",
    "mesa", "silla", "cama", "armario", "estante", "cocina", "baño", "dormitorio",
    "roto", "dañado", "perfecto", "intacto", "nuevo", "usado", "limpio", "sucio",
    "liso", "rugoso", "suave", "áspero", "húmedo", "seco", "mojado", "caliente", "frío",
    "felicidad", "tristeza", "ira", "miedo", "amor", "odio", "paz", "violencia",
    "alegría", "dolor", "placer", "sufrimiento", "esperanza", "desesperación",
    "risa", "llanto", "sonrisa", "lágrima", "grito", "susurro", "silencio",
    "sueño", "realidad", "ilusión", "verdad", "mentira", "secreto", "misterio",
    "acción", "reacción", "causa", "efecto", "origen", "destino", "principio", "fin",
    "pregunta", "respuesta", "duda", "certeza", "problema", "solución",
    "camino", "ruta", "dirección", "destino", "viaje", "aventura", "exploración",
    "guerra", "paz", "batalla", "lucha", "victoria", "derrota", "triunfo",
    "reino", "imperio", "nación", "república", "democracia", "monarquía",
    "ley", "derecho", "justicia", "libertad", "igualdad", "fraternidad",
    "riqueza", "pobreza", "dinero", "oro", "plata", "comercio", "negocio",
    "arte", "cultura", "música", "danza", "pintura", "escultura", "literatura",
    "poesía", "novela", "cuento", "leyenda", "mito", "fábula", "drama",
    "acto", "escena", "obra", "personaje", "héroe", "villano", "príncipe",
    "reina", "caballero", "dragón", "espada", "escudo", "corona", "trono",
    "magia", "hechizo", "poder", "fuerza", "energía", "espíritu", "alma",
    "cielo", "infierno", "paraíso", "demonio", "ángel", "dios", "diosa",
    "religión", "creencia", "fe", "oración", "rito", "ceremonia", "templo",
    "sabiduría", "conocimiento", "entendimiento", "comprensión", "percepción",
    "intuición", "razón", "lógica", "sentido", "significado", "propósito",
    "análisis", "síntesis", "deducción", "inducción", "abstracción",
    "hipótesis", "teoría", "ley", "principio", "axioma", "postulado",
    "demostración", "prueba", "evidencia", "dato", "hecho", "observación",
    "experimento", "método", "técnica", "procedimiento", "sistema",
    "estructura", "forma", "modelo", "patrón", "diseño", "plan", "estrategia",
    "meta", "objetivo", "propósito", "misión", "visión", "ideal",
    "motivación", "inspiración", "creatividad", "imaginación", "innovación",
    "descubrimiento", "invento", "creación", "obra", "proyecto",
    "colaboración", "cooperación", "equipo", "grupo", "comunidad", "sociedad",
    "comunicación", "conversación", "diálogo", "debate", "discusión",
    "acuerdo", "desacuerdo", "consenso", "conflicto", "negociación",
    "apoyo", "ayuda", "asistencia", "servicio", "contribución",
    "respeto", "tolerancia", "comprensión", "empatía", "solidaridad",
    "honestidad", "integridad", "responsabilidad", "compromiso", "lealtad",
    "perseverancia", "paciencia", "disciplina", "dedicación", "esfuerzo",
    "éxito", "fracaso", "logro", "mérito", "reconocimiento", "premio",
    "cambio", "transformación", "evolución", "desarrollo", "crecimiento",
    "aprendizaje", "educación", "enseñanza", "formación", "entrenamiento",
    "habilidad", "destreza", "capacidad", "talento", "genio", "don",
    "experiencia", "práctica", "sabiduría", "pericia", "maestría",
    "memoria", "recuerdo", "olvido", "nostalgia", "añoranza",
    "expectativa", "esperanza", "ilusión", "sueño", "deseo", "anhelo",
    "decisión", "elección", "opción", "alternativa", "posibilidad",
    "azar", "suerte", "destino", "fortuna", "casualidad", "coincidencia",
    "necesidad", "obligación", "deber", "responsabilidad", "carga",
    "riesgo", "peligro", "amenaza", "protección", "seguridad", "defensa",
    "ataque", "ofensa", "defensa", "resistencia", "fortaleza", "debilidad",
    "salud", "enfermedad", "cura", "medicina", "tratamiento", "terapia",
    "nacimiento", "vida", "muerte", "vejez", "juventud", "infancia",
    "alimento", "nutrición", "dieta", "hambre", "sed", "apetito",
    "ejercicio", "movimiento", "actividad", "descanso", "sueño", "fatiga",
    "familia", "matrimonio", "pareja", "relación", "amistad", "amor",
    "hijo", "padre", "madre", "hermano", "abuelo", "tío", "primo", "sobrino",
]

for b in essential:
    vocab.add(b)

sources = [
    ("https://raw.githubusercontent.com/javierarce/palabras/master/listado-general.txt", "Javier Arce"),
    ("https://raw.githubusercontent.com/JorgeDuenasLerin/diccionario-espanol/master/diccionario.csv", "JorgeDuenasLerin"),
]

for url, name in sources:
    try:
        print(f"Descargando fuente: {name}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode('utf-8', errors='ignore')
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                word = parts[0].strip().lower()
                word = re.sub(r'[^a-záéíóúüñabcdefghijklmnopqrstuvwxyz0123456789-]', '', word)
                if word and len(word) > 1:
                    vocab.add(word)
        print(f"  -> {name}: OK")
    except Exception as e:
        print(f"  -> {name}: ERROR - {e}")

# Generar plurales automáticos para palabras que faltan
extra_plurals = []
for w in list(vocab):
    if w.endswith('z'):
        plural = w[:-1] + 'ces'
        if plural not in vocab:
            extra_plurals.append(plural)
    elif w.endswith('ón'):
        plural = w[:-2] + 'ones'
        if plural not in vocab:
            extra_plurals.append(plural)
    elif w.endswith('ción') or w.endswith('sión'):
        plural = w[:-1] + 'es'
        if plural not in vocab:
            extra_plurals.append(plural)
    elif w.endswith('dad') or w.endswith('tad') or w.endswith('tud'):
        plural = w + 'es'
        if plural not in vocab:
            extra_plurals.append(plural)
    elif w.endswith('a') or w.endswith('e') or w.endswith('o'):
        plural = w + 's'
        if plural not in vocab:
            extra_plurals.append(plural)

for p in extra_plurals:
    vocab.add(p)

with open(output_path, "w") as f:
    for word in sorted(vocab):
        f.write(f"{word}\n")

print(f"\nVocabulario generado: {len(vocab)} palabras únicas en {output_path}")
