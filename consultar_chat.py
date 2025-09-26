import ollama
import hnswlib
import pickle
import math
from sentence_transformers import SentenceTransformer

# -----------------------
# Configuración
# -----------------------
DIM = 384
TOP_K = 5
TOKEN_LIMIT = 8000  # estimado para llama3
MAX_HISTORY_PAIRS = 6  # conservar hasta N pares (user+assistant) para la sesión

# -----------------------
# Cargar índice y referencias
# -----------------------
with open("referencias.pkl", "rb") as f:
    referencias = pickle.load(f)

index = hnswlib.Index(space='cosine', dim=DIM)
index.load_index("index_hnsw.bin")

modelo = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------
# Mensaje system base (no cambiar en cada llamada)
# -----------------------
SYSTEM_BASE = (
    "Eres un asistente experto en directivas de la Armada de Chile. "
    "Responde de forma clara y breve usando SOLO la información extraída de los fragmentos provistos. "
    "Siempre cita el archivo PDF y la página exacta (ej: A-006.pdf, pág 2) cuando la respuesta se basa en un fragmento. "
    "Si la información no aparece en los fragmentos provistos, responde exactamente: "
    "\"No se encontró información en las directivas indexadas\". No inventes información."
)

# -----------------------
# Historial de conversación (solo user/assistant turns, sin fragmentos)
# -----------------------
historial = []  # ej.: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]

# -----------------------
# Estimador de tokens (aprox: 1 token ≈ 4 caracteres)
# -----------------------
def estimar_tokens(texto):
    return math.ceil(len(texto) / 4)

# -----------------------
# Función que arma messages sin persistir el contexto
# -----------------------
def build_messages(contexto, pregunta):
    """
    Construye la lista de mensajes a enviar a Ollama:
    - SYSTEM_BASE (siempre)
    - historial (user/assistant previos)
    - contexto actual (como system) -> NO se guarda en historial
    - pregunta actual (user)
    """
    messages = []
    messages.append({"role": "system", "content": SYSTEM_BASE})
    # Añadir turns previos (user/assistant) desde historial
    messages.extend(historial)
    # Añadir contexto actual (no lo guardamos en historial)
    messages.append({"role": "system", "content": f"Fragmentos relevantes:\n{contexto}"})
    # Añadir la pregunta
    messages.append({"role": "user", "content": pregunta})
    return messages

# -----------------------
# Consulta conversacional (mejorada)
# -----------------------
def consultar_conversacional(pregunta, k=TOP_K):
    global historial
    try:
        # 1) Recuperar fragmentos
        query_vec = modelo.encode([pregunta])
        ids, _ = index.knn_query(query_vec, k=k)

        fragmentos = []
        contexto = ""
        for idx in ids[0]:
            ref = referencias[idx]   # suponemos ref = (archivo, pagina, texto)
            fragmentos.append(ref)
            contexto += f"[{ref[0]} — pág {ref[1]}]\n{ref[2]}\n\n"

        # 2) Estimar tokens (aprox)
        tokens_contexto = estimar_tokens(contexto)
        tokens_pregunta = estimar_tokens(pregunta)
        tokens_historial = sum(estimar_tokens(m["content"]) for m in historial)
        tokens_total = tokens_contexto + tokens_pregunta + tokens_historial + 300  # margen

        print("\n📊 Estimación de tokens:")
        print(f"- Fragmentos: {tokens_contexto}")
        print(f"- Pregunta: {tokens_pregunta}")
        print(f"- Historial: {tokens_historial}")
        print(f"- TOTAL: {tokens_total} / {TOKEN_LIMIT}")
        if tokens_total > TOKEN_LIMIT:
            print("⚠️ Advertencia: Podría exceder el límite del modelo.\n")

        # 3) Construir messages sin persistir contexto en historial
        messages = build_messages(contexto, pregunta)

        # 4) Llamar a Ollama
        respuesta = ollama.chat(model="llama3", messages=messages)
        contenido = respuesta["message"]["content"]

        # 5) Actualizar historial SOLO con pregunta y respuesta (no con el contexto)
        historial.append({"role": "user", "content": pregunta})
        historial.append({"role": "assistant", "content": contenido})

        # Mantener historial acotado (últimos MAX_HISTORY_PAIRS pares)
        max_items = MAX_HISTORY_PAIRS * 2
        if len(historial) > max_items:
            # conservar solo los últimos max_items mensajes
            historial = historial[-max_items:]

        # 6) Mostrar fragmentos utilizados (archivo y página)
        print("\n📚 Fragmentos utilizados:")
        for frag in fragmentos:
            print(f"- {frag[0]} — pág {frag[1]}")

        return contenido
    except hnswlib.Error as e:
        return "Error en la búsqueda de similitud. Intenta reformular tu pregunta."
    except Exception as e:
        print(f"Error inesperado: {e}")
        return "Lo siento, ocurrió un error procesando tu consulta."
# -----------------------
# Bucle interactivo
# -----------------------
if __name__ == "__main__":
    print("=== Chat con directivas de la Armada (historial controlado + contexto no persistente) ===")
    print("Escribe 'salir', 'exit' o 'quit' para terminar.\n")

    while True:
        q = input("\n❓ Pregunta: ")
        if q.lower() in ["salir", "exit", "quit"]:
            break
        try:
            r = consultar_conversacional(q)
            print("\n📌 Respuesta:\n", r)
        except Exception as e:
            print("Error al consultar:", e)
