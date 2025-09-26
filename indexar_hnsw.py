import os, pickle, re
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
import hnswlib

# -----------------------
# Configuración
# -----------------------
MODEL = "all-MiniLM-L6-v2"
CARPETA = "./directivas/DGPA"
INDEX_FILE = "index_hnsw.bin"
REF_FILE = "referencias.pkl"

# -----------------------
# Extraer fragmentos de PDF
# -----------------------
def extraer_fragmentos(ruta_pdf, max_chars=700, overlap=150):
    """
    Divide cada página en fragmentos con solapamiento para no perder contexto.
    - max_chars: tamaño máximo de cada fragmento
    - overlap: cantidad de caracteres que se repite entre fragmentos consecutivos
    """
    frags = []
    try:
        with pdfplumber.open(ruta_pdf) as pdf:
            for num, pagina in enumerate(pdf.pages, start=1):
                texto = pagina.extract_text()
                if texto:
                    #texto = texto.replace("\n", " ").strip()
                    texto = re.sub(r'\s+', ' ', texto.replace('\n', ' ')).strip()
                    texto = re.sub(r'[^\w\s\.\,\;\:\-\(\)]', '', texto)  # limpiar caracteres especiales
                    if not texto:
                        continue
                    start = 0
                    while start < len(texto):
                        end = min(start + max_chars, len(texto))
                        fragmento = texto[start:end].strip()
                        if len(fragmento) > 50:  # evitar basura muy corta
                            frags.append((os.path.basename(ruta_pdf), num, fragmento))
                        start += max_chars - overlap
        return frags
    except Exception as e:
        print(f"❌ Error procesando {ruta_pdf}: {e}")
        return []

# -----------------------
# Construcción del índice
# -----------------------
def main():
    modelo = SentenceTransformer(MODEL)
    todos = []

    # 1️⃣ Extraer fragmentos de todos los PDFs
    for fname in os.listdir(CARPETA):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(CARPETA, fname)
            print("📄 Procesando", path)
            todos.extend(extraer_fragmentos(path))

    if not todos:
        print("⚠️ No se encontraron fragmentos. Revisa la carpeta.")
        return

    # 2️⃣ Generar embeddings
    textos = [t[2] for t in todos]
    print(f"🔎 Generando embeddings para {len(textos)} fragmentos...")
    emb = modelo.encode(textos, convert_to_numpy=True, show_progress_bar=True)

    # 3️⃣ Crear índice HNSW
    dim = emb.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(emb), ef_construction=200, M=16)
    index.add_items(emb, ids=np.arange(len(emb)))
    index.set_ef(50)

    # 4️⃣ Guardar índice y referencias
    print("💾 Guardando índice y referencias...")
    index.save_index(INDEX_FILE)
    with open(REF_FILE, "wb") as f:
        pickle.dump(todos, f)

    print(f"✅ Listo. {len(todos)} fragmentos indexados.")

# -----------------------
# Ejecutar
# -----------------------
if __name__ == "__main__":
    main()
