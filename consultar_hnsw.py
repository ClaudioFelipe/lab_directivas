# consultar_hnsw.py
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import hnswlib

MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "index_hnsw.bin"
REF_FILE = "referencias.pkl"

def buscar(pregunta, k=3):
    modelo = SentenceTransformer(MODEL)
    q_emb = modelo.encode([pregunta], convert_to_numpy=True)

    p = hnswlib.Index(space='cosine', dim=q_emb.shape[1])
    p.load_index(INDEX_FILE)
    p.set_ef(50)

    labels, distances = p.knn_query(q_emb, k=k)
    with open(REF_FILE, "rb") as f:
        refs = pickle.load(f)

    resultados = []
    for lab in labels[0]:
        doc, pagina, frag = refs[int(lab)]
        resultados.append((doc, pagina, frag))
    return resultados

if __name__ == "__main__":
    pregunta = input("🔎 Escribe tu pregunta: ")
    res = buscar(pregunta, k=5)
    if res:
        for doc, pag, frag in res:
            print(f"\n[{doc} — pág {pag}]\n{frag}\n")
    else:
        print("No se encontraron resultados.")
