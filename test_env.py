from sentence_transformers import SentenceTransformer
import pdfplumber
import numpy as np
import hnswlib

print("pdfplumber OK")
print("hnswlib OK")
m = SentenceTransformer("all-MiniLM-L6-v2")
print("Modelo cargado. Dimensión embeddings:", m.get_sentence_embedding_dimension())