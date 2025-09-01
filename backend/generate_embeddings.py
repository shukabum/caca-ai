# backend/generate_embeddings.py
import os, pickle
from sentence_transformers import SentenceTransformer
import numpy as np

CODE_SNIPPETS_DIR = "../data/codeSnippets"
EMBEDDINGS_DIR = "../data/embeddings"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "code_embeddings.pkl")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")   # lightweight & fast

file_names = []
code_snippets = []

for fname in os.listdir(CODE_SNIPPETS_DIR):
    if fname.endswith(".py"):
        p = os.path.join(CODE_SNIPPETS_DIR, fname)
        with open(p, "r", encoding="utf-8") as f:
            code = f.read()
        file_names.append(fname)
        code_snippets.append(code)

embeddings = model.encode(code_snippets, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

data = {
    "file_names": file_names,
    "code_snippets": code_snippets,
    "embeddings": embeddings
}

with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"Saved embeddings for {len(file_names)} files to {EMBEDDINGS_FILE}")
