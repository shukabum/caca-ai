from sentence_transformers import SentenceTransformer
import os, pickle, faiss
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2" 
# MODEL_NAME = "microsoft/codebert-base"
# MODEL_NAME = "bigcode/starcoder"

def load_model():
    return SentenceTransformer(MODEL_NAME)

def load_code_snippets():
    snippets = []
    for f in os.listdir("../data/codeSnippets"):
        if f.endswith(".py"):
            with open(f"../data/codeSnippets/{f}", "r",encoding ="utf-8") as file:
                snippets.append({"filename": f, "code": file.read()})
    return snippets

def compute_embeddings(snippets, model):
    codes = [s["code"] for s in snippets]
    embeddings = model.encode(codes, show_progress_bar=True)
    for i, s in enumerate(snippets):
        s["embedding"] = np.array(embeddings[i]).astype("float32")
    return snippets

def save_embeddings(snippets):
    data = {
        "file_names": [s["filename"] for s in snippets],
        "codes": [s["code"] for s in snippets],
        "embeddings": np.array([s["embedding"] for s in snippets])
    }
    with open("data/embeddings/code_embeddings.pkl", "wb") as f:
        pickle.dump(data, f)

def load_embeddings():
    with open("data/embeddings/code_embeddings.pkl", "rb") as f:
        return pickle.load(f)

def build_faiss_index(snippets):
    dim = snippets[0]["embedding"].shape[0]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    embeddings = np.array([s["embedding"] for s in snippets]).astype("float32")
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    index.add(embeddings)
    return index

def embed_query(query, model):
    emb = model.encode([query])[0].astype("float32")
    # Normalize if using inner product FAISS
    emb = emb / np.linalg.norm(emb)
    return emb
