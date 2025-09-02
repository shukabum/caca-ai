# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from backend.utils import load_model, load_embeddings, build_faiss_index, embed_query
from backend.model import llm_summarize

app = FastAPI()

# Load embedding model and embeddings
embed_model = load_model()
emb_data = load_embeddings()   # dict with keys file_names, code_snippets, embeddings

# Build in-memory snippets list used for FAISS and metadata
snippets_list = [
    {"filename": name, "code": code, "embedding": emb}
    for name, code, emb in zip(emb_data["file_names"], emb_data["code_snippets"], emb_data["embeddings"])
]

faiss_index = build_faiss_index(snippets_list)

class Query(BaseModel):
    code_context: str
    top_k: int = 3
    
class NLQuery(BaseModel):
    question: str
    top_k: int = 3

@app.post("/autocomplete")
def autocomplete(query: Query):
    q_emb = embed_query(query.code_context, embed_model)
    D, I = faiss_index.search(np.array([q_emb]), query.top_k)
    results = []
    for idx in I[0]:
        s = snippets_list[idx]
        # include a small preview (first few lines)
        preview = "\n".join(s["code"].splitlines()[:8])
        results.append({"filename": s["filename"], "preview": preview})
    return {"results": results}

@app.post("/query")
def answer_query(query: NLQuery):
    q_emb = embed_query(query.question, embed_model)
    D, I = faiss_index.search(np.array([q_emb]), query.top_k)
    top_snippets = [snippets_list[i] for i in I[0]]

    results = []
    for snip in top_snippets:
        code = snip["code"]
        # call LLM summarizer (lazy-loaded)
        try:
            explanation = llm_summarize(code,query.question)
        except Exception as e:
            explanation = f"Could not summarize (LLM error): {e}"
        results.append({"filename": snip["filename"], "code": code, "explanation": explanation})
    return {"results": results}
