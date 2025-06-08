import os
import glob
import pickle
import re
import time
from typing import List, Optional, Tuple
from functools import lru_cache
from collections import Counter

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

# ================ CONFIGURACIÓN ================
CORPUS_DIR = os.getenv(
    "CORPUS_DIR",
    r"C:\Users\roble\OneDrive\Documentos\GitHub\Proyecto-RI-1er-BIm\corpus2_clean"
)
MODEL_PICKLE_PATH = os.getenv("MODEL_PICKLE_PATH", "bm25_programmers.pkl")
K1 = float(os.getenv("BM25_K1", 1.2))
B  = float(os.getenv("BM25_B", 0.75))
DEFAULT_FB_DOCS = 5
DEFAULT_FB_TERMS = 10
TOKEN_PATTERN = r"[a-z0-9]+"

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
STOPWORDS = set(stopwords.words("english")) | {
    # Stopwords de dominio “programming”
    "code", "function", "variable", "class", "int", "string", "print", "value",
    "return", "python", "java", "error", "output", "input", "type", "method",
    "true", "false", "null", "void", "main", "line", "run", "loop", "data",
    "object", "name", "file", "use", "using", "issue", "problem", "num", "let",
    "know", "would", "could", "also"
}
LEMMATIZER = WordNetLemmatizer()

app = FastAPI(title="BM25 API Programmers", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchResult(BaseModel):
    doc_id: str
    score: float
    snippet: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    topk: int
    offset: int
    duration_ms: float
    expanded: bool
    expansion_terms: Optional[List[str]]
    results: List[SearchResult]

class ReloadResponse(BaseModel):
    message: str
    total_docs: int

bm25_model: Optional[BM25Okapi] = None
document_ids: List[str] = []
docs_tokens: List[List[str]] = []
docs_raw: List[str] = []

def add_bigrams(tokens: List[str]) -> List[str]:
    # Genera bigramas a partir de la lista de tokens
    return tokens + [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]

@lru_cache(maxsize=128)
def preprocess_query(text: str) -> List[str]:
    txt = re.sub(r"http\S+", " ", text)
    txt = re.sub(r"[^a-zA-Z0-9\s]", " ", txt).lower()
    tokens = re.findall(TOKEN_PATTERN, txt)
    lemmas = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS]
    return add_bigrams(lemmas)

@lru_cache(maxsize=64)
def rm3_expand(tokens_tuple: Tuple[str, ...], fb_docs: int, fb_terms: int) -> List[str]:
    global bm25_model, docs_tokens
    query_tokens = list(tokens_tuple)
    scores = bm25_model.get_scores(query_tokens)
    top_idxs = np.argsort(scores)[::-1][:fb_docs]
    all_terms = []
    for idx in top_idxs:
        all_terms.extend(docs_tokens[idx])
    freq = Counter(all_terms)
    candidates = [t for t, _ in freq.most_common() if t not in query_tokens and t not in STOPWORDS]
    return candidates[:fb_terms]

def build_or_load_bm25_index():
    global bm25_model, document_ids, docs_tokens, docs_raw
    if os.path.exists(MODEL_PICKLE_PATH):
        with open(MODEL_PICKLE_PATH, "rb") as f:
            data = pickle.load(f)
            bm25_model = data["bm25"]
            document_ids = data["doc_ids"]
            docs_tokens = data["docs_tokens"]
            docs_raw = data.get("docs_raw", [""] * len(document_ids))
        if not document_ids:
            raise RuntimeError("El pickle no contiene documentos.")
        return

    paths = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    if not paths:
        raise RuntimeError(f"No se encontraron .txt en {CORPUS_DIR}")

    docs = []
    ids = []
    raw_docs = []
    for p in paths:
        doc_id = os.path.splitext(os.path.basename(p))[0]
        ids.append(doc_id)
        raw_txt = open(p, encoding="utf-8").read()
        tokens = raw_txt.split()  # Suponiendo ya limpio y tokenizado
        lemmas = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS]
        tokens_and_bigrams = add_bigrams(lemmas)
        docs.append(tokens_and_bigrams)
        raw_docs.append(raw_txt)
    document_ids = ids
    docs_tokens = docs
    docs_raw = raw_docs
    bm25_model = BM25Okapi(docs_tokens, k1=K1, b=B)
    with open(MODEL_PICKLE_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25_model,
            "doc_ids": document_ids,
            "docs_tokens": docs_tokens,
            "docs_raw": docs_raw
        }, f)

def get_snippet(doc_tokens: List[str], query_tokens: List[str], win=25) -> str:
    indices = [i for i, t in enumerate(doc_tokens) if t in query_tokens]
    if indices:
        idx = indices[0]
        start = max(0, idx - win//2)
        end = min(len(doc_tokens), idx + win//2)
        return " ".join(doc_tokens[start:end])
    else:
        return " ".join(doc_tokens[:win])

@app.on_event("startup")
def on_startup():
    try:
        build_or_load_bm25_index()
    except Exception as e:
        raise RuntimeError(f"Error al inicializar índice BM25: {e}")

@app.post("/reload", response_model=ReloadResponse, summary="Reload index")
def reload_index():
    preprocess_query.cache_clear()
    rm3_expand.cache_clear()
    if os.path.exists(MODEL_PICKLE_PATH):
        os.remove(MODEL_PICKLE_PATH)
    build_or_load_bm25_index()
    return ReloadResponse(message="Índice recargado", total_docs=len(document_ids))

@app.get("/", summary="Health check")
def health_check():
    total = len(document_ids)
    return {"status": "ok", "total_docs": total}

@app.get("/search", response_model=SearchResponse, summary="Buscar documentos")
def search(
    query: str = Query(..., description="Texto de búsqueda"),
    topk: int = Query(10, ge=1, le=100, description="Número de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
    feedback: bool = Query(False, description="Usar expansión RM3"),
    fb_docs: int = Query(DEFAULT_FB_DOCS, ge=1, le=20, description="Docs para feedback"),
    fb_terms: int = Query(DEFAULT_FB_TERMS, ge=1, le=50, description="Términos expand")
):
    if bm25_model is None:
        raise HTTPException(status_code=500, detail="Índice no inicializado")

    start_time = time.time()
    tokens = preprocess_query(query)

    expanded_terms: Optional[List[str]] = None
    if feedback and tokens:
        expanded_terms = rm3_expand(tuple(tokens), fb_docs, fb_terms)
        tokens += expanded_terms

    scores = bm25_model.get_scores(tokens)
    ranking = np.argsort(scores)[::-1]
    sel = ranking[offset: offset + topk]
    results = [
        SearchResult(
            doc_id=document_ids[i],
            score=float(scores[i]),
            snippet=get_snippet(docs_tokens[i], tokens)
        )
        for i in sel
    ]
    elapsed_ms = (time.time() - start_time) * 1000.0

    return SearchResponse(
        query=query,
        topk=topk,
        offset=offset,
        duration_ms=round(elapsed_ms, 2),
        expanded=bool(expanded_terms),
        expansion_terms=expanded_terms,
        results=results
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bm25_api_programmers:app", host="0.0.0.0", port=8000, reload=True)
