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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================ CONFIGURACIÓN ================
CORPUS_DIR = os.getenv(
    "CORPUS_DIR",
    r"C:\Users\roble\OneDrive\Documentos\GitHub\Proyecto-RI-1er-BIm\corpus2_clean"
)
MODEL_PICKLE_PATH = os.getenv("MODEL_PICKLE_PATH", "tfidf_programmers.pkl")
TOKEN_PATTERN = r"[a-z0-9]+"

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
STOPWORDS = set(stopwords.words("english")) | {
    # Stopwords de dominio “programming”
    "code", "function", "variable", "class", "int", "string", "print", "value",
    "return", "python", "java", "error", "output", "input", "type", "method",
    "true", "false", "null", "void", "main", "line", "run", "loop", "data",
    "object", "name", "file", "use", "using", "issue", "problem", "num", "let",
    "know", "would", "could", "also", "help", "maybe", "get", "want", "work"
}
LEMMATIZER = WordNetLemmatizer()

app = FastAPI(title="TF-IDF Cosine API Programmers", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
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

tfidf_vectorizer: Optional[TfidfVectorizer] = None
tfidf_matrix: Optional[np.ndarray] = None
document_ids: List[str] = []
docs_tokens: List[List[str]] = []
docs_raw: List[str] = []

# ====== FUNCIONES DE PREPROCESAMIENTO ======

def add_bigrams_trigrams(tokens: List[str]) -> List[str]:
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    trigrams = [f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}" for i in range(len(tokens)-2)]
    return tokens + bigrams + trigrams

@lru_cache(maxsize=128)
def preprocess_query(text: str) -> List[str]:
    txt = re.sub(r"http\S+", " ", text)
    txt = re.sub(r"[^a-zA-Z0-9\s]", " ", txt).lower()
    tokens = re.findall(TOKEN_PATTERN, txt)
    lemmas = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS]
    return add_bigrams_trigrams(lemmas)

# Puedes ejecutar este método aparte para ampliar tus stopwords de dominio:
def get_top_terms(docs, n=30):
    counter = Counter()
    for doc in docs:
        counter.update(doc)
    return counter.most_common(n)

@lru_cache(maxsize=64)
def rm3_expand(tokens_tuple: Tuple[str, ...], fb_docs: int, fb_terms: int) -> List[str]:
    global tfidf_matrix, docs_tokens
    query_tokens = list(tokens_tuple)
    # Simula RM3: toma los docs más similares, saca términos más frecuentes
    # Calcula la query vector
    query_str = " ".join(query_tokens)
    query_vec = tfidf_vectorizer.transform([query_str])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idxs = np.argsort(sims)[::-1][:fb_docs]
    all_terms = []
    for idx in top_idxs:
        all_terms.extend(docs_tokens[idx])
    freq = Counter(all_terms)
    candidates = [t for t, _ in freq.most_common() if t not in query_tokens and t not in STOPWORDS]
    return candidates[:fb_terms]

def build_or_load_tfidf_index():
    global tfidf_vectorizer, tfidf_matrix, document_ids, docs_tokens, docs_raw
    if os.path.exists(MODEL_PICKLE_PATH):
        with open(MODEL_PICKLE_PATH, "rb") as f:
            data = pickle.load(f)
            tfidf_vectorizer = data["tfidf_vectorizer"]
            tfidf_matrix = data["tfidf_matrix"]
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
        tokens = raw_txt.split()  # Ya está preprocesado/tokenizado
        lemmas = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS]
        tokens_and_ngrams = add_bigrams_trigrams(lemmas)
        docs.append(" ".join(tokens_and_ngrams))
        docs_tokens.append(tokens_and_ngrams)
        raw_docs.append(raw_txt)
    document_ids = ids
    docs_raw = raw_docs

    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(),
        lowercase=False,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1, 3)  # Ya pasamos ngrams en el preprocesamiento
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    with open(MODEL_PICKLE_PATH, "wb") as f:
        pickle.dump({
            "tfidf_vectorizer": tfidf_vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "doc_ids": document_ids,
            "docs_tokens": docs_tokens,
            "docs_raw": docs_raw
        }, f)

def get_snippet(doc_raw: str, query_tokens: List[str], win=25) -> str:
    tokens = doc_raw.split()
    indices = [i for i, t in enumerate(tokens) if t in query_tokens]
    if indices:
        idx = indices[0]
        start = max(0, idx - win//2)
        end = min(len(tokens), idx + win//2)
        return " ".join(tokens[start:end])
    else:
        return " ".join(tokens[:win])

@app.on_event("startup")
def on_startup():
    try:
        build_or_load_tfidf_index()
    except Exception as e:
        raise RuntimeError(f"Error al inicializar índice TF-IDF: {e}")

@app.post("/reload", response_model=ReloadResponse, summary="Reload index")
def reload_index():
    preprocess_query.cache_clear()
    rm3_expand.cache_clear()
    if os.path.exists(MODEL_PICKLE_PATH):
        os.remove(MODEL_PICKLE_PATH)
    build_or_load_tfidf_index()
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
    fb_docs: int = Query(5, ge=1, le=20, description="Docs para feedback"),
    fb_terms: int = Query(10, ge=1, le=50, description="Términos expand")
):
    if tfidf_vectorizer is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="Índice no inicializado")

    start_time = time.time()
    tokens = preprocess_query(query)
    query_str = " ".join(tokens)

    expanded_terms: Optional[List[str]] = None
    if feedback and tokens:
        expanded_terms = rm3_expand(tuple(tokens), fb_docs, fb_terms)
        tokens += expanded_terms
        query_str = " ".join(tokens)

    query_vec = tfidf_vectorizer.transform([query_str])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranking = np.argsort(scores)[::-1]
    sel = ranking[offset: offset + topk]
    results = [
        SearchResult(
            doc_id=document_ids[i],
            score=float(scores[i]),
            snippet=get_snippet(docs_raw[i], tokens)
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

from fastapi.responses import PlainTextResponse

CORPUS_ORIGINAL_DIR = r"C:\Users\roble\OneDrive\Documentos\GitHub\Proyecto-RI-1er-BIm\corpus2"

@app.get("/document/{doc_name}", response_class=PlainTextResponse, summary="Obtener documento original por nombre")
def get_document(doc_name: str):
    base_name = doc_name.replace("_clean", "")
    fname = f"{base_name}.txt"
    path = os.path.join(CORPUS_ORIGINAL_DIR, fname)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Documento '{doc_name}' no encontrado en corpus original")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bm25_api_programmers:app", host="0.0.0.0", port=8000, reload=True)
