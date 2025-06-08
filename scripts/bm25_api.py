#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bm25_api.py (v5)
API REST con BM25 afinado, CORS, cache, feedback, paginación,
y manejo de errores si no hay documentos.
"""
import os
import glob
import pickle
import re
import time
from typing import List, Optional
from functools import lru_cache

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from collections import Counter

# ==================== CONFIGURACIÓN ====================
# Ajusta la ruta a tu carpeta de .txt limpios
CORPUS_DIR = os.getenv(
    "CORPUS_DIR",
    r"C:\Users\roble\OneDrive\Documentos\GitHub\Proyecto-RI-1er-BIm\corpus2_clean"
)
MODEL_PICKLE_PATH = os.getenv("MODEL_PICKLE_PATH", "bm25_model_v5.pkl")
# Hiperparámetros
K1 = float(os.getenv("BM25_K1", 1.5))
B  = float(os.getenv("BM25_B", 0.75))
# Feedback por defecto
default_fb_docs = 5
default_fb_terms = 10
# Token pattern
TOKEN_PATTERN = r"[a-z0-9]+"

# NLTK resources
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english")) | {"theorem","lemma","proof","example","equation"}
nltk.download("wordnet", quiet=True)
LEMMATIZER = WordNetLemmatizer()

# FastAPI app
app = FastAPI(title="BM25 API v5", version="5.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class SearchResult(BaseModel):
    doc_id: str
    score: float

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

# Globals
bm25_model: Optional[BM25Okapi] = None
document_ids: Optional[List[str]] = None
docs_tokens: Optional[List[List[str]]] = None

# --- Helpers ---
@lru_cache(maxsize=128)
def preprocess_query(text: str) -> List[str]:
    txt = re.sub(r"http\S+", " ", text)
    txt = re.sub(r"[^a-zA-Z0-9\s]", " ", txt).lower()
    tokens = re.findall(TOKEN_PATTERN, txt)
    return [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS]

@lru_cache(maxsize=64)
def rm3_expand(tokens_tuple: tuple, fb_docs: int, fb_terms: int) -> List[str]:
    query_tokens = list(tokens_tuple)
    scores = bm25_model.get_scores(query_tokens)
    idxs = np.argsort(scores)[::-1][:fb_docs]
    all_tokens = []
    for idx in idxs:
        all_tokens.extend(docs_tokens[idx])
    freq = Counter(all_tokens)
    candidates = [t for t, _ in freq.most_common() if t not in query_tokens and t not in STOPWORDS]
    return candidates[:fb_terms]

# --- Build or load index ---
def build_or_load_bm25_index():
    global bm25_model, document_ids, docs_tokens
    # Load from pickle if exists
    if os.path.exists(MODEL_PICKLE_PATH):
        with open(MODEL_PICKLE_PATH, 'rb') as f:
            data = pickle.load(f)
            bm25_model = data['bm25']
            document_ids = data['doc_ids']
            docs_tokens = data['docs_tokens']
        if not document_ids:
            raise RuntimeError("No documents loaded from pickle.")
        return
    # Build from files
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.txt')))
    if not files:
        raise RuntimeError(f"No .txt files found in {CORPUS_DIR}")
    docs, ids = [], []
    for path in files:
        ids.append(os.path.splitext(os.path.basename(path))[0])
        docs.append(open(path, encoding='utf-8').read().split())
    document_ids = ids
    docs_tokens = docs
    bm25_model = BM25Okapi(docs_tokens, k1=K1, b=B)
    with open(MODEL_PICKLE_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25_model, 'doc_ids': document_ids, 'docs_tokens': docs_tokens}, f)

@app.on_event("startup")
def startup():
    try:
        build_or_load_bm25_index()
    except Exception as e:
        raise RuntimeError(f"Failed to load index: {e}")

@app.get("/", summary="Health check")
def health():
    total = len(document_ids) if document_ids else 0
    return {"status": "ok", "total_docs": total}

@app.post("/reload", response_model=ReloadResponse, summary="Reload index")
def reload_index():
    preprocess_query.cache_clear()
    rm3_expand.cache_clear()
    if os.path.exists(MODEL_PICKLE_PATH):
        os.remove(MODEL_PICKLE_PATH)
    build_or_load_bm25_index()
    return ReloadResponse(message="Index reloaded", total_docs=len(document_ids))

@app.get("/search", response_model=SearchResponse, summary="Search endpoint")
def search(
    query: str = Query(..., description="Search query"),
    topk: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    feedback: bool = Query(False),
    fb_docs: int = Query(default_fb_docs, ge=1, le=20),
    fb_terms: int = Query(default_fb_terms, ge=1, le=50)
):
    if bm25_model is None:
        raise HTTPException(500, "Index not initialized")
    start = time.time()
    tokens = preprocess_query(query)
    expanded_terms = None
    if feedback and tokens:
        expanded_terms = rm3_expand(tuple(tokens), fb_docs, fb_terms)
        tokens += expanded_terms
    scores = bm25_model.get_scores(tokens)
    idxs = np.argsort(scores)[::-1]
    sliced = idxs[offset: offset + topk]
    results = [SearchResult(doc_id=document_ids[i], score=float(scores[i])) for i in sliced]
    duration_ms = (time.time() - start) * 1000
    return SearchResponse(
        query=query,
        topk=topk,
        offset=offset,
        duration_ms=round(duration_ms,2),
        expanded=bool(expanded_terms),
        expansion_terms=expanded_terms,
        results=results
    )
