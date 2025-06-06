#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bm25_api.py

Servicio web (API REST) para indexar un corpus preprocesado con BM25 y
permitir consultas desde una interfaz externa.

Dependencias:
    pip install fastapi uvicorn rank_bm25 nltk

Cómo usar:
    1) Ajusta la constante CORPUS_DIR para que apunte a tu carpeta de archivos limpios (corpus2_clean).
    2) Ejecuta:
         uvicorn bm25_api:app --host 0.0.0.0 --port 8000 --reload
       (o el puerto que prefieras).
    3) Desde cualquier cliente HTTP (navegador, Postman, fetch en JS, Angular/React) 
       pide:
         http://localhost:8000/search?query=matrix%20rank&topk=10
       y obtendrás un JSON con los 10 documentos más relevantes.
"""

import os
import glob
import pickle
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

# ==================== CONFIGURACIÓN ====================

# Carpeta donde están los .txt limpios (uno por documento).
# Cámbiala según tu estructura de directorios.
CORPUS_DIR = r"D:\Universidad\8 - Octavo\Recuperacion de la informacion\Proyecto-RI-1er-BIm\corpus2_clean"

# Nombre del archivo pickle donde podemos guardar/recuperar el índice BM25
MODEL_PICKLE_PATH = "bm25_model.pkl"

# ==================== FIN DE CONFIGURACIÓN ===============

app = FastAPI(
    title="BM25 Retrieval API",
    description="API para consultar documentos más relevantes usando BM25 sobre un corpus preprocesado.",
    version="1.0.0",
)


class SearchResult(BaseModel):
    """
    Esquema de un resultado individual de búsqueda.
    """
    doc_id: str
    score: float


class SearchResponse(BaseModel):
    """
    Esquema de la respuesta al endpoint /search
    """
    query: str
    topk: int
    results: List[SearchResult]


# Variables globales que almacenarán el índice BM25 una vez construido o cargado
bm25_model: Optional[BM25Okapi] = None
document_ids: Optional[List[str]] = None


def build_or_load_bm25_index(corpus_dir: str, pickle_path: str):
    """
    1) Si existe el pickle en disco (pickle_path), lo carga y retorna (bm25, doc_ids).
    2) Si no existe, lee todos los .txt en corpus_dir, construye BM25Okapi y 
       guarda (pickle.dump) la tupla (bm25, doc_ids) en pickle_path para reutilizar luego.
    """
    global bm25_model, document_ids

    # Si ya está cargado en memoria, no lo volvemos a cargar
    if bm25_model is not None and document_ids is not None:
        return bm25_model, document_ids

    # 1) Intentar cargar el pickle si existe
    if os.path.exists(pickle_path):
        print(f"[INFO] Cargando índice BM25 existente desde '{pickle_path}'...")
        with open(pickle_path, "rb") as f_in:
            data = pickle.load(f_in)
            bm25_model = data["bm25"]
            document_ids = data["doc_ids"]
        print(f"[INFO] Índice BM25 cargado. Total de documentos: {len(document_ids)}")
        return bm25_model, document_ids

    # 2) Si no existe pickle, construir de cero
    print(f"[INFO] No se encontró '{pickle_path}'. Construyendo índice BM25 desde '{corpus_dir}'...")

    # Obtener todos los archivos .txt en corpus_dir
    pattern = os.path.join(corpus_dir, "*.txt")
    archivos = sorted(glob.glob(pattern))
    if not archivos:
        raise FileNotFoundError(f"No se hallaron archivos .txt en '{corpus_dir}'")

    docs_tokens = []
    doc_ids_list = []

    for ruta in archivos:
        nombre = os.path.basename(ruta)
        nombre_base, _ = os.path.splitext(nombre)
        with open(ruta, "r", encoding="utf-8") as f:
            texto = f.read().strip()
            tokens = texto.split()  # Ya están lematizados y separados por espacios
            docs_tokens.append(tokens)
            doc_ids_list.append(nombre_base)

    # Construir BM25Okapi
    bm25_model = BM25Okapi(docs_tokens)
    document_ids = doc_ids_list

    # Guardar en disco para reutilizar después
    with open(pickle_path, "wb") as f_out:
        pickle.dump({
            "bm25": bm25_model,
            "doc_ids": document_ids
        }, f_out)
    print(f"[INFO] Índice BM25 construido y guardado en '{pickle_path}'. Documentos: {len(document_ids)}")
    return bm25_model, document_ids


@app.on_event("startup")
def on_startup():
    """
    Al iniciar FastAPI, construir o cargar el índice BM25.
    """
    try:
        build_or_load_bm25_index(CORPUS_DIR, MODEL_PICKLE_PATH)
    except Exception as e:
        # Si hay un error al arrancar (por ejemplo carpeta no existe), lo propagamos
        print(f"[ERROR] Al inicializar BM25: {e}")
        raise e


@app.get("/", summary="Página de bienvenida")
def read_root():
    return {
        "message": "API BM25 en línea. Consulta en /search?query=tu+texto&topk=10"
    }


@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Busca los documentos más relevantes para una consulta dada"
)
def search(
    query: str = Query(..., description="Consulta de búsqueda (palabras separadas por espacios)."),
    topk: int = Query(10, ge=1, le=100, description="Número de resultados a devolver (máx 100).")
):
    """
    Endpoint GET /search?query=...&topk=...
    - query: texto con la consulta. Ej: "matrix rank theorem"
    - topk: cuántos documentos devolver (por defecto 10).
    
    Retorna un JSON con:
    {
      "query": "...",
      "topk": 10,
      "results": [
        {"doc_id": "beir0001", "score": 12.3456},
        {"doc_id": "beir0203", "score": 11.2345},
        ...
      ]
    }
    """
    global bm25_model, document_ids

    # Asegurarnos de que el índice BM25 está cargado
    if bm25_model is None or document_ids is None:
        raise HTTPException(status_code=500, detail="El índice BM25 no está inicializado.")

    # 1) Tokenizar la consulta muy sencillamente (split por espacios).
    #    Si quieres remover stopwords o lematizar la consulta, podrías agregarlo aquí.
    consulta_tokens = query.lower().strip().split()
    if len(consulta_tokens) == 0:
        raise HTTPException(status_code=400, detail="La consulta está vacía después de procesar.")

    # 2) Calcular puntajes BM25
    scores = bm25_model.get_scores(consulta_tokens)

    # 3) Obtener índices de topk documentos (orden descendente)
    #    Hecho con argsort en NumPy (rank_bm25 utiliza NumPy internamente)
    import numpy as np
    idxs_ordenados = np.argsort(scores)[::-1][:topk]

    # 4) Construir lista de resultados
    resultados = []
    for idx in idxs_ordenados:
        doc_id = document_ids[idx]
        score = float(scores[idx])  # convertir a float puro para JSON
        resultados.append(SearchResult(doc_id=doc_id, score=score))

    return SearchResponse(query=query, topk=topk, results=resultados)


# Si prefieres un POST en vez de GET, podrías definir otro endpoint como este:
#
# class SearchRequest(BaseModel):
#     query: str
#     topk: Optional[int] = 10
#
# @app.post("/search", response_model=SearchResponse)
# def search_post(body: SearchRequest):
#     return search(query=body.query, topk=body.topk)
#
# Con esto, en lugar de usar /search?query=... usarías
# POST /search   Content-Type: application/json   {"query":"matrix rank","topk":5}
