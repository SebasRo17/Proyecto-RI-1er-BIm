#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
eval_bm25_programmers_enhanced.py

Evalúa BM25 sobre el corpus CLEAN de "programmers" (beir/cqadupstack/programmers),
incorporando:

 - Unigrams + bigrams en índice y queries
 - Pseudo‐relevance feedback RM3 opcional
 - Ajuste de hiperparámetros K1 y B desde la línea de comandos

Métricas: MAP, nDCG@10, P@10, Recall@100.

Requisitos:
    pip install ir-datasets rank_bm25 pytrec_eval nltk

Uso:
    python eval_bm25_programmers_enhanced.py \
      --clean_corpus_dir "ruta/a/corpus2_clean" \
      --qrels_dataset "beir/cqadupstack/programmers" \
      --topk 100 \
      [--k1 1.5] [--b 0.75] [--feedback] [--fb_docs 5] [--fb_terms 10]
"""

import os
import glob
import argparse
import ir_datasets
import pytrec_eval
import numpy as np
import re
import nltk
from rank_bm25 import BM25Okapi
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ——— Configuración por defecto ———
DEFAULT_K1        = 1.5
DEFAULT_B         = 0.75
DEFAULT_FB_DOCS   = 5
DEFAULT_FB_TERMS  = 10
TOKEN_PATTERN     = r"[a-z0-9]+"

DOMAIN_STOPWORDS = {
    "code","function","variable","class","int","string","print","value",
    "return","python","java","error","output","input","type","method",
    "true","false","null","void","main","line","run","loop","data",
    "object","name","file","use","using","issue","problem","num","let",
    "know","would","could","also"
}

# Descarga recursos NLTK
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
STOPWORDS    = set(stopwords.words("english")) | DOMAIN_STOPWORDS
LEMMATIZER   = WordNetLemmatizer()

def limpiar_y_tokenizar(text: str):
    """
    Limpia texto, tokeniza, lematiza y filtra stopwords.
    Luego genera bigrams de la lista de lemmas.
    """
    txt = re.sub(r"http\S+", " ", text)
    txt = re.sub(r"[^a-zA-Z0-9\s]", " ", txt).lower()
    toks = re.findall(TOKEN_PATTERN, txt)
    lemmas = [LEMMATIZER.lemmatize(t) for t in toks if t not in STOPWORDS]
    # añadir bigrams
    bigrams = [f"{lemmas[i]}_{lemmas[i+1]}" for i in range(len(lemmas)-1)]
    return lemmas + bigrams

def cargar_corpus_tokens(clean_dir):
    """
    Lee .txt lematizados de clean_dir y devuelve:
      docs_tokens: [ [term_1, ..., bigram_n], ... ]
      doc_ids:     [ id1, id2, ... ]
    """
    docs_tokens, doc_ids = [], []
    for path in sorted(glob.glob(os.path.join(clean_dir, "*.txt"))):
        base = os.path.splitext(os.path.basename(path))[0]
        doc_id = base.replace("_clean", "")
        tokens = open(path, encoding="utf-8").read().split()
        # añadimos bigrams
        bigr = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        docs_tokens.append(tokens + bigr)
        doc_ids.append(doc_id)
    return docs_tokens, doc_ids

def cargar_qrels(dataset_name):
    """
    Carga qrels de ir_datasets.
    Devuelve qrels dict y dataset para iterar queries.
    """
    ds = ir_datasets.load(dataset_name)
    qrels = {}
    for qrel in ds.qrels_iter():
        qid, did, rel, _ = qrel
        qrels.setdefault(qid, {})[did] = rel
    return qrels, ds

def rm3_expand(bm25, docs_tokens, tokens, fb_docs, fb_terms):
    """
    RM3: expande tokens con términos frecuentes de top-docs.
    """
    scores = bm25.get_scores(tokens)
    top_idxs = np.argsort(scores)[::-1][:fb_docs]
    pool = []
    for i in top_idxs:
        pool.extend(docs_tokens[i])
    freq = Counter(pool)
    # candidatos no en query y no stopwords
    cand = [t for t,_ in freq.most_common() if t not in tokens and t not in STOPWORDS]
    return cand[:fb_terms]

def construir_run(bm25, docs_tokens, doc_ids, ds, topk, feedback, fb_docs, fb_terms):
    """
    Para cada query:
     - tokeniza + bigrams
     - opcionalmente expande con RM3
     - ejecuta bm25.get_scores
    Devuelve run dict.
    """
    run = {}
    for q in ds.queries_iter():
        qid, text = q.query_id, q.text
        tokens = limpiar_y_tokenizar(text)
        if feedback and tokens:
            expanded = rm3_expand(bm25, docs_tokens, tokens, fb_docs, fb_terms)
            tokens += expanded
        scores = bm25.get_scores(tokens)
        idxs = np.argsort(scores)[::-1][:topk]
        run[qid] = { doc_ids[i]: float(scores[i]) for i in idxs }
    return run

def main(args):
    # 1) Carga corpus
    print("1) Cargando corpus limpio …")
    docs_tokens, doc_ids = cargar_corpus_tokens(args.clean_corpus_dir)
    print(f"   → {len(doc_ids)} documentos.")

    # 2) Construye BM25 con K1/B
    print(f"2) Creando BM25Okapi (k1={args.k1}, b={args.b}) …")
    bm25 = BM25Okapi(docs_tokens, k1=args.k1, b=args.b)

    # 3) Carga qrels
    print("3) Cargando qrels y queries …")
    qrels, ds = cargar_qrels(args.qrels_dataset)
    total_q = sum(len(v) for v in qrels.values())
    print(f"   → {total_q} juicios de relevancia.")

    # 4) Construir run
    fb_flag = "ON" if args.feedback else "OFF"
    print(f"4) Generando run top-{args.topk} (feedback={fb_flag}) …")
    run = construir_run(
        bm25, docs_tokens, doc_ids, ds,
        args.topk, args.feedback, args.fb_docs, args.fb_terms
    )

    # 5) Evalúa
    print("5) Evaluando con pytrec_eval …")
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"map", "ndcg_cut_10", "P_10", "recall_100"}
    )
    results = evaluator.evaluate(run)

    # 6) Métricas agregadas
    import statistics
    metrics = {
        "map":        statistics.mean([r["map"]         for r in results.values()]),
        "ndcg_cut_10":statistics.mean([r["ndcg_cut_10"] for r in results.values()]),
        "P_10":       statistics.mean([r["P_10"]        for r in results.values()]),
        "recall_100": statistics.mean([r["recall_100"]  for r in results.values()])
    }
    print("\n===== Métricas agregadas =====")
    for m, v in metrics.items():
        print(f"{m:12s}: {v:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clean_corpus_dir", required=True,
                   help="Carpeta con .txt lematizados (corpus2_clean).")
    p.add_argument("--qrels_dataset", default="beir/cqadupstack/programmers",
                   help="Dataset IR (programmers).")
    p.add_argument("--topk", type=int, default=100,
                   help="Docs a recuperar por query.")
    p.add_argument("--k1",       type=float, default=DEFAULT_K1,       help="BM25 k1")
    p.add_argument("--b",        type=float, default=DEFAULT_B,        help="BM25 b")
    p.add_argument("--feedback", action="store_true",                help="Usar RM3")
    p.add_argument("--fb_docs",  type=int,   default=DEFAULT_FB_DOCS, help="RM3 fb_docs")
    p.add_argument("--fb_terms", type=int,   default=DEFAULT_FB_TERMS,help="RM3 fb_terms")
    args = p.parse_args()
    main(args)
