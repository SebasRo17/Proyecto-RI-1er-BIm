#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_bm25.py

Evalúa un índice BM25 sobre el corpus BEIR/CQADupStack/mathematica,
usando los qrels para calcular métricas estándar (MAP, nDCG@10, P@10, Recall@100).

Requisitos:
    pip install ir-datasets rank_bm25 pytrec_eval

Uso:
    python eval_bm25.py \
        --clean_corpus_dir "path/to/corpus2_clean" \
        --qrels_dataset "beir/cqadupstack/mathematica" \
        --topk 100
"""

import os, glob, argparse
import ir_datasets
import pytrec_eval
from rank_bm25 import BM25Okapi
print(">>> eval_bm25.py arrancó correctamente", flush=True)

def cargar_corpus_tokens(clean_dir):
    """
    Lee todos los .txt lematizados de clean_dir y devuelve:
      docs_tokens: [ [token1, token2, …], … ]
      doc_ids:     [ doc_id1, doc_id2, … ]
    Asume que cada archivo se llama "<doc_id>_clean.txt".
    """
    docs_tokens, doc_ids = [], []
    pattern = os.path.join(clean_dir, "*.txt")
    for path in sorted(glob.glob(pattern)):
        base = os.path.splitext(os.path.basename(path))[0]
        # si tu sufijo es "_clean", lo quitamos:
        doc_id = base.replace("_clean", "")
        text = open(path, encoding="utf-8").read().strip()
        tokens = text.split()
        docs_tokens.append(tokens)
        doc_ids.append(doc_id)
    return docs_tokens, doc_ids

def cargar_qrels(dataset_name):
    """
    Carga los qrels (ground-truth) del dataset ir_datasets.
    Devuelve un dict: { query_id: { doc_id: relevance_int, … }, … }
    """
    ds = ir_datasets.load(dataset_name)
    qrels = {}
    for qrel in ds.qrels_iter():
        qid, did, rel, _ = qrel  # namedtuple: (query_id, doc_id, relevance, iteration)
        qrels.setdefault(qid, {})[did] = rel
    return qrels, ds

def construir_run(bm25, doc_ids, ds, topk):
    """
    Para cada query en ds.queries_iter(), calcula BM25 scores
    y construye un diccionario:
      run = {
        query_id1: { docA: scoreA, docB: scoreB, … },
        query_id2: { … }
      }
    Solo conserva los topk documentos por consulta.
    """
    run = {}
    for q in ds.queries_iter():
        qid, text = q.query_id, q.text
        # tokenización simple; si quieres lematizar/stopwords, aplica el mismo pipeline
        tokens = text.lower().split()
        scores = bm25.get_scores(tokens)
        # get topk indices
        import numpy as np
        idxs = np.argsort(scores)[::-1][:topk]
        run[qid] = { doc_ids[i]: float(scores[i]) for i in idxs }
    return run

def main(args):
    # 1) Cargar corpus preprocesado
    print("1) Cargando corpus limpio …")
    docs_tokens, doc_ids = cargar_corpus_tokens(args.clean_corpus_dir)
    print(f"   → {len(doc_ids)} documentos.")

    # 2) Construir BM25
    print("2) Entrenando BM25Okapi …")
    bm25 = BM25Okapi(docs_tokens)

    # 3) Cargar qrels y queries
    print("3) Cargando qrels desde ir-datasets …")
    qrels, ds = cargar_qrels(args.qrels_dataset)
    print(f"   → {sum(len(v) for v in qrels.values())} juicios de relevancia.")

    # 4) Construir run
    print(f"4) Generando run top-{args.topk} para cada consulta …")
    run = construir_run(bm25, doc_ids, ds, args.topk)

    # 5) Evaluar con pytrec_eval
    print("5) Evaluando con pytrec_eval …")
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"map", "ndcg_cut_10", "P_10", "recall_100"}
    )
    resultados = evaluator.evaluate(run)

    # 6) Calcular promedios
    import statistics
    metrics = {}
    for m in ["map", "ndcg_cut_10", "P_10", "recall_100"]:
        metrics[m] = statistics.mean([ r[m] for r in resultados.values() ])

    print("\n===== Métricas agregadas =====")
    for m, v in metrics.items():
        print(f"{m:12s}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_corpus_dir", required=True,
                        help="Carpeta con .txt lematizados (ej: corpus2_clean).")
    parser.add_argument("--qrels_dataset", default="beir/cqadupstack/mathematica",
                        help="Identificador ir-datasets para qrels.")
    parser.add_argument("--topk", type=int, default=100,
                        help="Cuántos documentos considerar por consulta.")
    args = parser.parse_args()
    main(args)
