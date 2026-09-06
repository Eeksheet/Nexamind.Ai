"""Independent retrieval evaluation (no LLM): BM25 vs dense vs hybrid vs hybrid+reranker,
plus alpha and top-k sweeps.  Gold = supporting-fact sentences of each question."""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..datasets.hotpotqa import Dataset
from ..retrieval.hybrid import KnowledgeIndex, SharedRetrievalResources
from ..utils.config import RetrievalConfig
from ..utils.logging import get_logger
from .metrics import retrieval_metrics

log = get_logger(__name__)

KS = (1, 3, 5, 10)


def _mean_dicts(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def evaluate_method(ds: Dataset, cfg: RetrievalConfig, method: str, alpha: Optional[float] = None,
                    rerank: bool = False, shared: Optional[SharedRetrievalResources] = None,
                    ks: Sequence[int] = KS) -> Dict[str, float]:
    cfg = dataclasses.replace(cfg, method=method, reranker=rerank,
                              alpha=cfg.alpha if alpha is None else alpha)
    shared = (shared or SharedRetrievalResources(cfg)).ensure(cfg)
    rows = []
    kmax = max(ks)
    for ex in ds.examples:
        idx = KnowledgeIndex(ex.paragraphs, cfg, shared)
        if rerank:
            ranked = [e.unit for e in idx.search(ex.question, k=kmax, alpha=cfg.alpha, rerank=True)]
        else:
            ranked = idx.rank_all(ex.question, alpha=cfg.alpha)[:kmax]
        rows.append(retrieval_metrics(ranked, ex.supporting_facts, ks))
    out = _mean_dicts(rows)
    out["n"] = len(rows)
    return out


def compare_methods(ds: Dataset, cfg: RetrievalConfig, shared: Optional[SharedRetrievalResources] = None,
                    methods: Sequence[str] = ("bm25", "tfidf", "dense", "hybrid"),
                    include_rerank: bool = True) -> Dict[str, Dict[str, float]]:
    shared = shared or SharedRetrievalResources(cfg)
    res: Dict[str, Dict[str, float]] = {}
    for m in methods:
        log.info("retrieval eval: %s", m)
        res[m] = evaluate_method(ds, cfg, m, shared=shared)
    if include_rerank:
        log.info("retrieval eval: hybrid+reranker")
        res["hybrid+rerank"] = evaluate_method(ds, cfg, "hybrid", rerank=True, shared=shared)
    return res


def alpha_sweep(ds: Dataset, cfg: RetrievalConfig, alphas: Sequence[float],
                shared: Optional[SharedRetrievalResources] = None) -> Dict[float, Dict[str, float]]:
    shared = shared or SharedRetrievalResources(cfg)
    return {float(a): evaluate_method(ds, cfg, "hybrid", alpha=a, shared=shared) for a in alphas}


def topk_table(res: Dict[str, Dict[str, float]], ks: Sequence[int] = KS) -> List[Dict[str, float]]:
    """Flatten to rows (method, k, recall, precision, ndcg) for plotting/tables."""
    rows = []
    for m, r in res.items():
        for k in ks:
            rows.append({"method": m, "k": k, "recall": r[f"recall@{k}"], "precision": r[f"precision@{k}"],
                         "ndcg": r[f"ndcg@{k}"], "mrr": r["mrr"]})
    return rows
