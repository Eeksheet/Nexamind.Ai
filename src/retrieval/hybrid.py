"""Hybrid retriever: BM25 → Dense → weighted fusion → optional cross-encoder rerank.

``KnowledgeIndex`` is the per-question index used by every system (baselines and
multi-agent).  Which components are built is driven by :class:`RetrievalConfig`
so disabled components cost nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..utils.config import RetrievalConfig, resolve_path
from .base import Corpus, Evidence, Unit, normalize_scores
from .bm25 import BM25
from .dense import DenseEncoder, DenseIndex
from .reranker import Reranker

METHODS = ("bm25", "dense", "hybrid", "tfidf")


@dataclass
class RetrievalStats:
    calls: int = 0
    cache_hits: int = 0


class RetrievalCache:
    """In-memory (query, k, alpha, exclude, method) → ranked indices cache."""

    def __init__(self) -> None:
        self._d: Dict[Tuple, List[Tuple[int, float]]] = {}

    def get(self, key: Tuple) -> Optional[List[Tuple[int, float]]]:
        return self._d.get(key)

    def put(self, key: Tuple, v: List[Tuple[int, float]]) -> None:
        self._d[key] = v


class SharedRetrievalResources:
    """Heavy, process-wide objects (encoder, reranker) shared across indexes."""

    def __init__(self, cfg: RetrievalConfig) -> None:
        self.cfg = cfg
        self.encoder: Optional[DenseEncoder] = None
        self.reranker: Optional[Reranker] = None
        self.ensure(cfg)

    def ensure(self, cfg: RetrievalConfig) -> "SharedRetrievalResources":
        """Lazily create the encoder / reranker that ``cfg`` needs (models load on first use)."""
        if cfg.method in ("dense", "hybrid") and self.encoder is None:
            cache_dir = resolve_path(cfg.cache_dir) if cfg.cache_dir else None
            self.encoder = DenseEncoder(cfg.embedding_model, cache_dir, cfg.embedding_batch_size)
        if cfg.reranker and self.reranker is None:
            self.reranker = Reranker(cfg.reranker_model)
        return self

    def flush(self) -> None:
        if self.encoder is not None:
            self.encoder.flush()


class KnowledgeIndex:
    """Sentence-level hybrid index for one question's candidate paragraphs."""

    def __init__(self, paragraphs: Dict[str, Sequence[str]], cfg: RetrievalConfig,
                 shared: Optional[SharedRetrievalResources] = None) -> None:
        if cfg.method not in METHODS:
            raise ValueError(f"unknown retrieval method {cfg.method!r}; expected one of {METHODS}")
        self.cfg = cfg
        self.corpus = Corpus.from_paragraphs(paragraphs)
        if not self.corpus.units:
            raise ValueError("empty corpus: no paragraphs/sentences to index")
        self.shared = (shared or SharedRetrievalResources(cfg)).ensure(cfg)
        self.stats = RetrievalStats()
        self._cache = RetrievalCache()

        self.bm25: Optional[BM25] = None
        self.dense: Optional[DenseIndex] = None
        self._tfidf = None
        if cfg.method in ("bm25", "hybrid"):
            self.bm25 = BM25(self.corpus.units, k1=cfg.bm25_k1, b=cfg.bm25_b)
        if cfg.method in ("dense", "hybrid"):
            assert self.shared.encoder is not None
            self.dense = DenseIndex(self.corpus.units, self.shared.encoder)
        if cfg.method == "tfidf":  # prototype's original "dense" stand-in, kept as a baseline
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(self.corpus.units)
            self._tfidf_M = self._tfidf.transform(self.corpus.units)

    # ------------------------------------------------------------------ scoring
    def component_scores(self, query: str) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        if self.bm25 is not None:
            out["bm25"] = self.bm25.score(query)
        if self.dense is not None:
            out["dense"] = self.dense.score(query)
        if self._tfidf is not None:
            from sklearn.metrics.pairwise import cosine_similarity

            out["tfidf"] = cosine_similarity(self._tfidf.transform([query]), self._tfidf_M)[0]
        return out

    def fused_scores(self, query: str, alpha: Optional[float] = None) -> np.ndarray:
        """hybrid = alpha * norm(bm25) + (1 - alpha) * norm(dense)."""
        alpha = self.cfg.alpha if alpha is None else alpha
        comp = self.component_scores(query)
        how = self.cfg.normalize
        if self.cfg.method == "hybrid":
            return alpha * normalize_scores(comp["bm25"], how) + (1 - alpha) * normalize_scores(comp["dense"], how)
        return normalize_scores(next(iter(comp.values())), how)

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: Optional[int] = None, alpha: Optional[float] = None,
               exclude: Iterable[Unit] = (), hop: int = 0,
               rerank: Optional[bool] = None) -> List[Evidence]:
        k = self.cfg.top_k if k is None else k
        alpha = self.cfg.alpha if alpha is None else alpha
        rerank = self.cfg.reranker if rerank is None else rerank
        excl: Set[Unit] = set(exclude)
        use_rerank = bool(rerank and self.shared.reranker is not None)
        key = (query, k, alpha, tuple(sorted(excl)), use_rerank)
        ranked = self._cache.get(key)
        if ranked is None:
            self.stats.calls += 1
            s = self.fused_scores(query, alpha)
            if excl:
                for i, m in enumerate(self.corpus.meta):
                    if m in excl:
                        s[i] = -np.inf
            n_cand = self.cfg.rerank_candidates if use_rerank else k
            cand = [int(i) for i in np.argsort(-s, kind="stable")[:n_cand] if np.isfinite(s[i])]
            if use_rerank and cand:
                assert self.shared.reranker is not None
                rs = self.shared.reranker.score(query, [self.corpus.units[i] for i in cand])
                order = np.argsort(-rs, kind="stable")[:k]
                ranked = [(cand[int(o)], float(rs[int(o)])) for o in order]
            else:
                ranked = [(i, float(s[i])) for i in cand[:k]]
            self._cache.put(key, ranked)
        else:
            self.stats.cache_hits += 1
        method = self.cfg.method + ("+rerank" if use_rerank else "")
        return [Evidence(title=self.corpus.meta[i][0], sent_idx=self.corpus.meta[i][1],
                         text=self.corpus.units[i], sentence=self.corpus.sentences[i],
                         score=sc, method=method, hop=hop, query=query, rank=r + 1)
                for r, (i, sc) in enumerate(ranked)]

    def rank_all(self, query: str, alpha: Optional[float] = None) -> List[Unit]:
        """Full ranking of every unit (for Recall@k / MRR / nDCG evaluation)."""
        s = self.fused_scores(query, alpha)
        return [self.corpus.meta[int(i)] for i in np.argsort(-s, kind="stable")]

    def __len__(self) -> int:
        return len(self.corpus)
