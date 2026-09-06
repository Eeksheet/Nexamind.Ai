"""BM25 lexical scorer (vectorised port of the prototype implementation)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from .base import tokenize


class BM25:
    """Okapi BM25 over a small in-memory corpus.

    Formula identical to the notebook prototype (``idf = ln(1 + (N-df+0.5)/(df+0.5))``),
    verified against ``rank_bm25.BM25Okapi`` in the test-suite, but with an inverted
    index so scoring is O(|q| * postings) instead of O(|q| * N).
    """

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.toks: List[List[str]] = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.doc_len = np.array([len(t) for t in self.toks], dtype=float)
        self.avg_len = max(float(self.doc_len.mean()) if self.N else 0.0, 1e-9)
        self.postings: Dict[str, Dict[int, int]] = {}
        for i, t in enumerate(self.toks):
            for w, f in Counter(t).items():
                self.postings.setdefault(w, {})[i] = f
        self.idf: Dict[str, float] = {
            w: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.postings.items()
        }

    def score(self, query: str) -> np.ndarray:
        s = np.zeros(self.N)
        denom_norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avg_len)
        for w in tokenize(query):
            post = self.postings.get(w)
            if not post:
                continue
            idf = self.idf[w]
            for i, f in post.items():
                s[i] += idf * f * (self.k1 + 1) / (f + denom_norm[i])
        return s

    def top_k(self, query: str, k: int) -> List[int]:
        s = self.score(query)
        return list(np.argsort(-s, kind="stable")[:k])
