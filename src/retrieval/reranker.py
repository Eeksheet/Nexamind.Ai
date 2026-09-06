"""Optional cross-encoder reranking."""
from __future__ import annotations

import threading
from typing import Dict, List, Sequence

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)
_LOCK = threading.Lock()
_MODELS: Dict[str, object] = {}


def get_cross_encoder(model_name: str):
    with _LOCK:
        if model_name not in _MODELS:
            from sentence_transformers import CrossEncoder

            log.info("loading cross-encoder %s", model_name)
            _MODELS[model_name] = CrossEncoder(model_name, device="cpu")
        return _MODELS[model_name]


class Reranker:
    """Score (query, passage) pairs with a cross-encoder; higher is better."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.n_calls = 0

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0)
        model = get_cross_encoder(self.model_name)
        self.n_calls += 1
        s = model.predict([(query, p) for p in passages], batch_size=self.batch_size,
                          show_progress_bar=False)
        return np.asarray(s, dtype=float)

    def rerank(self, query: str, passages: Sequence[str], k: int) -> List[int]:
        s = self.score(query, passages)
        return list(np.argsort(-s, kind="stable")[:k])
