import numpy as np
import pytest

from src.retrieval.base import Corpus, normalize_scores
from src.retrieval.bm25 import BM25
from src.retrieval.hybrid import KnowledgeIndex, SharedRetrievalResources
from src.utils.config import RetrievalConfig

DOCS = {"Alpha": ["Alpha is a river in Spain.", "It was named in 1802 by explorers."],
        "Beta": ["Beta is a mountain in Chile.", "The mountain was first climbed in 1911."],
        "Gamma": ["Gamma is a small town.", "Gamma hosts an annual cheese festival."]}


def test_bm25_ranks_lexical_match_first():
    c = Corpus.from_paragraphs(DOCS)
    bm = BM25(c.units)
    top = bm.top_k("cheese festival", 2)
    assert c.meta[top[0]] == ("Gamma", 1)
    assert len(bm.score("cheese")) == len(c.units)


def test_normalize_scores_constant_vector_is_zero():
    assert np.allclose(normalize_scores(np.array([0.0, 0.0, 0.0])), 0)
    assert np.allclose(normalize_scores(np.array([2.0, 4.0])), [0.0, 1.0])


@pytest.mark.parametrize("method", ["bm25", "tfidf"])
def test_index_search_and_exclude(method):
    cfg = RetrievalConfig(method=method, top_k=2)
    idx = KnowledgeIndex(DOCS, cfg, SharedRetrievalResources(cfg))
    hits = idx.search("when was the mountain climbed", k=2)
    assert hits[0].unit == ("Beta", 1)
    assert hits[0].method == method and hits[0].rank == 1
    again = idx.search("when was the mountain climbed", k=2, exclude={("Beta", 1)})
    assert all(h.unit != ("Beta", 1) for h in again)
    ranked = idx.rank_all("mountain")
    assert len(ranked) == 6 and len(set(ranked)) == 6


def test_hybrid_alpha_extremes_match_components(monkeypatch):
    """alpha=1 -> pure BM25 ordering, alpha=0 -> pure dense ordering (dense stubbed)."""
    cfg = RetrievalConfig(method="hybrid", top_k=3)
    shared = SharedRetrievalResources(cfg)

    class FakeEncoder:
        def encode(self, texts, is_query=False):
            # embed by presence of the word "town" so dense prefers Gamma
            rows = [[1.0, 0.0] if "town" in t.lower() or is_query else [0.0, 1.0] for t in texts]
            return np.array(rows, dtype=np.float32)

    shared.encoder = FakeEncoder()
    idx = KnowledgeIndex(DOCS, cfg, shared)
    comp = idx.component_scores("named 1802 town")
    assert set(comp) >= {"bm25", "dense"}
    bm_order = idx.rank_all("named 1802 town", alpha=1.0)
    assert bm_order[0] == ("Alpha", 1)
    dense_order = idx.rank_all("named 1802 town", alpha=0.0)
    assert dense_order[0] == ("Gamma", 0)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        KnowledgeIndex(DOCS, RetrievalConfig(method="magic"))


@pytest.mark.slow
def test_dense_encoder_real_model():
    pytest.importorskip("sentence_transformers")
    cfg = RetrievalConfig(method="dense", top_k=2)
    idx = KnowledgeIndex(DOCS, cfg, SharedRetrievalResources(cfg))
    hits = idx.search("which peak was ascended", k=1)
    assert hits[0].title == "Beta"
