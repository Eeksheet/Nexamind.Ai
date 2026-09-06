"""Shared retrieval types."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Unit = Tuple[str, int]  # (title, sentence index)

STOPWORDS = frozenset("""a an the of in on at to for from by with and or is are was were be been being
what which who whom whose where when why how does do did that this these those it its as than then
also more most other same such between into during about against""".split())


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def content_terms(text: str) -> List[str]:
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2]


@dataclass
class Evidence:
    """A retrieved sentence with provenance."""

    title: str
    sent_idx: int
    text: str                 # "title: sentence"
    sentence: str
    score: float
    method: str               # bm25 | dense | hybrid | hybrid+rerank | tfidf
    hop: int = 0
    query: str = ""
    is_gold: Optional[bool] = None
    rank: int = 0

    @property
    def unit(self) -> Unit:
        return (self.title, self.sent_idx)

    def to_dict(self) -> dict:
        return dict(title=self.title, sent_idx=self.sent_idx, sentence=self.sentence,
                    score=round(float(self.score), 4), method=self.method, hop=self.hop,
                    query=self.query, is_gold=self.is_gold, rank=self.rank)


@dataclass
class Corpus:
    """Sentence-level corpus for one question (distractor setting)."""

    units: List[str]
    meta: List[Unit]
    sentences: List[str]
    titles: List[str] = field(default_factory=list)

    @classmethod
    def from_paragraphs(cls, paragraphs: Dict[str, Sequence[str]]) -> "Corpus":
        units, meta, sents, titles = [], [], [], []
        for title, ss in paragraphs.items():
            titles.append(title)
            for j, s in enumerate(ss):
                s = s.strip()
                units.append(f"{title}: {s}")
                meta.append((title, j))
                sents.append(s)
        return cls(units, meta, sents, titles)

    def __len__(self) -> int:
        return len(self.units)


def normalize_scores(s: np.ndarray, how: str = "minmax") -> np.ndarray:
    """Normalise a score vector to a comparable range for fusion.

    ``minmax`` maps to [0,1]; a constant vector maps to all-zeros (not all-ones),
    which fixes the prototype's degenerate case where a single lexical hit
    dominated the hybrid score.
    """
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return s
    if how == "none":
        return s
    if how == "zscore":
        sd = s.std()
        return (s - s.mean()) / sd if sd > 1e-9 else np.zeros_like(s)
    rng = np.ptp(s)
    if rng < 1e-9:
        return np.zeros_like(s)
    return (s - s.min()) / rng
