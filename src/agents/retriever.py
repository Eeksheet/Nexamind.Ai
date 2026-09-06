"""Retriever agent + evidence manager.

The retriever owns *query construction* (rewriting, entity carry-over, missing
evidence queries) and delegates scoring to :class:`KnowledgeIndex`.  The
:class:`EvidenceManager` deduplicates evidence across hops and tracks what has
been consumed (the prototype's ``exclude=used`` set, made explicit).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

from ..llm.backend import LLMBackend
from ..retrieval.base import Evidence, Unit, content_terms
from ..retrieval.hybrid import KnowledgeIndex
from .base import Agent

_CAP = re.compile(r"[A-Z][A-Za-z]+(?: (?:of |the |de |van |von )?[A-Z][A-Za-z]+)*")
_ENT_STOP = {"the", "in", "what", "which", "who", "where", "when", "he", "she", "it", "they", "this",
             "that", "its", "his", "her", "there", "are", "is", "were", "was", "those", "these", "as",
             "at", "on", "for", "from", "with", "after", "before", "during", "however", "although", "while",
             "since", "one", "two", "both", "some", "many", "most", "other", "an", "a", "by", "of", "to",
             "he", "we", "you", "i", "but", "and", "or", "not", "no", "yes", "also", "then", "later"}


def extract_entities(texts: Iterable[str], limit: int = 5, exclude: Iterable[str] = (),
                     titles: Iterable[str] = ()) -> List[str]:
    """Entity extraction for carry-over: paragraph titles first (most reliable bridge
    entities in HotpotQA), then frequency-ranked capitalised spans; question entities excluded."""
    ex = {e.lower() for e in exclude}
    cnt: Counter[str] = Counter()
    for t in titles:
        clean = re.sub(r"\s*\(.*?\)\s*$", "", t).strip()
        if clean and clean.lower() not in ex:
            cnt[clean] += 2
    for t in texts:
        body = t.split(": ", 1)[-1]
        for e in _CAP.findall(body):
            if e.lower() in _ENT_STOP or len(e) < 3 or e.lower() in ex:
                continue
            if len(e.split()) == 1 and (body.startswith(e) or f". {e}" in body) and not e.isupper():
                continue  # sentence-initial single capitalised word: ambiguous
            cnt[e] += 1
    return [e for e, _ in cnt.most_common(limit)]


@dataclass
class EvidenceManager:
    """Deduplicated, ordered evidence pool across hops."""

    evidence: List[Evidence] = field(default_factory=list)
    used: Set[Unit] = field(default_factory=set)
    per_hop_new: List[int] = field(default_factory=list)

    def add(self, hits: Sequence[Evidence]) -> List[Evidence]:
        new = []
        for h in hits:
            if h.unit not in self.used:
                self.used.add(h.unit)
                self.evidence.append(h)
                new.append(h)
        self.per_hop_new.append(len(new))
        return new

    def titles(self) -> List[str]:
        seen: List[str] = []
        for e in self.evidence:
            if e.title not in seen:
                seen.append(e.title)
        return seen

    def summary(self, max_items: int = 6) -> str:
        if not self.evidence:
            return "(none)"
        return "\n".join(f"- {e.text[:160]}" for e in self.evidence[-max_items:])

    def mark_gold(self, gold: Iterable[Unit]) -> None:
        g = set(gold)
        for e in self.evidence:
            e.is_gold = e.unit in g

    def __len__(self) -> int:
        return len(self.evidence)


@dataclass
class RetrievalRequest:
    query: str
    target: str            # initial | entity | bridge | missing_evidence | plain
    subquestion: str
    entities: List[str]
    rewritten: bool

    def to_dict(self) -> Dict:
        return dict(query=self.query, target=self.target, subquestion=self.subquestion,
                    entities=self.entities, rewritten=self.rewritten)


class RetrieverAgent(Agent):
    """Builds hop queries and retrieves non-redundant evidence."""

    ROLE = "rewrite"

    def __init__(self, llm: LLMBackend, index: KnowledgeIndex, top_k: int = 4,
                 use_query_rewriting: bool = True, use_entity_carryover: bool = True) -> None:
        super().__init__(llm)
        self.index = index
        self.top_k = top_k
        self.use_query_rewriting = use_query_rewriting
        self.use_entity_carryover = use_entity_carryover
        self.retrieval_calls = 0
        self.requests: List[RetrievalRequest] = []

    # ---------------------------------------------------------------- queries
    def build_query(self, question: str, subquestion: str, hop: int, entities: Sequence[str],
                    manager: EvidenceManager, feedback: Optional[str] = None) -> RetrievalRequest:
        ents = list(entities) if self.use_entity_carryover else []
        if not self.use_query_rewriting:
            q = subquestion if hop == 0 or not ents else f"{subquestion} {' '.join(ents[:3])}"
            return RetrievalRequest(q, "plain", subquestion, ents, rewritten=False)
        if hop == 0 and not feedback:
            # First hop: the sub-question is already the retrieval query (no LLM needed).
            return RetrievalRequest(subquestion, "initial", subquestion, [], rewritten=False)
        obj = self._ask("rewrite", question=question, subquestion=subquestion,
                        entities=", ".join(ents) or "(none)", evidence=manager.summary(),
                        feedback=feedback or "(none)", max_tokens=150)
        query = str(obj.get("query", "")).strip()
        if not query:
            query = f"{subquestion} {' '.join(ents[:3])}".strip()
        target = obj.get("target", "bridge" if ents else "initial")
        if target not in ("initial", "entity", "bridge", "missing_evidence"):
            target = "bridge"
        return RetrievalRequest(query, target, subquestion, ents, rewritten=True)

    # -------------------------------------------------------------- retrieval
    def retrieve(self, request: RetrievalRequest, manager: EvidenceManager, hop: int,
                 k: Optional[int] = None) -> List[Evidence]:
        self.retrieval_calls += 1
        self.requests.append(request)
        hits = self.index.search(request.query, k=k or self.top_k, exclude=manager.used, hop=hop)
        return manager.add(hits)

    def discover_entities(self, manager: EvidenceManager, question: str, limit: int = 5) -> List[str]:
        if not self.use_entity_carryover:
            return []
        q_ents = _CAP.findall(question)
        return extract_entities([e.text for e in manager.evidence], limit=limit, exclude=q_ents,
                                titles=[e.title for e in manager.evidence])

    @staticmethod
    def missing_terms(question: str, manager: EvidenceManager) -> List[str]:
        have = set(content_terms(" ".join(e.text for e in manager.evidence)))
        return [t for t in content_terms(question) if t not in have]
