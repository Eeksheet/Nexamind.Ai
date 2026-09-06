"""Reasoner agent: composes retrieved evidence into a short, cited answer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ..llm.prompts import format_evidence
from ..retrieval.base import Evidence
from .base import Agent, clamp01


@dataclass
class ReasonerOutput:
    answer: str
    reasoning_summary: str
    evidence_ids: List[int]
    confidence: float
    supporting_units: List[tuple] = field(default_factory=list)
    parse_ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return dict(answer=self.answer, reasoning_summary=self.reasoning_summary,
                    evidence_ids=self.evidence_ids, confidence=self.confidence,
                    supporting_units=[list(u) for u in self.supporting_units], parse_ok=self.parse_ok)


class ReasonerAgent(Agent):
    ROLE = "reasoner"

    def answer(self, question: str, evidence: Sequence[Evidence]) -> ReasonerOutput:
        obj = self._ask("reasoner", question=question, evidence=format_evidence(evidence), max_tokens=300)
        ans = str(obj.get("answer", "")).strip().strip('."\'')
        ids: List[int] = []
        for i in obj.get("evidence_ids", []) or []:
            try:
                j = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= j < len(evidence) and j not in ids:
                ids.append(j)
        return ReasonerOutput(answer=ans, reasoning_summary=str(obj.get("reasoning_summary", ""))[:400],
                              evidence_ids=ids, confidence=clamp01(obj.get("confidence"), 0.0),
                              supporting_units=[evidence[j].unit for j in ids], parse_ok=bool(obj))

    def closed_book(self, question: str) -> ReasonerOutput:
        """Baseline: answer from parametric memory only (no evidence)."""
        obj = self._ask("closed_book", question=question, max_tokens=60)
        return ReasonerOutput(answer=str(obj.get("answer", "")).strip().strip('."\''),
                              reasoning_summary="closed-book", evidence_ids=[],
                              confidence=clamp01(obj.get("confidence"), 0.0), parse_ok=bool(obj))
