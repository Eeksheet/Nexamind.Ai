"""Critic / verification agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ..llm.prompts import format_evidence
from ..retrieval.base import Evidence
from .base import Agent, clamp01

RECOMMENDATIONS = ("STOP", "RETRIEVE", "REPLAN")


@dataclass
class CriticVerdict:
    supported: bool
    confidence: float
    missing_evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    answer_type_ok: bool = True
    recommendation: str = "STOP"
    reason: str = ""
    parse_ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return dict(supported=self.supported, confidence=self.confidence,
                    missing_evidence=self.missing_evidence, contradictions=self.contradictions,
                    answer_type_ok=self.answer_type_ok, recommendation=self.recommendation,
                    reason=self.reason, parse_ok=self.parse_ok)

    @classmethod
    def disabled(cls) -> "CriticVerdict":
        return cls(True, 1.0, recommendation="STOP", reason="critic disabled")

    @property
    def is_disabled(self) -> bool:
        return self.reason == "critic disabled"

    @property
    def feedback(self) -> str:
        parts = list(self.missing_evidence)
        if self.reason and self.reason not in parts:
            parts.append(self.reason)
        return "; ".join(parts)


class CriticAgent(Agent):
    ROLE = "critic"

    def review(self, question: str, answer: str, summary: str,
               evidence: Sequence[Evidence]) -> CriticVerdict:
        if not answer.strip():
            return CriticVerdict(False, 0.0, ["any answer-bearing evidence"], [], False,
                                 "RETRIEVE", "empty answer")
        obj = self._ask("critic", question=question, answer=answer, summary=summary or "(none)",
                        evidence=format_evidence(evidence), max_tokens=300)
        if not obj:
            return CriticVerdict(False, 0.0, [], [], True, "RETRIEVE", "critic output unparsable", False)
        rec = str(obj.get("recommendation", "")).upper()
        supported = bool(obj.get("supported", False))
        if rec not in RECOMMENDATIONS:
            rec = "STOP" if supported else "RETRIEVE"
        return CriticVerdict(
            supported=supported, confidence=clamp01(obj.get("confidence"), 0.0),
            missing_evidence=[str(x) for x in obj.get("missing_evidence", []) or []][:5],
            contradictions=[str(x) for x in obj.get("contradictions", []) or []][:5],
            answer_type_ok=bool(obj.get("answer_type_ok", True)), recommendation=rec,
            reason=str(obj.get("reason", ""))[:300])
