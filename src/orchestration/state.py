"""Explicit orchestrator state and execution trace."""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agents.critic import CriticVerdict
from ..agents.planner import Plan
from ..agents.reasoner import ReasonerOutput
from ..agents.retriever import EvidenceManager
from ..llm.backend import Usage


class Termination(str, enum.Enum):
    CRITIC_APPROVED = "critic_approved"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    MAX_HOPS = "max_hops"
    BUDGET_EXCEEDED = "budget_exceeded"
    NO_NEW_EVIDENCE = "no_new_evidence"
    REPEATED_RETRIEVAL = "repeated_retrieval"
    NOT_ITERATIVE = "single_pass"
    ERROR = "error"


@dataclass
class TraceStep:
    hop: int
    agent: str                      # planner | retriever | reasoner | critic | orchestrator
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return dict(hop=self.hop, agent=self.agent, action=self.action, **self.payload)


@dataclass
class OrchestratorState:
    question: str
    qid: str = ""
    plan: Optional[Plan] = None
    completed_subquestions: List[str] = field(default_factory=list)
    evidence: EvidenceManager = field(default_factory=EvidenceManager)
    entities: List[str] = field(default_factory=list)
    candidate: Optional[ReasonerOutput] = None
    critic: Optional[CriticVerdict] = None
    hop: int = 0
    retrieval_calls: int = 0
    queries_issued: List[str] = field(default_factory=list)
    usage_start: Usage = field(default_factory=Usage)
    usage: Usage = field(default_factory=Usage)
    t_start: float = field(default_factory=time.perf_counter)
    latency_s: float = 0.0
    trace: List[TraceStep] = field(default_factory=list)
    termination: Optional[Termination] = None
    error: Optional[str] = None

    # ---------------------------------------------------------------- helpers
    def log(self, hop: int, agent: str, action: str, **payload: Any) -> None:
        self.trace.append(TraceStep(hop, agent, action, payload))

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t_start

    @property
    def answer(self) -> str:
        return self.candidate.answer if self.candidate else ""

    @property
    def confidence(self) -> float:
        if self.critic is not None and self.critic.reason != "critic disabled":
            return self.critic.confidence
        return self.candidate.confidence if self.candidate else 0.0

    def finish(self, reason: Termination, usage_now: Usage) -> None:
        self.termination = reason
        self.usage = usage_now.diff(self.usage_start)
        self.latency_s = self.elapsed
        self.log(self.hop, "orchestrator", "terminate", reason=reason.value)


@dataclass
class RunResult:
    """Uniform output of every system (baseline or multi-agent)."""

    qid: str
    question: str
    answer: str
    confidence: float
    evidence: List[Dict[str, Any]]
    supporting_units: List[List[Any]]
    plan: Dict[str, Any]
    trace: List[Dict[str, Any]]
    hops: int
    llm_calls: int
    retrieval_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_s: float
    termination: str
    system: str
    critic: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @property
    def evidence_units(self) -> List[tuple]:
        return [(e["title"], e["sent_idx"]) for e in self.evidence]

    @classmethod
    def from_state(cls, st: OrchestratorState, system: str) -> "RunResult":
        return cls(
            qid=st.qid, question=st.question, answer=st.answer, confidence=round(st.confidence, 4),
            evidence=[e.to_dict() for e in st.evidence.evidence],
            supporting_units=[list(u) for u in (st.candidate.supporting_units if st.candidate else [])],
            plan=st.plan.to_dict() if st.plan else {}, trace=[t.to_dict() for t in st.trace],
            hops=st.hop, llm_calls=st.usage.calls, retrieval_calls=st.retrieval_calls,
            input_tokens=st.usage.input_tokens, output_tokens=st.usage.output_tokens,
            total_tokens=st.usage.total_tokens, cost_usd=round(st.usage.cost_usd, 6),
            latency_s=round(st.latency_s, 4),
            termination=st.termination.value if st.termination else "unknown", system=system,
            critic=st.critic.to_dict() if st.critic else None, error=st.error)
