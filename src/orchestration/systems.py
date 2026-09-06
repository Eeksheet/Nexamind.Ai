"""Baseline systems and the registry mapping experiment names to callables.

Every system has the signature ``(Example) -> RunResult`` so the evaluator
treats them uniformly.
"""
from __future__ import annotations

from typing import Callable, Optional

from ..agents.critic import CriticVerdict
from ..agents.planner import Plan
from ..agents.reasoner import ReasonerAgent
from ..agents.retriever import RetrieverAgent
from ..datasets.hotpotqa import Example
from ..llm.backend import LLMBackend
from ..retrieval.hybrid import KnowledgeIndex, SharedRetrievalResources
from ..utils.config import Config
from .orchestrator import Orchestrator
from .state import OrchestratorState, RunResult, Termination

System = Callable[[Example], RunResult]

SYSTEM_KINDS = ("closed_book", "single_pass", "single_agent_iterative", "multi_agent")


class ClosedBookSystem:
    """No retrieval: isolates how much of the score comes from parametric memory."""

    def __init__(self, llm: LLMBackend, name: str = "closed_book") -> None:
        self.llm, self.name = llm, name

    def __call__(self, ex: Example) -> RunResult:
        st = OrchestratorState(ex.question, ex.qid, usage_start=self.llm.usage.snapshot())
        st.plan = Plan.trivial(ex.question)
        st.candidate = ReasonerAgent(self.llm).closed_book(ex.question)
        st.hop = 1
        st.log(1, "reasoner", "closed_book", **st.candidate.to_dict())
        st.finish(Termination.NOT_ITERATIVE, self.llm.usage)
        return RunResult.from_state(st, self.name)


class SinglePassRAG:
    """Vanilla RAG: one retrieval with the raw question, one reasoning call."""

    def __init__(self, cfg: Config, llm: LLMBackend, shared: Optional[SharedRetrievalResources] = None,
                 name: str = "single_pass") -> None:
        self.cfg, self.llm, self.name = cfg, llm, name
        self.shared = shared or SharedRetrievalResources(cfg.retrieval)

    def __call__(self, ex: Example) -> RunResult:
        st = OrchestratorState(ex.question, ex.qid, usage_start=self.llm.usage.snapshot())
        st.plan = Plan.trivial(ex.question)
        idx = KnowledgeIndex(ex.paragraphs, self.cfg.retrieval, self.shared)
        hits = idx.search(ex.question, k=self.cfg.retrieval.top_k, hop=0)
        new = st.evidence.add(hits)
        st.retrieval_calls = 1
        st.queries_issued.append(ex.question)
        st.log(0, "retriever", "retrieve", query=ex.question, target="initial", n_new=len(new),
               evidence=[e.to_dict() for e in new])
        st.hop = 1
        st.candidate = ReasonerAgent(self.llm).answer(ex.question, st.evidence.evidence)
        st.log(1, "reasoner", "answer", **st.candidate.to_dict())
        st.critic = CriticVerdict.disabled()
        st.finish(Termination.NOT_ITERATIVE, self.llm.usage)
        st.evidence.mark_gold(ex.supporting_facts)
        return RunResult.from_state(st, self.name)


class SingleAgentIterativeRAG:
    """One agent, no planner/critic: retrieve → answer → self-assess → re-retrieve.

    Continues while the reasoner's own confidence is below the threshold, using
    entity carry-over and (LLM) query rewriting, up to ``max_hops``.  This is the
    strongest single-agent control: it has iteration but no role specialisation.
    """

    def __init__(self, cfg: Config, llm: LLMBackend, shared: Optional[SharedRetrievalResources] = None,
                 name: str = "single_agent_iterative") -> None:
        self.cfg, self.llm, self.name = cfg, llm, name
        self.shared = shared or SharedRetrievalResources(cfg.retrieval)

    def __call__(self, ex: Example) -> RunResult:
        cfg, budget = self.cfg, self.cfg.orchestrator.budget
        st = OrchestratorState(ex.question, ex.qid, usage_start=self.llm.usage.snapshot())
        st.plan = Plan.trivial(ex.question)
        idx = KnowledgeIndex(ex.paragraphs, cfg.retrieval, self.shared)
        retr = RetrieverAgent(self.llm, idx, cfg.retrieval.top_k, use_query_rewriting=True,
                              use_entity_carryover=True)
        reasoner = ReasonerAgent(self.llm)
        q = ex.question
        while True:
            feedback = None
            if st.candidate is not None:
                feedback = f"previous answer '{st.candidate.answer}' had low confidence"
            req = retr.build_query(q, q, st.hop, st.entities, st.evidence, feedback)
            if st.hop > 0 and req.query.lower() in {x.lower() for x in st.queries_issued}:
                st.finish(Termination.REPEATED_RETRIEVAL, self.llm.usage)
                break
            st.queries_issued.append(req.query)
            new = retr.retrieve(req, st.evidence, hop=st.hop)
            st.retrieval_calls = retr.retrieval_calls
            st.entities = retr.discover_entities(st.evidence, q)
            st.log(st.hop, "retriever", "retrieve", **req.to_dict(), n_new=len(new),
                   evidence=[e.to_dict() for e in new], carried_entities=st.entities[:5])
            st.hop += 1
            st.candidate = reasoner.answer(q, st.evidence.evidence)
            st.log(st.hop, "reasoner", "answer", **st.candidate.to_dict())
            st.critic = CriticVerdict.disabled()
            if st.candidate.confidence >= budget.confidence_threshold:
                st.finish(Termination.CONFIDENCE_THRESHOLD, self.llm.usage)
                break
            if st.hop >= budget.max_hops:
                st.finish(Termination.MAX_HOPS, self.llm.usage)
                break
            if not new:
                st.finish(Termination.NO_NEW_EVIDENCE, self.llm.usage)
                break
        st.evidence.mark_gold(ex.supporting_facts)
        return RunResult.from_state(st, self.name)


def build_system(kind: str, cfg: Config, llm: LLMBackend,
                 shared: Optional[SharedRetrievalResources] = None, name: Optional[str] = None) -> System:
    name = name or kind
    if kind == "closed_book":
        return ClosedBookSystem(llm, name)
    if kind == "single_pass":
        return SinglePassRAG(cfg, llm, shared, name)
    if kind == "single_agent_iterative":
        return SingleAgentIterativeRAG(cfg, llm, shared, name)
    if kind == "multi_agent":
        return Orchestrator(cfg, llm, shared, name).run
    raise ValueError(f"unknown system kind {kind!r}; expected one of {SYSTEM_KINDS}")
