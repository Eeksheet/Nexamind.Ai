"""Central controller for the multi-agent loop.

::

    plan ──► ┌─ retrieve ─► evidence manager ─► reason ─► critic ─► decide ─┐
             └──────────────── continue / re-retrieve / replan / stop ◄──────┘
"""
from __future__ import annotations

import traceback
from typing import Optional

from ..agents.critic import CriticAgent, CriticVerdict
from ..agents.planner import Plan, PlannerAgent
from ..agents.reasoner import ReasonerAgent
from ..agents.retriever import RetrieverAgent
from ..datasets.hotpotqa import Example
from ..llm.backend import LLMBackend
from ..retrieval.hybrid import KnowledgeIndex, SharedRetrievalResources
from ..utils.config import Config
from ..utils.logging import get_logger
from .policies import check_budget, decide_after_critic, is_repeated_query, next_subquestion
from .state import OrchestratorState, RunResult, Termination

log = get_logger(__name__)


class Orchestrator:
    """Runs the planner → (retrieve → reason → critic)* loop under explicit budgets."""

    MAX_REPLANS = 1

    def __init__(self, cfg: Config, llm: LLMBackend,
                 shared: Optional[SharedRetrievalResources] = None, name: str = "multi_agent") -> None:
        self.cfg = cfg
        self.o = cfg.orchestrator
        self.budget = cfg.orchestrator.budget
        self.llm = llm
        self.shared = shared or SharedRetrievalResources(cfg.retrieval)
        self.name = name

    # ---------------------------------------------------------------- public
    def run(self, example: Example) -> RunResult:
        st = OrchestratorState(question=example.question, qid=example.qid,
                               usage_start=self.llm.usage.snapshot())
        try:
            self._run(example, st)
        except Exception as exc:  # never lose a whole experiment to one bad question
            log.error("orchestrator error on %s: %s", example.qid, exc)
            st.error = f"{type(exc).__name__}: {exc}"
            st.log(st.hop, "orchestrator", "error", detail=traceback.format_exc()[-800:])
            st.finish(Termination.ERROR, self.llm.usage)
        st.evidence.mark_gold(example.supporting_facts)
        return RunResult.from_state(st, self.name)

    # -------------------------------------------------------------- internal
    def _run(self, example: Example, st: OrchestratorState) -> None:
        index = KnowledgeIndex(example.paragraphs, self.cfg.retrieval, self.shared)
        planner = PlannerAgent(self.llm, max_hops=self.budget.max_hops)
        retriever = RetrieverAgent(self.llm, index, top_k=self.cfg.retrieval.top_k,
                                   use_query_rewriting=self.o.use_query_rewriting,
                                   use_entity_carryover=self.o.use_entity_carryover)
        reasoner = ReasonerAgent(self.llm)
        critic = CriticAgent(self.llm)
        q = example.question

        # ---- plan
        st.plan = planner.plan(q) if self.o.use_planner else Plan.trivial(q)
        st.log(0, "planner", "plan", **st.plan.to_dict())
        replans = 0

        while True:
            b = check_budget(st, self.budget, self.llm.usage)
            if b.exceeded and st.candidate is not None:
                st.log(st.hop, "orchestrator", "budget", detail=b.reason)
                st.finish(Termination.BUDGET_EXCEEDED, self.llm.usage)
                return

            # ---- retrieve
            subq = next_subquestion(st)
            feedback = st.critic.feedback if (st.critic and not st.critic.supported) else None
            req = retriever.build_query(q, subq, st.hop, st.entities, st.evidence, feedback)
            if st.hop > 0 and is_repeated_query(st, req.query):
                missing = retriever.missing_terms(q, st.evidence)
                if missing:
                    req.query = f"{req.query} {' '.join(missing[:4])}"
                    req.target = "missing_evidence"
                if is_repeated_query(st, req.query) and st.candidate is not None:
                    st.log(st.hop, "orchestrator", "repeated_query", query=req.query)
                    st.finish(Termination.REPEATED_RETRIEVAL, self.llm.usage)
                    return
            st.queries_issued.append(req.query)
            new = retriever.retrieve(req, st.evidence, hop=st.hop)
            st.retrieval_calls = retriever.retrieval_calls
            st.entities = retriever.discover_entities(st.evidence, q)
            if subq not in st.completed_subquestions:
                st.completed_subquestions.append(subq)
            st.log(st.hop, "retriever", "retrieve", **req.to_dict(), n_new=len(new),
                   n_total=len(st.evidence), evidence=[e.to_dict() for e in new],
                   carried_entities=st.entities[:5])
            st.hop += 1

            # ---- reason
            st.candidate = reasoner.answer(q, st.evidence.evidence)
            st.log(st.hop, "reasoner", "answer", **st.candidate.to_dict())

            # ---- critic
            if self.o.use_critic:
                st.critic = critic.review(q, st.candidate.answer, st.candidate.reasoning_summary,
                                          st.evidence.evidence)
            else:
                st.critic = CriticVerdict.disabled()
            st.log(st.hop, "critic", "review", decision=st.critic.recommendation, **st.critic.to_dict())

            # ---- decide
            if (st.critic.recommendation == "REPLAN" and self.o.use_planner and replans < self.MAX_REPLANS
                    and st.hop < self.budget.max_hops):
                replans += 1
                st.plan = planner.plan(f"{q}\n(Previous attempt failed: {st.critic.reason})")
                st.log(st.hop, "planner", "replan", **st.plan.to_dict())
                # restart plan traversal from the first sub-question not yet completed
                st.plan.subquestions = [s for s in st.plan.subquestions if s not in st.completed_subquestions] \
                    or st.plan.subquestions
            decision = decide_after_critic(st, self.budget, self.llm.usage, self.o.iterative)
            st.log(st.hop, "orchestrator", "decide",
                   decision=decision.value if decision else "continue")
            if decision is not None:
                st.finish(decision, self.llm.usage)
                return
