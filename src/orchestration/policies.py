"""Routing and termination policies (pure functions over the state)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..llm.backend import Usage
from ..utils.config import BudgetConfig
from .state import OrchestratorState, Termination


@dataclass
class BudgetStatus:
    exceeded: bool
    reason: str = ""


def check_budget(st: OrchestratorState, budget: BudgetConfig, usage_now: Usage) -> BudgetStatus:
    """Return whether any hard budget has been exhausted."""
    used = usage_now.diff(st.usage_start)
    if used.calls >= budget.max_llm_calls:
        return BudgetStatus(True, f"llm_calls {used.calls} >= {budget.max_llm_calls}")
    if st.retrieval_calls >= budget.max_retrieval_calls:
        return BudgetStatus(True, f"retrieval_calls {st.retrieval_calls} >= {budget.max_retrieval_calls}")
    if used.total_tokens >= budget.max_tokens:
        return BudgetStatus(True, f"tokens {used.total_tokens} >= {budget.max_tokens}")
    if st.elapsed >= budget.max_latency_s:
        return BudgetStatus(True, f"latency {st.elapsed:.1f}s >= {budget.max_latency_s}s")
    if budget.max_cost_usd is not None and used.cost_usd >= budget.max_cost_usd:
        return BudgetStatus(True, f"cost ${used.cost_usd:.4f} >= ${budget.max_cost_usd}")
    return BudgetStatus(False)


def decide_after_critic(st: OrchestratorState, budget: BudgetConfig, usage_now: Usage,
                        iterative: bool) -> Optional[Termination]:
    """Decide whether to stop after a reason→critic round. ``None`` means continue."""
    verdict, cand = st.critic, st.candidate
    if not iterative:
        return Termination.NOT_ITERATIVE
    critic_active = verdict is not None and not verdict.is_disabled
    if critic_active and verdict.supported and verdict.recommendation == "STOP":
        return Termination.CRITIC_APPROVED
    if critic_active and verdict.supported and verdict.confidence >= budget.confidence_threshold:
        return Termination.CONFIDENCE_THRESHOLD
    if not critic_active and cand is not None and cand.confidence >= budget.confidence_threshold:
        return Termination.CONFIDENCE_THRESHOLD
    if st.hop >= budget.max_hops:
        return Termination.MAX_HOPS
    b = check_budget(st, budget, usage_now)
    if b.exceeded:
        st.log(st.hop, "orchestrator", "budget", detail=b.reason)
        return Termination.BUDGET_EXCEEDED
    if st.evidence.per_hop_new and st.evidence.per_hop_new[-1] == 0:
        return Termination.NO_NEW_EVIDENCE
    return None


def is_repeated_query(st: OrchestratorState, query: str) -> bool:
    return query.strip().lower() in {q.strip().lower() for q in st.queries_issued}


def next_subquestion(st: OrchestratorState) -> str:
    """Pick the sub-question for the current hop (plan order, then last one)."""
    assert st.plan is not None
    subs = st.plan.subquestions
    return subs[min(st.hop, len(subs) - 1)]
