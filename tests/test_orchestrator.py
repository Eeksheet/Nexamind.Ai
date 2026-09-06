import dataclasses

import pytest

from src.agents.critic import CriticVerdict
from src.agents.reasoner import ReasonerOutput
from src.llm.backend import Usage
from src.orchestration.orchestrator import Orchestrator
from src.orchestration.policies import check_budget, decide_after_critic
from src.orchestration.state import OrchestratorState, RunResult, Termination
from src.orchestration.systems import SYSTEM_KINDS, build_system


def test_full_run_produces_state_and_trace(cfg, llm, example):
    r = Orchestrator(cfg, llm).run(example)
    assert isinstance(r, RunResult) and r.qid == example.qid and r.error is None
    assert r.termination in {t.value for t in Termination}
    agents = {t["agent"] for t in r.trace}
    assert {"planner", "retriever", "reasoner", "critic", "orchestrator"} <= agents
    assert r.hops >= 1 and r.llm_calls >= 3 and r.retrieval_calls == r.hops
    assert r.total_tokens == r.input_tokens + r.output_tokens > 0
    assert r.evidence and all("is_gold" in e for e in r.evidence)
    assert r.plan["subquestions"]


def test_ablation_flags_change_behaviour(cfg, llm, example):
    no_planner = cfg.copy(orchestrator={"use_planner": False})
    r = Orchestrator(no_planner, llm).run(example)
    assert r.plan["source"] == "disabled" and r.plan["subquestions"] == [example.question]
    no_critic = cfg.copy(orchestrator={"use_critic": False})
    r2 = Orchestrator(no_critic, llm).run(example)
    assert r2.critic["reason"] == "critic disabled"
    assert all(t["agent"] != "critic" or t.get("reason") == "critic disabled" for t in r2.trace)
    single = cfg.copy(orchestrator={"iterative": False})
    r3 = Orchestrator(single, llm).run(example)
    assert r3.hops == 1 and r3.termination == Termination.NOT_ITERATIVE.value


def test_max_hops_budget_is_respected(cfg, llm, synthetic):
    hard = cfg.copy(orchestrator={"use_critic": False, "budget": {"max_hops": 2, "confidence_threshold": 1.01}})
    for ex in synthetic.examples[:4]:
        r = Orchestrator(hard, llm).run(ex)
        assert r.hops <= 2
        assert r.termination in (Termination.MAX_HOPS.value, Termination.NO_NEW_EVIDENCE.value,
                                 Termination.REPEATED_RETRIEVAL.value)


def test_llm_call_budget_terminates(cfg, llm, example):
    tight = cfg.copy(orchestrator={"use_critic": False,
                                   "budget": {"max_llm_calls": 2, "max_hops": 5, "confidence_threshold": 1.01}})
    r = Orchestrator(tight, llm).run(example)
    assert r.termination == Termination.BUDGET_EXCEEDED.value
    assert r.llm_calls <= 4  # the hop in flight may finish, but no further hop is started
    assert any(t["action"] == "budget" for t in r.trace)


def test_budget_policy_functions():
    from src.utils.config import BudgetConfig
    st = OrchestratorState(question="q", usage_start=Usage())
    b = BudgetConfig(max_llm_calls=2, max_retrieval_calls=10, max_tokens=10_000, max_latency_s=100)
    assert not check_budget(st, b, Usage(calls=1)).exceeded
    assert check_budget(st, b, Usage(calls=2)).exceeded
    st.retrieval_calls = 10
    assert "retrieval" in check_budget(st, b, Usage()).reason
    st.retrieval_calls = 0
    assert check_budget(st, dataclasses.replace(b, max_cost_usd=0.01), Usage(cost_usd=0.02)).exceeded


def test_decide_after_critic_ordering():
    from src.utils.config import BudgetConfig
    b = BudgetConfig(max_hops=3, confidence_threshold=0.9)
    st = OrchestratorState(question="q", usage_start=Usage())
    st.hop = 1
    st.candidate = ReasonerOutput("a", "s", [0], 0.5, [("T", 0)])
    st.critic = CriticVerdict(True, 0.95, [], [], True, "STOP", "ok")
    assert decide_after_critic(st, b, Usage(), iterative=True) == Termination.CRITIC_APPROVED
    assert decide_after_critic(st, b, Usage(), iterative=False) == Termination.NOT_ITERATIVE
    st.critic = CriticVerdict(False, 0.2, ["x"], [], True, "RETRIEVE", "thin")
    assert decide_after_critic(st, b, Usage(), iterative=True) is None
    st.evidence.per_hop_new = [3, 0]
    assert decide_after_critic(st, b, Usage(), iterative=True) == Termination.NO_NEW_EVIDENCE
    st.evidence.per_hop_new = [3]
    st.hop = 3
    assert decide_after_critic(st, b, Usage(), iterative=True) == Termination.MAX_HOPS
    st.hop = 1
    st.critic = CriticVerdict.disabled()
    st.candidate = ReasonerOutput("a", "s", [0], 0.95, [("T", 0)])
    assert decide_after_critic(st, b, Usage(), iterative=True) == Termination.CONFIDENCE_THRESHOLD


def test_errors_are_captured_not_raised(cfg, llm, example):
    broken = dataclasses.replace(example, paragraphs={})
    r = Orchestrator(cfg, llm).run(broken)
    assert r.termination == Termination.ERROR.value and r.error


@pytest.mark.parametrize("kind", SYSTEM_KINDS)
def test_all_system_kinds_return_uniform_results(kind, cfg, llm, example):
    r = build_system(kind, cfg, llm)(example)
    assert isinstance(r, RunResult) and r.system == kind and r.error is None
    assert r.termination != "unknown" and r.answer is not None
    if kind == "closed_book":
        assert r.retrieval_calls == 0 and r.evidence == []
    else:
        assert r.retrieval_calls >= 1 and r.evidence
