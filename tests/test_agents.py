from src.agents.critic import CriticAgent, CriticVerdict
from src.agents.planner import Plan, PlannerAgent
from src.agents.reasoner import ReasonerAgent
from src.agents.retriever import EvidenceManager, RetrieverAgent, extract_entities
from src.retrieval.base import Evidence
from src.retrieval.hybrid import KnowledgeIndex, SharedRetrievalResources


def _ev(title, i, text, score=1.0):
    return Evidence(title=title, sent_idx=i, text=f"{title}: {text}", sentence=text, score=score, method="bm25")


def test_planner_structured_output_and_clamp(llm):
    p = PlannerAgent(llm, max_hops=2).plan("In what year was the institution where Boris Kaltag studied founded?")
    assert isinstance(p, Plan)
    assert p.reasoning_type in ("bridge", "comparison", "single_hop", "intersection", "unknown")
    assert 1 <= p.hops <= 2 and 1 <= len(p.subquestions) <= 2
    assert all(isinstance(d, tuple) and len(d) == 2 for d in p.dependencies)
    t = Plan.trivial("q")
    assert t.subquestions == ["q"] and t.hops == 1 and t.source == "disabled"


def test_evidence_manager_dedup_and_gold():
    m = EvidenceManager()
    new = m.add([_ev("A", 0, "x"), _ev("A", 0, "x"), _ev("B", 1, "y")])
    assert len(new) == 2 and len(m) == 2 and m.per_hop_new == [2]
    assert m.add([_ev("B", 1, "y")]) == [] and m.per_hop_new == [2, 0]
    m.mark_gold([("A", 0)])
    assert [e.is_gold for e in m.evidence] == [True, False]
    assert m.titles() == ["A", "B"]


def test_entity_extraction_prefers_titles_and_skips_question_entities():
    ents = extract_entities(["Boris Kaltag: He completed doctoral study at Bexley School of Science."],
                            exclude=["Boris Kaltag"], titles=["Boris Kaltag"])
    assert ents[0] == "Bexley School of Science"
    assert "Boris Kaltag" not in ents and "He" not in ents


def test_retriever_query_building_ablations(llm, cfg, example):
    idx = KnowledgeIndex(example.paragraphs, cfg.retrieval, SharedRetrievalResources(cfg.retrieval))
    m = EvidenceManager()
    r = RetrieverAgent(llm, idx, top_k=3)
    req0 = r.build_query(example.question, example.question, 0, [], m)
    assert req0.query == example.question and req0.target == "initial"
    hits = r.retrieve(req0, m, hop=0)
    assert hits and r.retrieval_calls == 1 and all(h.hop == 0 for h in hits)
    ents = r.discover_entities(m, example.question)
    req1 = r.build_query(example.question, example.question, 1, ents, m)
    assert req1.query != example.question
    assert set(ents) & set(req1.entities) or not ents
    # ablation: no rewriting & no carry-over -> plain query, no entities
    plain = RetrieverAgent(llm, idx, top_k=3, use_query_rewriting=False, use_entity_carryover=False)
    assert plain.discover_entities(m, example.question) == []
    req = plain.build_query(example.question, example.question, 1, ["X"], m)
    assert req.entities == [] and not req.rewritten


def test_reasoner_uses_only_evidence(llm, example):
    r = ReasonerAgent(llm)
    empty = r.answer(example.question, [])
    assert empty.answer.lower() in ("unknown", "") or empty.confidence <= 0.1
    evs = [_ev(t, i, s) for t, ss in example.paragraphs.items() for i, s in enumerate(ss)]
    out = r.answer(example.question, evs)
    assert out.answer and out.parse_ok and 0 <= out.confidence <= 1
    assert all(u in {e.unit for e in evs} for u in out.supporting_units)
    assert len(out.reasoning_summary) < 400  # concise summary, not chain-of-thought


def test_critic_verdict_and_disabled(llm):
    v = CriticAgent(llm).review("In what year was X founded?", "1933", "summary",
                                [_ev("X", 0, "X was founded in 1933.")])
    assert isinstance(v, CriticVerdict) and v.recommendation in ("STOP", "RETRIEVE", "REPLAN")
    bad = CriticAgent(llm).review("In what year was X founded?", "Science", "s", [_ev("X", 0, "X School of Science")])
    assert not bad.supported and not bad.answer_type_ok and bad.recommendation != "STOP"
    d = CriticVerdict.disabled()
    assert d.supported and d.reason == "critic disabled"
