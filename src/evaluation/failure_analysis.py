"""Failure taxonomy for multi-hop QA runs.

Categories are assigned deterministically from the run record and gold data so
that the analysis is reproducible.  A run may only receive one primary category
(first match wins, ordered from "most upstream" to "most downstream").
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

from .metrics import exact_match, f1_score, normalize_answer

Unit = Tuple[str, int]

CATEGORIES = (
    "correct",
    "error",                    # exception inside the system
    "retrieval_miss",           # <50% of supporting facts ever retrieved
    "partial_retrieval",        # some but not all supporting facts retrieved
    "budget_exhausted",         # all facts retrieved but stopped on a budget/hop limit with a wrong answer
    "reasoning_error",          # all gold facts present, answer wrong, critic approved (false accept)
    "critic_false_reject",      # answer was right at some hop but system kept going and drifted
    "answer_format",            # F1 high (>0.5) but EM 0 → span boundary / verbosity
    "unsupported_answer",       # answer text does not appear in retrieved evidence (hallucination proxy)
    "other",
)


def categorize(run: Dict, gold_answer: str, gold_units: Sequence[Unit]) -> str:
    pred = run.get("answer", "") or ""
    if exact_match(pred, gold_answer):
        return "correct"
    if run.get("error"):
        return "error"
    retrieved = {(e["title"], e["sent_idx"]) for e in run.get("evidence", [])}
    gs = set(gold_units)
    frac = len(gs & retrieved) / len(gs) if gs else 1.0
    if frac < 0.5:
        return "retrieval_miss"
    if frac < 1.0:
        return "partial_retrieval"
    f1 = f1_score(pred, gold_answer)
    if f1 > 0.5:
        return "answer_format"
    # right answer appeared at an earlier hop?
    for step in run.get("trace", []):
        if step.get("agent") == "reasoner" and exact_match(str(step.get("answer", "")), gold_answer):
            return "critic_false_reject"
    if run.get("termination") in ("max_hops", "budget_exceeded"):
        return "budget_exhausted"
    ev_text = " ".join(e.get("sentence", "") for e in run.get("evidence", [])).lower()
    if normalize_answer(pred) and normalize_answer(pred) not in normalize_answer(ev_text) \
            and pred.lower() not in ("yes", "no"):
        return "unsupported_answer"
    if run.get("termination") == "critic_approved":
        return "reasoning_error"
    return "other"


def failure_table(rows: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    """rows: per-question dicts with ``category`` (and optional ``qtype``/``level``)."""
    n = max(len(rows), 1)
    c = Counter(r["category"] for r in rows)
    out = {cat: {"count": c.get(cat, 0), "fraction": c.get(cat, 0) / n} for cat in CATEGORIES}
    return out


def breakdown(rows: Sequence[Dict], key: str, metric: str = "em") -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[float]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key, "unknown")), []).append(float(r.get(metric, 0.0)))
    return {g: {"n": len(v), metric: sum(v) / len(v)} for g, v in sorted(groups.items())}
