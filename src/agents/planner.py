"""Planner agent: query analysis + decomposition into ordered sub-questions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .base import Agent

COMPLEXITIES = ("simple", "moderate", "complex")
REASONING_TYPES = ("single_hop", "bridge", "comparison", "intersection", "other")


@dataclass
class Plan:
    complexity: str
    reasoning_type: str
    hops: int
    subquestions: List[str]
    dependencies: List[Tuple[int, int]] = field(default_factory=list)
    source: str = "llm"  # llm | fallback | disabled

    def to_dict(self) -> Dict[str, Any]:
        return dict(complexity=self.complexity, reasoning_type=self.reasoning_type, hops=self.hops,
                    subquestions=self.subquestions, dependencies=[list(d) for d in self.dependencies],
                    source=self.source)

    @classmethod
    def trivial(cls, question: str, source: str = "disabled") -> "Plan":
        return cls("simple", "single_hop", 1, [question], [], source)


class PlannerAgent(Agent):
    """Decides 1-hop / 2-hop / multi-hop and produces retrieval sub-questions.

    Unlike the prototype, the planner's output is used as-is (no forced 3 hops,
    no silent prepending of the full question).  Validation only clamps values
    into the allowed ranges; a non-parsable reply falls back to a trivial plan.
    """

    ROLE = "planner"

    def __init__(self, llm, max_hops: int = 3) -> None:
        super().__init__(llm)
        self.max_hops = max_hops

    def plan(self, question: str) -> Plan:
        obj = self._ask("planner", question=question, max_tokens=300)
        subs = [str(s).strip() for s in obj.get("subquestions", []) if str(s).strip()]
        if not subs:
            return Plan.trivial(question, source="fallback")
        hops = obj.get("hops", len(subs))
        try:
            hops = int(hops)
        except (TypeError, ValueError):
            hops = len(subs)
        hops = max(1, min(self.max_hops, hops, len(subs)))
        subs = subs[:hops]
        deps: List[Tuple[int, int]] = []
        for d in obj.get("dependencies", []) or []:
            if isinstance(d, (list, tuple)) and len(d) == 2:
                try:
                    i, j = int(d[0]), int(d[1])
                except (TypeError, ValueError):
                    continue
                if 0 <= i < hops and 0 <= j < hops and i != j:
                    deps.append((i, j))
        complexity = obj.get("complexity", "moderate")
        rtype = obj.get("reasoning_type", "other")
        return Plan(complexity if complexity in COMPLEXITIES else "moderate",
                    rtype if rtype in REASONING_TYPES else "other", hops, subs, deps, "llm")
