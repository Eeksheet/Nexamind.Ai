"""Lightweight intent router for Nexamind.Ai.

The router deliberately uses deterministic heuristics so simple questions do not
consume an extra LLM call. It can be replaced by a learned/classifier router later.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class Route(str, Enum):
    LLM_ONLY = "LLM_ONLY"
    WEB_SEARCH = "WEB_SEARCH"
    RAG = "RAG"
    HYBRID = "HYBRID"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str
    confidence: float


CURRENT_PATTERNS = re.compile(
    r"\b(latest|today|tonight|yesterday|tomorrow|current|currently|recent|recently|news|breaking|live|now|this week|this month|price|weather|stock|score|election|president|prime minister|ceo)\b",
    re.I,
)
RAG_PATTERNS = re.compile(
    r"\b(according to (the )?(documents|document|knowledge base|knowledgebase|corpus)|in (the )?(documents|knowledge base|corpus)|from (the )?(documents|knowledge base|corpus)|our knowledge base|available documents)\b",
    re.I,
)
HYBRID_PATTERNS = re.compile(
    r"\b(compare|contrast|difference|versus|vs\.?|against)\b.*\b(latest|current|today|online|web|internet)\b|\b(latest|current|today|online|web|internet)\b.*\b(document|knowledge base|corpus)\b",
    re.I,
)


def route_question(question: str, *, web_enabled: bool = True, rag_enabled: bool = True) -> RouteDecision:
    q = " ".join(question.strip().split())
    if not q:
        return RouteDecision(Route.LLM_ONLY, "empty question", 0.0)

    current = bool(CURRENT_PATTERNS.search(q))
    rag = bool(RAG_PATTERNS.search(q))
    hybrid = bool(HYBRID_PATTERNS.search(q))

    if hybrid and web_enabled and rag_enabled:
        return RouteDecision(Route.HYBRID, "question explicitly combines current/web and knowledge-base context", 0.95)
    if rag and rag_enabled and current and web_enabled:
        return RouteDecision(Route.HYBRID, "question requests both knowledge-base and current information", 0.9)
    if current and web_enabled:
        return RouteDecision(Route.WEB_SEARCH, "question contains time-sensitive/current-information signals", 0.9)
    if rag and rag_enabled:
        return RouteDecision(Route.RAG, "question explicitly refers to the available knowledge base", 0.95)
    if current and not web_enabled and rag_enabled:
        return RouteDecision(Route.RAG, "web search is disabled; using the available knowledge base", 0.55)
    return RouteDecision(Route.LLM_ONLY, "stable/general question with no explicit retrieval requirement", 0.85)
