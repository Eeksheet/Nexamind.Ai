"""Conversation orchestration for Nexamind.Ai."""
from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Dict, List, Optional

from ..agents.reasoner import ReasonerAgent
from ..datasets.hotpotqa import Dataset, Example
from ..llm.backend import LLMBackend, parse_json_object
from ..orchestration.systems import build_system
from ..retrieval.hybrid import SharedRetrievalResources
from ..routing.question_router import Route, RouteDecision, route_question
from ..web.fetch import fetch_text
from ..web.search import SearchProvider, SearchResult
from ..web.sources import format_web_evidence, normalize_sources
from .memory import ConversationMemory


GENERIC_SYSTEM = """You are Nexamind.Ai, a helpful general-purpose AI assistant. Answer clearly and accurately. Use the supplied conversation and evidence when present. For web-grounded questions, rely on the supplied web evidence and do not invent facts or sources. Do not reveal hidden reasoning. Return JSON only: {\"answer\": \"...\", \"confidence\": 0.0, \"used_source_indexes\": [0]}."""


@dataclass
class ChatResult:
    answer: str
    confidence: float
    sources: List[Dict[str, str]]
    route: str
    conversation_id: str
    backend: Dict[str, Any]
    metrics: Dict[str, Any]
    error: Optional[str] = None


class ChatService:
    def __init__(self, llm: LLMBackend, *, web_provider: Optional[SearchProvider], memory: ConversationMemory,
                 rag_service: Any = None) -> None:
        self.llm = llm
        self.web = web_provider
        self.memory = memory
        self.rag_service = rag_service

    def chat(self, message: str, conversation_id: Optional[str] = None,
             web_enabled: bool = True, rag_enabled: bool = True) -> ChatResult:
        started = time.perf_counter()
        conversation = self.memory.get_or_create(conversation_id)
        decision = route_question(message, web_enabled=web_enabled, rag_enabled=rag_enabled)
        history = self.memory.history(conversation.id)
        self.memory.add(conversation.id, "user", message)
        sources: List[Dict[str, str]] = []
        evidence_text = ""
        route = decision.route

        try:
            if route in (Route.WEB_SEARCH, Route.HYBRID) and self.web is not None:
                results = self.web.search(message, max_results=5)
                sources = normalize_sources(results, 5)
                if sources:
                    fetched = self._fetch_sources(sources[:3])
                    evidence_text = format_web_evidence(sources, fetched)
                elif route == Route.WEB_SEARCH:
                    route = Route.LLM_ONLY
            elif route == Route.WEB_SEARCH:
                route = Route.LLM_ONLY

            if route in (Route.RAG, Route.HYBRID) and self.rag_service is not None:
                rag_text = self._rag_evidence(message)
                evidence_text = (evidence_text + "\n\nKNOWLEDGE BASE EVIDENCE:\n" + rag_text).strip()
            elif route == Route.RAG and self.rag_service is None:
                route = Route.LLM_ONLY

            answer, confidence = self._generate(message, history, evidence_text, route)
            self.memory.add(conversation.id, "assistant", answer)
            return ChatResult(
                answer=answer,
                confidence=confidence,
                sources=sources,
                route=route.value,
                conversation_id=conversation.id,
                backend={"provider": self.llm.provider_name, "model": self.llm.model_name, "is_mock": not self.llm.live},
                metrics={"latency_s": round(time.perf_counter() - started, 3), "history_messages": len(history),
                         "web_sources": len(sources), "search_used": route in (Route.WEB_SEARCH, Route.HYBRID),
                         "rag_used": route in (Route.RAG, Route.HYBRID)},
            )
        except Exception as exc:
            msg = self._friendly_error(exc)
            return ChatResult("", 0.0, sources, route.value, conversation.id,
                              {"provider": self.llm.provider_name, "model": self.llm.model_name, "is_mock": not self.llm.live},
                              {"latency_s": round(time.perf_counter() - started, 3)}, msg)

    def _generate(self, question: str, history: List[Dict[str, str]], evidence: str, route: Route) -> tuple[str, float]:
        history_text = "\n".join(f"{m['role'].upper()}: {m['content'][:1200]}" for m in history[-8:]) or "(none)"
        prompt = f"QUESTION: {question}\nROUTE: {route.value}\nCONVERSATION:\n{history_text}\nEVIDENCE:\n{evidence or '(none)'}"
        if self.llm.live:
            response = self.llm.complete(GENERIC_SYSTEM, prompt, role="generic", max_tokens=min(self.llm.cfg.max_tokens, 700), temperature=0.2)
            obj = response.json() or {}
            if not obj:
                try:
                    import json
                    obj = json.loads(response.text)
                except Exception:
                    obj = {}
            answer = str(obj.get("answer", "")).strip()
            if not answer:
                answer = response.text.strip()
            try:
                confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.7))))
            except (TypeError, ValueError):
                confidence = 0.7
            return answer, confidence
        # Offline/dev fallback: use the existing extractive reasoner when evidence exists.
        if evidence:
            lines = []
            for block in evidence.split("\n\n"):
                for line in block.splitlines():
                    if line.startswith("SNIPPET:") or line.startswith("PAGE TEXT:") or line.startswith("KNOWLEDGE BASE EVIDENCE:"):
                        lines.append(line.split(":", 1)[1].strip())
            text = " ".join(lines[:3]).strip()
            return text[:1200] or "I don't have enough evidence to answer that reliably.", 0.35
        return "I can answer this when a live LLM provider is configured. Add OPENAI_API_KEY and set OPENAI_BASE_URL to your OpenRouter endpoint.", 0.1

    @staticmethod
    def _fetch_sources(sources: List[Dict[str, str]]) -> Dict[str, str]:
        fetched: Dict[str, str] = {}
        for s in sources:
            try:
                text = fetch_text(s["url"])
                if text:
                    fetched[s["url"]] = text
            except Exception:
                continue
        return fetched

    def _rag_evidence(self, question: str) -> str:
        if not self.rag_service:
            return "(knowledge base unavailable)"
        tool = self.rag_service() if callable(self.rag_service) else self.rag_service
        return tool.evidence_text(question)

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        s = str(exc).lower()
        if "429" in s or "rate" in s and "limit" in s:
            return "Nexamind.Ai is temporarily rate-limited by the AI provider. Please try again shortly."
        if "timeout" in s or "timed out" in s:
            return "The request timed out. Please try again."
        if "api_key" in s or "not set" in s:
            return "Nexamind.Ai needs an AI provider API key to answer this question."
        return "Something went wrong while generating the answer. Please try again."
