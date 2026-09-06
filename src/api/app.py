"""FastAPI application for the original research API plus Nexamind.Ai chat."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..chat.conversation import ChatResult, ChatService
from ..chat.memory import ConversationMemory
from ..datasets.hotpotqa import Dataset, Example, load_dataset
from ..llm.backend import LLMBackend
from ..orchestration.systems import SYSTEM_KINDS, build_system
from ..retrieval.hybrid import KnowledgeIndex, SharedRetrievalResources
from ..routing.question_router import route_question
from ..utils.config import Config, load_config
from ..utils.logging import get_logger
from ..web.search import get_search_provider

log = get_logger(__name__)
DEMO_CORPUS_QUESTIONS = int(os.environ.get("DEMO_CORPUS_QUESTIONS", "20"))


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    config: Dict[str, Any] = Field(default_factory=dict)
    paragraphs: Optional[Dict[str, List[str]]] = None
    qid: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    evidence: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    plan: Dict[str, Any]
    critic: Optional[Dict[str, Any]]
    termination: str
    system: str
    backend: Dict[str, Any]
    corpus: Dict[str, Any]
    error: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=12000)
    conversation_id: Optional[str] = Field(default=None, max_length=100)
    web_enabled: bool = True
    rag_enabled: bool = True


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, str]]
    conversation_id: str
    route: str
    backend: Dict[str, Any]
    metrics: Dict[str, Any]
    error: Optional[str] = None


class RAGTool:
    def __init__(self, cfg: Config, dataset: Dataset):
        self.cfg = cfg
        self.dataset = dataset
        self.shared = SharedRetrievalResources(cfg.retrieval)

    def evidence_text(self, question: str) -> str:
        paragraphs: Dict[str, Sequence[str]] = {}
        for ex in self.dataset.examples:
            paragraphs.update(ex.paragraphs)
        if not paragraphs:
            return "(knowledge base is empty)"
        idx = KnowledgeIndex(paragraphs, self.cfg.retrieval, self.shared)
        hits = idx.search(question, k=min(6, self.cfg.retrieval.top_k), rerank=False)
        if not hits:
            return "(no relevant knowledge-base evidence)"
        return "\n".join(f"[{i}] {h.title}: {h.sentence}" for i, h in enumerate(hits))


class Service:
    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or load_config(os.environ.get("MARAG_CONFIG") or None)
        self.llm = LLMBackend.from_config(self.cfg.llm)
        self._shared: Dict[str, SharedRetrievalResources] = {}
        self._dataset: Optional[Dataset] = None
        self.memory = ConversationMemory()
        self.web = get_search_provider()
        self._chat_service: Optional[ChatService] = None
        self._rag_tool: Optional[RAGTool] = None

    @property
    def dataset(self) -> Dataset:
        if self._dataset is None:
            ds_cfg = self.cfg.copy(dataset={"size": str(DEMO_CORPUS_QUESTIONS)}).dataset
            self._dataset = load_dataset(ds_cfg)
        return self._dataset

    def shared(self, cfg: Config) -> SharedRetrievalResources:
        key = repr(sorted(cfg.retrieval.__dict__.items()))
        if key not in self._shared:
            self._shared[key] = SharedRetrievalResources(cfg.retrieval)
        return self._shared[key]

    def demo_corpus(self) -> Dict[str, List[str]]:
        paras: Dict[str, List[str]] = {}
        for ex in self.dataset.examples:
            paras.update(ex.paragraphs)
        return paras

    def chat(self, req: ChatRequest) -> ChatResponse:
        if self._chat_service is None:
            self._chat_service = ChatService(self.llm, web_provider=self.web, memory=self.memory, rag_service=lambda: self.rag_tool)
        result: ChatResult = self._chat_service.chat(req.message, req.conversation_id, req.web_enabled, req.rag_enabled)
        return ChatResponse(**result.__dict__)

    @property
    def rag_tool(self) -> RAGTool:
        if self._rag_tool is None:
            self._rag_tool = RAGTool(self.cfg, self.dataset)
        return self._rag_tool

    def query(self, req: QueryRequest) -> QueryResponse:
        overrides = dict(req.config)
        system_kind = overrides.pop("system", "multi_agent")
        if system_kind not in SYSTEM_KINDS:
            raise HTTPException(400, f"unknown system {system_kind!r}; choose from {SYSTEM_KINDS}")
        try:
            cfg = self.cfg.copy(**overrides) if overrides else self.cfg
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"bad config: {exc}") from exc
        if req.paragraphs:
            paragraphs, source = req.paragraphs, "request"
        elif req.qid:
            ex = next((e for e in self.dataset.examples if e.qid == req.qid), None)
            if ex is None:
                raise HTTPException(404, f"qid {req.qid!r} not in the loaded {len(self.dataset.examples)} examples")
            paragraphs, source = ex.paragraphs, f"dataset:{req.qid}"
        else:
            paragraphs, source = self.demo_corpus(), f"demo:{self.dataset.kind.value}"
        if not paragraphs:
            raise HTTPException(400, "empty corpus")
        example = Example(qid=req.qid or "api", question=req.question, answer="", paragraphs=paragraphs, supporting_facts=[])
        system = build_system(system_kind, cfg, self.llm, self.shared(cfg))
        r = system(example)
        return QueryResponse(
            answer=r.answer, confidence=r.confidence, evidence=r.evidence, trace=r.trace, plan=r.plan,
            critic=r.critic, termination=r.termination, system=r.system, error=r.error,
            metrics={"hops": r.hops, "llm_calls": r.llm_calls, "retrieval_calls": r.retrieval_calls,
                     "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                     "total_tokens": r.total_tokens, "estimated_cost_usd": r.cost_usd, "latency_s": r.latency_s},
            backend={"provider": self.llm.provider_name, "model": self.llm.model_name, "is_mock": not self.llm.live},
            corpus={"source": source, "n_paragraphs": len(paragraphs), "n_sentences": sum(len(s) for s in paragraphs.values())})


@lru_cache(maxsize=1)
def get_service() -> Service:
    return Service()


def create_app(service: Optional[Service] = None) -> FastAPI:
    app = FastAPI(title="Nexamind.Ai", version="1.0.0",
                  description="Nexamind.Ai — a general AI assistant with web search and retrieval-augmented generation")
    svc = service

    def _svc() -> Service:
        return svc or get_service()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        s = _svc()
        return {"status": "ok", "product": "Nexamind.Ai", "provider": s.llm.provider_name, "model": s.llm.model_name,
                "web_search": s.web.name if s.web else None}

    @app.get("/examples")
    def examples(n: int = 10) -> List[Dict[str, str]]:
        s = _svc()
        return [{"qid": e.qid, "question": e.question, "answer": e.answer, "type": e.qtype, "level": e.level}
                for e in s.dataset.examples[:n]]

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        return _svc().query(req)

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        return _svc().chat(req)

    @app.get("/api/chats")
    def chats() -> List[Dict[str, str]]:
        return _svc().memory.recent()

    @app.get("/api/chats/{conversation_id}")
    def chat_history(conversation_id: str) -> Dict[str, Any]:
        history = _svc().memory.history(conversation_id)
        if not history:
            raise HTTPException(404, "conversation not found")
        return {"id": conversation_id, "messages": history}

    @app.get("/api/router")
    def router_preview(question: str, web_enabled: bool = True, rag_enabled: bool = True) -> Dict[str, Any]:
        d = route_question(question, web_enabled=web_enabled, rag_enabled=rag_enabled)
        return {"route": d.route.value, "reason": d.reason, "confidence": d.confidence}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    return app


app = create_app()
