"""LLM providers: Anthropic, OpenAI-compatible, and a deterministic mock.

The mock is *role-aware*: it emits the same structured JSON contract as a real
model for planner / rewrite / reasoner / critic so the full orchestration loop
is testable offline.  Its heuristics are the ones from the prototype notebook,
moved here so that they never post-process real LLM output.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import httpx

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

from ..retrieval.base import content_terms, tokenize


@dataclass
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int
    raw: Optional[Dict[str, Any]] = None


class Provider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, role: str, max_tokens: int,
                 temperature: float) -> ProviderResponse: ...


# --------------------------------------------------------------------------- #
# Real providers
# --------------------------------------------------------------------------- #
class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, timeout_s: float = 90.0,
                 base_url: Optional[str] = None) -> None:
        self.model = model
        self.url = base_url or "https://api.anthropic.com/v1/messages"
        self._client = httpx.Client(timeout=timeout_s, headers={
            "content-type": "application/json", "x-api-key": api_key,
            "anthropic-version": "2023-06-01"})

    def complete(self, system: str, user: str, role: str, max_tokens: int,
                 temperature: float) -> ProviderResponse:
        body = dict(model=self.model, max_tokens=max_tokens, temperature=temperature,
                    system=system, messages=[{"role": "user", "content": user}])
        r = self._client.post(self.url, json=body)
        r.raise_for_status()
        out = r.json()
        text = "".join(b.get("text", "") for b in out.get("content", []))
        usage = out.get("usage", {})
        return ProviderResponse(text, int(usage.get("input_tokens", 0)),
                                int(usage.get("output_tokens", 0)), out)


class GeminiProvider:
    """Google Gemini API provider using the official ``google-genai`` SDK."""

    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_s: float = 90.0) -> None:
        if genai is None or genai_types is None:
            raise RuntimeError("google-genai is not installed; run: pip install -U google-genai")
        self.model = model
        self._client = genai.Client(api_key=api_key, http_options=genai_types.HttpOptions(timeout=int(timeout_s * 1000)))

    def complete(self, system: str, user: str, role: str, max_tokens: int,
                 temperature: float) -> ProviderResponse:
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=config,
        )
        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        raw = None
        try:
            import json
            raw = json.loads(text)
        except Exception:
            try:
                raw = response.model_dump(mode="json")
            except Exception:
                raw = None
        return ProviderResponse(text, input_tokens, output_tokens, raw)


class OpenAIProvider:
    """OpenAI chat-completions API (also vLLM / Ollama / Together via ``base_url``)."""

    name = "openai"

    def __init__(self, api_key: str, model: str, timeout_s: float = 90.0,
                 base_url: Optional[str] = None) -> None:
        self.model = model
        base = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self._client = httpx.Client(timeout=timeout_s, headers={
            "content-type": "application/json", "authorization": f"Bearer {api_key}"})

    def complete(self, system: str, user: str, role: str, max_tokens: int,
                 temperature: float) -> ProviderResponse:
        body = dict(model=self.model, max_tokens=max_tokens, temperature=temperature,
            response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
        r = self._client.post(self.url, json=body)
        r.raise_for_status()
        out = r.json()
        text = out["choices"][0]["message"]["content"] or ""
        usage = out.get("usage", {})
        return ProviderResponse(text, int(usage.get("prompt_tokens", 0)),
                                int(usage.get("completion_tokens", 0)), out)


# --------------------------------------------------------------------------- #
# Deterministic mock
# --------------------------------------------------------------------------- #
_CAP = r"[A-Z][a-z]+(?: [A-Z][a-z]+)*"
_YEAR = r"\b(1[0-9]\d{2}|20\d{2})\b"
_YESNO_START = re.compile(r"^(are|is|were|was|do|does|did|can|could|has|have|had)\b", re.I)


def _field(prompt: str, name: str, until: Optional[str] = None) -> str:
    m = re.search(rf"{name}:\s*(.*?)(?:\n(?:[A-Z][A-Z -]+):|\Z)" if until is None
                  else rf"{name}:\s*(.*?)(?:\n{until}:|\Z)", prompt, re.S)
    return m.group(1).strip() if m else ""


def _evidence_lines(prompt: str) -> List[str]:
    block = prompt.split("EVIDENCE:", 1)[-1] if "EVIDENCE:" in prompt else ""
    return [re.sub(r"^\[\d+\]\s*", "", ln.strip()) for ln in block.split("\n")
            if re.match(r"^\[\d+\]", ln.strip())]


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockProvider:
    """Deterministic, extractive stand-in that exercises the full control flow."""

    name = "mock"
    model = "mock-extractive"

    def complete(self, system: str, user: str, role: str, max_tokens: int,
                 temperature: float) -> ProviderResponse:
        if role == "planner":
            out = self._plan(user)
        elif role == "rewrite":
            out = self._rewrite(user)
        elif role == "reasoner":
            out = self._reason(user)
        elif role == "closed_book":
            out = {"answer": "unknown", "confidence": 0.05}
        elif role == "critic":
            out = self._critic(user)
        else:
            out = {"text": self._extractive(user)}
        text = json.dumps(out)
        return ProviderResponse(text, _approx_tokens(system + user), _approx_tokens(text))

    # ------------------------------------------------------------- heuristics
    @staticmethod
    def _entities(text: str) -> List[str]:
        ents = re.findall(_CAP, text)
        return [e for e in ents if e.lower() not in ("in", "what", "which", "who", "the", "were", "are", "is")]

    def _plan(self, user: str) -> Dict[str, Any]:
        q = _field(user, "QUESTION")
        ents = self._entities(q)
        terms = content_terms(q)
        if _YESNO_START.match(q) or " or " in q.lower():
            rtype, hops = "comparison", 2
            subs = [f"{e}" for e in ents[:2]] or [q]
        elif re.search(r"\b(where|which|who|what)\b.*\b(that|which|who|whose|where)\b", q.lower()) or len(ents) <= 1:
            rtype, hops = "bridge", 2
            head = ents[0] if ents else " ".join(terms[:3])
            tail = " ".join(terms[-4:])
            subs = [q, f"{head} {tail}"]
        else:
            rtype, hops = "single_hop", 1
            subs = [q]
        return {"complexity": "simple" if hops == 1 else "moderate", "reasoning_type": rtype,
                "hops": hops, "subquestions": subs[:hops],
                "dependencies": [[i, i + 1] for i in range(hops - 1)]}

    def _rewrite(self, user: str) -> Dict[str, Any]:
        sub = _field(user, "SUB-QUESTION") or _field(user, "QUESTION")
        ents = [e.strip() for e in _field(user, "KNOWN ENTITIES").split(",") if e.strip() and e.strip() != "(none)"]
        feedback = _field(user, "CRITIC FEEDBACK")
        target = "initial"
        if ents:
            target = "entity"
        if feedback and feedback != "(none)":
            target = "missing_evidence"
        query = (sub + " " + " ".join(ents[:3])).strip()
        return {"query": query[:200], "target": target, "missing": feedback[:100] if feedback else ""}

    def _reason(self, user: str) -> Dict[str, Any]:
        q = _field(user, "QUESTION", until="EVIDENCE")
        ev = _evidence_lines(user)
        if not ev:
            return {"answer": "unknown", "reasoning_summary": "no evidence", "evidence_ids": [],
                    "confidence": 0.05}
        qt = set(content_terms(q))
        overlap = lambda i: len(qt & set(content_terms(ev[i])))  # noqa: E731
        ranked = sorted(range(len(ev)), key=lambda i: -overlap(i))
        anchor_ids = [i for i in ranked[:2] if overlap(i) > 0] or ranked[:1]
        if _YESNO_START.match(q):
            return {"answer": "yes", "reasoning_summary": "mock heuristic for yes/no",
                    "evidence_ids": anchor_ids, "confidence": 0.3}
        # two-stage: bridge entities from the anchor sentences → second-hop sentences
        q_caps = set(re.findall(_CAP, q))
        bridge_caps = {c for i in anchor_ids for c in re.findall(_CAP, ev[i].split(": ", 1)[-1])
                       if c not in q_caps and len(c.split()) > 1}
        hop2_ids = [i for i in range(len(ev)) if i not in anchor_ids
                    and any(c in ev[i] for c in bridge_caps)]
        answer, used = self._shorten(q, ev, anchor_ids, hop2_ids)
        conf = 0.6 if hop2_ids else 0.35
        return {"answer": answer, "reasoning_summary": "extractive span linked via bridge entity",
                "evidence_ids": used, "confidence": conf}

    @staticmethod
    def _shorten(question: str, ev: List[str], anchor_ids: List[int],
                 hop2_ids: List[int]) -> Tuple[str, List[int]]:
        """Prototype's extractive span heuristic (year → number → bridge entity → rare cap span)."""
        ql = question.lower()
        q_caps = set(re.findall(_CAP, question))
        body = lambda i: ev[i].split(": ", 1)[-1]  # noqa: E731
        want_year = bool(re.search(r"\b(year|founded|born|established|when|date)\b", ql))
        want_num = bool(re.search(r"\bhow many\b", ql))
        for pool in (hop2_ids, anchor_ids, list(range(len(ev)))):
            if not pool:
                continue
            space = " ".join(body(i) for i in pool)
            if want_year:
                yrs = re.findall(_YEAR, space)
                if yrs:
                    used = [i for i in pool if re.search(_YEAR, body(i))]
                    return yrs[-1], sorted(set(anchor_ids + used))
            elif want_num:
                nums = re.findall(r"\b\d[\d,]*\b", space)
                if nums:
                    return nums[0], sorted(set(anchor_ids + pool))
            else:
                caps = [c for c in re.findall(_CAP, space) if c not in q_caps]
                if caps:
                    return Counter(caps).most_common()[-1][0], sorted(set(anchor_ids + pool))
        pool = hop2_ids or anchor_ids
        return " ".join(" ".join(body(i) for i in pool).split()[:5]), pool

    def _extractive(self, user: str) -> str:
        q = _field(user, "QUESTION", until="EVIDENCE")
        ev = _evidence_lines(user)
        qt = set(content_terms(q))
        ranked = sorted(ev, key=lambda x: -len(qt & set(content_terms(x))))
        return " ".join(ranked[:3])

    def _critic(self, user: str) -> Dict[str, Any]:
        q = _field(user, "QUESTION")
        answer = _field(user, "CANDIDATE ANSWER")
        ev = _evidence_lines(user)
        joined = " ".join(ev).lower()
        if not answer.strip() or answer.lower() == "unknown":
            return {"supported": False, "confidence": 0.0, "missing_evidence": ["any answer-bearing sentence"],
                    "contradictions": [], "answer_type_ok": False, "recommendation": "RETRIEVE",
                    "reason": "empty answer"}
        ql = q.lower()
        yesno_q = bool(_YESNO_START.match(q))
        is_yesno = answer.strip().lower() in ("yes", "no")
        grounded = is_yesno or answer.lower() in joined
        echoes = (not is_yesno) and answer.lower() in ql
        qt = set(content_terms(q))
        et = set(tokenize(joined))
        cov = len(qt & et) / max(len(qt), 1)
        hit = next((e for e in ev if answer.lower() in e.lower()), "")
        linked = bool(hit) and bool(qt & set(content_terms(hit)))
        wants_year = bool(re.search(r"\b(year|founded|born|established|when)\b", ql))
        is_year = bool(re.fullmatch(_YEAR.strip(r"\b"), answer.strip()))
        type_ok = (is_yesno if yesno_q else (is_year if wants_year else not is_year))
        conf = 0.4 * grounded + 0.3 * cov + 0.3 * (linked or is_yesno)
        ok = grounded and not echoes and cov > 0.5 and type_ok
        reason = ("ok" if ok else "ungrounded span" if not grounded else "answer echoes question"
                  if echoes else "wrong answer type" if not type_ok else "thin question coverage")
        missing = [] if ok else [f"evidence about: {' '.join(sorted(qt - et)[:4])}"] if qt - et else ["bridge sentence"]
        rec = "STOP" if ok else ("REPLAN" if not type_ok and cov < 0.3 else "RETRIEVE")
        return {"supported": bool(ok), "confidence": round(float(conf), 3), "missing_evidence": missing,
                "contradictions": [], "answer_type_ok": bool(type_ok), "recommendation": rec,
                "reason": reason}
