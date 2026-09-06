"""Common agent plumbing: structured LLM call with JSON validation and fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..llm.backend import LLMBackend, LLMResponse
from ..llm.prompts import PROMPTS
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class AgentCall:
    """Record of one agent → LLM interaction (kept in the trace)."""

    role: str
    prompt_chars: int
    response: Dict[str, Any]
    parse_ok: bool
    cached: bool
    input_tokens: int
    output_tokens: int
    latency_s: float


class Agent:
    ROLE = "agent"

    def __init__(self, llm: LLMBackend) -> None:
        self.llm = llm
        self.calls: list[AgentCall] = []

    def _ask(self, prompt_key: str, max_tokens: Optional[int] = None, **fmt: Any) -> Dict[str, Any]:
        """Format prompt ``prompt_key``, call the LLM, return the parsed JSON (or ``{}``)."""
        p = PROMPTS[prompt_key]
        user = p["user"].format(**fmt)
        resp: LLMResponse = self.llm.complete(p["system"], user, role=self.ROLE, max_tokens=max_tokens)
        obj = resp.json()
        ok = obj is not None
        if not ok:
            log.warning("%s: could not parse JSON from LLM output: %r", self.ROLE, resp.text[:200])
            obj = {}
        self.calls.append(AgentCall(self.ROLE, len(p["system"]) + len(user), obj, ok, resp.cached,
                                    resp.usage.input_tokens, resp.usage.output_tokens,
                                    resp.usage.latency_s))
        return obj


def clamp01(x: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return default
