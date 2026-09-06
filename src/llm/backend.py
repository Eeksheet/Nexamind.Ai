"""LLM backend: one interface, several providers, full usage accounting.

::

    backend = LLMBackend.from_config(cfg.llm)
    resp = backend.complete(system, user, role="planner")
    resp.text, resp.json(), resp.usage.input_tokens, backend.usage.cost_usd
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..utils.config import CONFIG_DIR, LLMConfig, resolve_path
from ..utils.logging import get_logger
from .providers import AnthropicProvider, GeminiProvider, MockProvider, OpenAIProvider, Provider, ProviderResponse

log = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from ``text`` (tolerates code fences / prose)."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    for candidate in (t, *(m.group(0) for m in _JSON_RE.finditer(t))):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cache_hits: int = 0
    errors: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        self.latency_s += other.latency_s
        self.cache_hits += other.cache_hits
        self.errors += other.errors

    def diff(self, earlier: "Usage") -> "Usage":
        return Usage(self.calls - earlier.calls, self.input_tokens - earlier.input_tokens,
                     self.output_tokens - earlier.output_tokens, self.cost_usd - earlier.cost_usd,
                     self.latency_s - earlier.latency_s, self.cache_hits - earlier.cache_hits,
                     self.errors - earlier.errors)

    def snapshot(self) -> "Usage":
        return Usage(**self.__dict__)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["total_tokens"] = self.total_tokens
        d["cost_usd"] = round(self.cost_usd, 6)
        d["latency_s"] = round(self.latency_s, 4)
        return d


@dataclass
class LLMResponse:
    text: str
    role: str
    usage: Usage
    cached: bool = False
    raw: Optional[Dict[str, Any]] = None

    def json(self) -> Optional[Dict[str, Any]]:
        return parse_json_object(self.text)


def _load_pricing() -> Dict[str, Dict[str, Dict[str, float]]]:
    p = CONFIG_DIR / "models.yaml"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for prov, spec in (d.get("providers") or {}).items():
        out[prov] = spec.get("models") or {}
    return out


PRICING = _load_pricing()


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """USD estimate from ``configs/models.yaml`` list prices (0 when unknown)."""
    price = PRICING.get(provider, {}).get(model)
    if not price:
        return 0.0
    return input_tokens / 1e6 * price["input_per_m"] + output_tokens / 1e6 * price["output_per_m"]


def pricing_known(provider: str, model: str) -> bool:
    """True only for live providers with an entry in configs/models.yaml (mock cost is not a price)."""
    return provider != "mock" and model in PRICING.get(provider, {})


class DiskCache:
    """JSON-lines cache keyed by sha256(provider, model, system, user, temperature)."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._mem: Dict[str, Dict[str, Any]] = {}
        if path and path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        self._mem[rec["key"]] = rec
                    except (json.JSONDecodeError, KeyError):
                        continue

    @staticmethod
    def key(**parts: Any) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def get(self, k: str) -> Optional[Dict[str, Any]]:
        return self._mem.get(k)

    def put(self, k: str, rec: Dict[str, Any]) -> None:
        rec = dict(rec, key=k)
        self._mem[k] = rec
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")


class LLMBackend:
    """Provider-agnostic chat completion with caching, retries and usage tracking."""

    def __init__(self, provider: Provider, cfg: LLMConfig, cache: Optional[DiskCache] = None) -> None:
        self.provider = provider
        self.cfg = cfg
        self.cache = cache
        self.usage = Usage()
        self.per_role: Dict[str, Usage] = {}

    # ------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg: LLMConfig, env: Optional[Dict[str, str]] = None) -> "LLMBackend":
        env = dict(os.environ if env is None else env)
        try:
            from dotenv import dotenv_values

            env = {**dotenv_values(resolve_path(".env")), **env} if resolve_path(".env").exists() else env
        except ImportError:  # pragma: no cover
            pass
        provider_name = env.get("LLM_PROVIDER") or cfg.provider
        model = env.get("LLM_MODEL") or cfg.model
        if provider_name == "gemini":
            model = env.get("GEMINI_MODEL") or model or "gemini-3.8-flash"
        if provider_name == "auto":
            if env.get("ANTHROPIC_API_KEY"):
                provider_name = "anthropic"
            elif env.get("OPENAI_API_KEY"):
                provider_name = "openai"
            else:
                provider_name = "mock"
        if provider_name == "mock":
            provider: Provider = MockProvider()
        elif provider_name == "anthropic":
            key = env.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set (see .env.example)")
            provider = AnthropicProvider(key, model, cfg.timeout_s, cfg.base_url)
        elif provider_name == "gemini":
            key = env.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY not set (see .env.example)")
            provider = GeminiProvider(key, model, cfg.timeout_s)
        elif provider_name == "openai":
            key = env.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set (see .env.example)")
            provider = OpenAIProvider(key, model, cfg.timeout_s,
                                      cfg.base_url or env.get("OPENAI_BASE_URL") or None)
        else:
            raise ValueError(f"unknown LLM provider {provider_name!r}")
        cache = None
        if cfg.cache and provider_name != "mock":
            cache = DiskCache(resolve_path(cfg.cache_dir) / f"{provider_name}_{model.replace('/', '_')}.jsonl")
        be = cls(provider, cfg, cache)
        if provider_name != "mock" and not pricing_known(provider_name, model):
            log.warning("no pricing entry for %s/%s in configs/models.yaml – cost will be 0 (unknown)",
                        provider_name, model)
        log.info("LLM backend: provider=%s model=%s cache=%s", provider_name, be.model, bool(cache))
        return be

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def provider_name(self) -> str:
        return self.provider.name

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def pricing_known(self) -> bool:
        return self.provider.name != "mock" and pricing_known(self.provider.name, self.model)

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def live(self) -> bool:
        return self.provider.name != "mock"

    # ------------------------------------------------------------------- call
    def complete(self, system: str, user: str, role: str = "generic",
                 max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> LLMResponse:
        max_tokens = self.cfg.max_tokens if max_tokens is None else max_tokens
        temperature = self.cfg.temperature if temperature is None else temperature
        key = DiskCache.key(provider=self.name, model=self.model, system=system, user=user,
                            temperature=temperature, max_tokens=max_tokens)
        t0 = time.perf_counter()
        cached = False
        if self.cache is not None and (rec := self.cache.get(key)) is not None:
            pr = ProviderResponse(text=rec["text"], input_tokens=rec["input_tokens"],
                                  output_tokens=rec["output_tokens"], raw=None)
            cached = True
        else:
            pr = self._call_with_retries(system, user, role, max_tokens, temperature)
            if self.cache is not None:
                self.cache.put(key, dict(text=pr.text, input_tokens=pr.input_tokens,
                                         output_tokens=pr.output_tokens, role=role))
        latency = time.perf_counter() - t0
        u = Usage(calls=1, input_tokens=pr.input_tokens, output_tokens=pr.output_tokens,
                  cost_usd=estimate_cost(self.name, self.model, pr.input_tokens, pr.output_tokens),
                  latency_s=latency, cache_hits=int(cached))
        self.usage.add(u)
        self.per_role.setdefault(role, Usage()).add(u)
        return LLMResponse(text=pr.text, role=role, usage=u, cached=cached, raw=pr.raw)

    def _call_with_retries(self, system: str, user: str, role: str, max_tokens: int,
                           temperature: float) -> ProviderResponse:
        last: Optional[Exception] = None
        for attempt in range(max(1, self.cfg.max_retries)):
            try:
                return self.provider.complete(system, user, role=role, max_tokens=max_tokens,
                                              temperature=temperature)
            except Exception as exc:  # network / rate limit
                last = exc
                self.usage.errors += 1
                wait = min(2 ** attempt, 20)
                log.warning("LLM call failed (%s), retry %d in %ss", exc, attempt + 1, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.cfg.max_retries} attempts: {last}")

    def reset_usage(self) -> None:
        self.usage = Usage()
        self.per_role = {}
