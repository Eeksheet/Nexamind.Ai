import json

import pytest

from src.llm.backend import LLMBackend, Usage, estimate_cost, parse_json_object, pricing_known
from src.llm.providers import MockProvider, ProviderResponse
from src.utils.config import LLMConfig


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"answer": "x", "n": [1,2]} thanks') == {"answer": "x", "n": [1, 2]}
    assert parse_json_object("no json here") is None


def test_usage_accounting_and_cost():
    u = Usage(calls=1, input_tokens=100, output_tokens=50, cost_usd=0.01)
    v = Usage(calls=2, input_tokens=300, output_tokens=100, cost_usd=0.03)
    d = v.diff(u)
    assert d.calls == 1 and d.total_tokens == 250 and d.cost_usd == pytest.approx(0.02)
    assert estimate_cost("mock", "mock-extractive", 1000, 1000) == 0.0
    assert not pricing_known("mock", "mock-extractive")
    if pricing_known("anthropic", "claude-sonnet-4-5"):
        assert estimate_cost("anthropic", "claude-sonnet-4-5", 1_000_000, 0) > 0


def test_backend_auto_selects_mock_without_keys():
    be = LLMBackend.from_config(LLMConfig(provider="auto"), env={})
    assert be.provider_name == "mock" and not be.live
    r = be.complete("sys", "QUESTION: who?\nEVIDENCE:\n[0] A: b", role="reasoner")
    assert r.json() is not None
    assert be.usage.calls == 1 and be.usage.total_tokens > 0


def test_backend_missing_key_is_explicit():
    with pytest.raises(RuntimeError):
        LLMBackend.from_config(LLMConfig(provider="anthropic"), env={})


def test_backend_records_parse_failures_and_errors(monkeypatch):
    class Flaky:
        name, model = "mock", "flaky"
        n = 0

        def complete(self, system, user, role, max_tokens, temperature):
            self.n += 1
            if self.n == 1:
                raise ConnectionError("boom")
            return ProviderResponse("not json", 5, 2)

    be = LLMBackend(Flaky(), LLMConfig(provider="mock", max_retries=2, cache=False))
    r = be.complete("s", "u", role="generic")
    assert r.json() is None
    assert be.usage.errors >= 1 and be.usage.calls == 1


def test_mock_provider_roles_emit_json():
    m = MockProvider()
    for role, user in [("planner", "QUESTION: In what year was the university where X studied founded?"),
                       ("critic", "QUESTION: q\nCANDIDATE ANSWER: 1900\nEVIDENCE:\n[0] T: founded in 1900"),
                       ("rewrite", "SUB-QUESTION: x\nKNOWN ENTITIES: A, B\nCRITIC FEEDBACK: (none)")]:
        out = json.loads(m.complete("s", user, role, 100, 0.0).text)
        assert isinstance(out, dict) and out
