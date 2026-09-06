# API

Run: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000` (env: `MARAG_CONFIG=<yaml>` to override the config file).

## `GET /health`
```json
{"status": "ok", "provider": "mock", "model": "mock-extractive", "is_mock": true,
 "dataset": {"kind": "real", "name": "hotpotqa", "n": 20}}
```

## `GET /examples?n=5`
Sample dataset questions: `[{"qid", "question", "answer", "type", "level"}]`.

## `POST /query`

Request:
```json
{
  "question": "string (≥3 chars)",
  "config": {                                 // optional, same keys as configs/default.yaml + "system"
    "system": "multi_agent | single_pass | single_agent_iterative | closed_book",
    "retrieval": {"method": "hybrid", "top_k": 4, "alpha": 0.6, "reranker": false},
    "orchestrator": {"use_planner": true, "use_critic": true, "budget": {"max_hops": 3}}
  },
  "paragraphs": {"Title": ["sentence", "..."]},   // optional explicit corpus
  "qid": "5a8b57f25542995d1e6f1371"              // optional: use a dataset question's context
}
```
Unknown config keys → `400`; unknown `qid` → `404`; invalid body → `422`.

Response:
```json
{
  "answer": "...", "confidence": 0.8,
  "evidence": [{"title", "sent_idx", "sentence", "score", "method", "hop", "query", "rank", "is_gold"}],
  "trace":    [{"hop", "agent", "action", "...payload"}],
  "metrics":  {"hops", "llm_calls", "retrieval_calls", "input_tokens", "output_tokens", "total_tokens",
               "cost_usd", "cost_known", "latency_s"},
  "plan": {...}, "critic": {...}, "termination": "critic_approved",
  "system": "multi_agent",
  "backend": {"provider", "model", "is_mock", "prompt_version"},
  "corpus":  {"source": "request | dataset:<qid> | demo", "n_paragraphs", "n_sentences"},
  "error": null
}
```

## UI
`GET /` serves `src/api/static/index.html`: choose a dataset question or type your own,
pick system / retrieval / hops / top-k, toggle planner and critic, and inspect the answer,
evidence, plan, critic verdict and full trace. A banner is shown when the backend is mock.
