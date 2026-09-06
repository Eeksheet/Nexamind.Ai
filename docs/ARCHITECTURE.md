# Architecture

This document describes the runtime architecture of the system. For the history
(what the prototype did and why it was changed) see `IMPLEMENTATION_AUDIT.md`.

## 1. Layers

| layer | package | responsibility |
|---|---|---|
| data | `src/datasets` | HotpotQA download/parse/validate/subset; synthetic generator; `Dataset.kind ∈ {real, synthetic}` |
| retrieval | `src/retrieval` | sentence-level `KnowledgeIndex` per question: BM25, TF-IDF, dense (sentence-transformers), α-fusion, cross-encoder rerank, caching |
| llm | `src/llm` | provider abstraction (Anthropic / OpenAI-compatible / mock), versioned prompts, JSON parsing, retries, disk cache, token/cost/latency accounting |
| agents | `src/agents` | Planner, Retriever (+EvidenceManager), Reasoner, Critic — each returns a typed dataclass |
| orchestration | `src/orchestration` | `OrchestratorState`, pure policy functions, the `Orchestrator` loop, baseline systems sharing the `RunResult` contract |
| evaluation | `src/evaluation` | answer/SP/retrieval/cost metrics, statistics, failure taxonomy, `Evaluator` (per-question rows, summaries, manifests, traces), LLM-free retrieval evaluation |
| experiments | `src/experiments`, `experiments/` | grid × seeds driver, tables (mean ± std, paired tests), figures, CLI entry points |
| api | `src/api` | FastAPI service + static UI over the same `build_system()` factory the experiments use |

Nothing in `agents/` or `orchestration/` knows about datasets or files; nothing
in `evaluation/` knows about LLM providers. Configuration (`src/utils/config.py`)
is a tree of frozen dataclasses built from `configs/default.yaml` plus overrides,
with unknown keys rejected.

## 2. Retrieval

For one HotpotQA question the corpus is its 10 distractor paragraphs split into
sentences: unit = `(title, sent_idx)`, text = `"{title}: {sentence}"` (title prefix
preserved from the prototype — it helps lexical bridging).

```
component_scores(q) = {bm25: BM25(q), dense: cos(E(q), E(units)), tfidf: cos(TFIDF)}
fused(q)            = α · minmax(bm25) + (1-α) · minmax(dense)          (hybrid)
search(q, k, exclude, rerank) = top-k of fused, excluding used units,
                               optionally re-scored by a cross-encoder over the top `rerank_candidates`
```

Min-max normalisation maps a constant score vector to zeros (the prototype's
`ptp+1e-9` made a constant vector dominate). The encoder is shared process-wide,
embeddings are cached on disk keyed by `(model, sha1(text))`, and per-index query
results are memoised, so a hop that re-issues a query costs nothing.

## 3. Agents

All agents inherit `Agent._ask(prompt_key, **fields)` which formats a versioned
prompt (`src/llm/prompts.py`, `PROMPT_VERSION`), calls the backend with a `role`
tag (for per-role cost accounting) and returns parsed JSON or `{}`. Each agent
falls back to a conservative default when the JSON is malformed and records
`parse_ok=False`.

* **Planner** → `Plan{complexity, reasoning_type, hops, subquestions, dependencies, source}`;
  hop count is clamped to `budget.max_hops`; `Plan.trivial()` is used when disabled.
* **Retriever** → `RetrievalRequest{query, target, subquestion, entities, rewritten}`
  then `retrieve()` → new `Evidence` (deduplicated through `EvidenceManager`).
  Query building: hop 0 uses the sub-question; later hops append carried entities
  and critic feedback and, when rewriting is enabled, ask the LLM for a focused
  query. `discover_entities()` prefers paragraph titles of retrieved evidence
  (the canonical bridge entities in HotpotQA), then frequent capitalised spans,
  excluding entities already in the question. `missing_terms()` finds question
  terms absent from the evidence pool to break repeated-query loops.
* **Reasoner** → `ReasonerOutput{answer, reasoning_summary, evidence_ids, confidence, supporting_units}`.
  It sees only the numbered evidence list; the prompt asks for a short summary,
  not a chain of thought. Cited ids are mapped back to units and validated.
* **Critic** → `CriticVerdict{supported, confidence, missing_evidence, contradictions,
  answer_type_ok, recommendation ∈ {STOP, RETRIEVE, REPLAN}, reason}`.
  Deterministic checks (empty/unknown answer, yes/no type, answer grounded in
  evidence) are applied on top of the LLM verdict so a live model cannot approve
  an ungrounded answer.

## 4. Orchestrator

```python
state = OrchestratorState(question, qid, usage_start=llm.usage)
plan  = planner.plan(q) if use_planner else Plan.trivial(q)
while True:
    if check_budget(state).exceeded: stop(BUDGET_EXCEEDED)
    subq    = next_subquestion(state)                       # plan order, then last
    request = retriever.build_query(q, subq, hop, entities, evidence, critic.feedback)
    if is_repeated_query(state, request.query):
        request = with_missing_terms(request) or stop(REPEATED_RETRIEVAL)
    new = retriever.retrieve(request, evidence, hop)        # excludes used units
    entities = retriever.discover_entities(evidence, q)
    candidate = reasoner.answer(q, evidence.all)
    critic    = critic.review(q, candidate, evidence.all) if use_critic else CriticVerdict.disabled()
    if critic.recommendation == "REPLAN" and replans < MAX_REPLANS: plan = planner.replan(...)
    decision = decide_after_critic(state, budget, llm.usage, iterative)
    if decision: stop(decision)
    hop += 1
```

`decide_after_critic` is a pure function with a fixed precedence:
`single_pass` (non-iterative) → `critic_approved` → `confidence_threshold`
(critic-confidence when the critic is active, reasoner-confidence when disabled)
→ `max_hops` → `budget_exceeded` → `no_new_evidence` → continue.
Exceptions inside a run are caught and returned as `termination=error` so a
single bad question cannot abort an experiment. Gold supporting facts are marked
on the returned evidence *after* the run, purely for evaluation.

## 5. Systems (baselines) — one contract

`build_system(kind, cfg, llm)` returns a callable `Example → RunResult` for
`closed_book`, `single_pass`, `single_agent_iterative`, `multi_agent`. Every
result has the same fields (answer, confidence, evidence, plan, trace, hops,
llm/retrieval calls, tokens, cost, latency, termination, critic, error), so the
evaluator, tables, figures and API are system-agnostic.

## 6. Evaluation & provenance

`Evaluator.run(system, dataset, name, seed)` seeds Python/NumPy/torch, resets
usage counters, scores every question (`score_run`), writes `per_question.jsonl`,
`traces.jsonl`, `summary.json` and `manifest.json` (seed, dataset kind/version/size,
backend/model, embedding model, retrieval settings, α, top-k, reranker, max hops,
prompt version, timestamp, git revision, platform, full config). `run_grid()`
repeats this over a config grid × seeds and writes `tables/results.md|json` with
mean ± std, paired tests vs. the reference system (exact McNemar for EM, Wilcoxon
for F1, Holm-corrected), termination histograms, failure taxonomy and EM by
question type / difficulty.

## 7. Determinism

* Seeds fix subset sampling, Python/NumPy/torch RNGs and the synthetic generator.
* LLM temperature defaults to 0 and responses are cached on disk keyed by
  `(provider, model, system prompt, user prompt, temperature)`; the mock backend is
  a pure function of its input. Live providers are still not bit-reproducible —
  the multi-seed protocol and paired tests exist precisely for this reason.
