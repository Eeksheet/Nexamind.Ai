# Implementation Audit — `MultiAgent_RAG_Orchestration.ipynb`

This document records the state of the original prototype notebook
(preserved verbatim at `notebooks/prototype_original.ipynb`) **before** any
refactoring, so that every design decision in the new codebase can be traced
back to a concrete observation about the prototype.

---

## 1. Current architecture (as implemented in the notebook)

```
load_hotpot() ──► list[dict(qid, question, answer, paragraphs, supporting_facts, level, type)]
                       │  (synthetic fallback `_synth_multihop()` if ./data has no JSON)
                       ▼
             KnowledgeIndex(paragraphs)         per-question sentence index
             ├── BM25 (hand-written)             lexical
             └── TfidfVectorizer (1-2gram)       "dense" stand-in (actually sparse TF-IDF cosine)
                 search(q, k, alpha, exclude) →  alpha*minmax(bm25) + (1-alpha)*cosine
                       ▼
             LLMBackend  (Anthropic via urllib, or deterministic extractive stub)
                       ▼
   PlannerAgent.plan(q)          → list[str] sub-questions (heuristic when stub)
   RetrieverAgent.retrieve()     → index.search(subq + carried entities, exclude=used)
   RetrieverAgent.extract_entities()  regex capitalised-span counter
   ReasonerAgent.answer()        → llm(...) then `_shorten()` regex heuristics
   CriticAgent.review()          → purely heuristic (grounded / coverage / type check)
                       ▼
   Orchestrator.run(example)     plan once → loop(retrieve→reason→critic) ≤ max_hops
                                 stops when critic.supported
                       ▼
   Baselines: closed-book, single-agent RAG (k=4, k=8), no-critic, no-planner, full
   Evaluation: EM, F1, supporting-fact recall/precision, hops, llm_calls, latency
   Analyses: gain attribution (7.1), hop-budget sweep (7.2), trace inspection (8),
             failure taxonomy (8.1, 4 classes), CSV export (9)
```

## 2. Existing functionality (inventory)

| Cell | Component | Status |
|---|---|---|
| 2 | Seeding (`random`, `numpy`), dirs | works |
| 4 | `_synth_multihop`, `load_hotpot` | works; Kaggle-only real path, fallback silent-ish |
| 6 | `tokenize`, `BM25`, `KnowledgeIndex` | works; correct BM25 formula; O(N·|q|) python loops |
| 8 | `LLMBackend` | works for Anthropic only; no token accounting; no retries; no cache |
| 10 | `PlannerAgent`, `RetrieverAgent`, `ReasonerAgent`, `CriticAgent` | control flow works; **agents are heuristics, not LLM agents** (see §3) |
| 12 | `Orchestrator` | works; minimal state; termination = critic ok or max hops |
| 14 | `single_agent`, `no_retrieval`, `SYSTEMS` | works |
| 16 | `normalize`, `em`, `f1`, `support_scores`, `evaluate` | correct HotpotQA-style EM/F1 |
| 17 | 3-panel figure | works |
| 19 | Gain attribution split | works – genuinely useful research idea |
| 21 | Hop budget sweep | works |
| 23 | `show_trace` | works |
| 25 | `classify_failure` (4 classes) | works but too coarse |
| 27 | CSV + run_config export | works; records synthetic flag + backend |

## 3. Problems found

### 3.1 Bugs / correctness

1. **`PlannerAgent.plan` discards the LLM plan when the stub is active** but also
   *always* prepends the full question, so with a live LLM the plan becomes
   `[question, sub1, sub2]` truncated to `max_steps=3` — i.e. at most **two** LLM
   sub-questions survive regardless of what the model produced.
2. **`ReasonerAgent._shorten` overrides the LLM answer** whenever the reply is longer
   than 3 tokens, using regex heuristics tuned on the synthetic generator
   ("least frequent capitalised span", "last 4-digit year"). On real HotpotQA this
   destroys correct multi-word answers (e.g. "Chief of Protocol", "Arthur's Magazine").
3. **`CriticAgent.review` never calls the LLM.** Its `type_ok` check assumes only two
   answer types (year vs. not-year), so yes/no and numeric comparison questions
   (~6 % of HotpotQA) are systematically rejected → forced extra hops.
4. **Hop 0 sub-question is the full question always** (comment says this is deliberate),
   so the planner's first sub-question is never actually used for retrieval.
5. **`Orchestrator.run` selects `plan[min(hop, len(plan)-1)]`** – once the plan is
   exhausted it re-issues the *same* sub-question; only the growing `exclude` set
   makes later hops differ. No "no new evidence" detection.
6. **BM25 min-max normalisation**: `(b-b.min())/ptp` when all scores are 0 yields
   0/1e-9 → all zeros, fine, but when only one sentence matches the top gets 1.0 and
   the hybrid score becomes dominated by BM25 regardless of `alpha`.
7. **`evaluate()` uses `r["evidence"]` as the retrieved set for support metrics** —
   for the multi-agent system this is *cumulative* evidence across hops (k·hops
   sentences) whereas single-agent uses `k` sentences, so support-precision is not
   comparable across systems without also reporting `n_evidence` (it is reported, but
   the table does not normalise for it).
8. Live-API path uses `urllib` with no error handling, no retry, no rate-limit
   handling, and swallows the `usage` block → **no token counts, no cost**.
9. `load_hotpot(limit=300)` picks *any* JSON containing "dev" – would also load
   `hotpot_dev_fullwiki_v1.json` and treat it as distractor.

### 3.2 Research weaknesses

* **"Dense" retrieval is TF-IDF cosine**, not an embedding model. The paper claim
  "BM25 + dense hybrid" is not supported by the implementation.
* All agents degrade to regex heuristics in stub mode, and the reasoner/critic apply
  heuristics *even with a live LLM*. Results therefore measure the heuristics, not
  multi-agent LLM collaboration.
* Only one seed; no confidence intervals; no significance tests.
* No retrieval-only evaluation (Recall@k, MRR, nDCG) – retrieval quality is only
  measured indirectly through supporting-fact recall of whatever the orchestrator kept.
* No query-rewriting ablation (rewriting is entangled with entity carry-over).
* No reranker, no top-k / alpha sweep.
* Failure taxonomy has four classes; cannot distinguish bridge failure, critic
  false-positive/negative, premature termination, etc.
* No cost model (tokens / USD).
* Hop budget sweep uses `eval_set[:80]` while main table uses 150 → different splits.

### 3.3 Reproducibility problems

* Seeds set only for `random`/`numpy`; LLM sampling temperature not fixed; no
  recording of model name, prompt version, alpha, k, timestamp.
* Results land in a flat `results/` dir and are overwritten on every run.
* Kaggle download requires credentials; no automatic public download.
* `IS_SYNTH` printed but easy to miss; tables do not carry the label.

### 3.4 Hard-coded assumptions

* `alpha=0.6`, `k=4`, `max_hops=3` scattered as literals.
* Model name `claude-sonnet-4-6` hard-coded; only Anthropic supported.
* Year regex `1[6-9]\d{2}|20[0-2]\d` (breaks for 2030+, pre-1600).
* Capitalised-span entity regex (fails on lowercase entities, acronyms, non-Latin).
* STOPWORDS list hand-written.

### 3.5 Synthetic data usage

* `_synth_multihop()` – used automatically if `./data` is empty. All figures in the
  delivered notebook were produced on synthetic data (Cell 4 output).

### 3.6 Heuristic vs. genuine LLM agent

| Agent | LLM used? | Notes |
|---|---|---|
| Planner | only if live; output then partially overwritten | heuristic in stub |
| Retriever | never (rule-based query concat) | acceptable, but no rewriting strategy |
| Reasoner | yes, then heuristically post-processed | post-processing dominates |
| Critic | **never** | pure heuristic |
| Orchestrator | n/a | rule-based, fine |

## 4. What is worth preserving (and is preserved)

* Per-question sentence-level index (distractor setting) → `src/retrieval/`.
* Hand-written BM25 (verified against `rank_bm25`) → `src/retrieval/bm25.py`.
* Hybrid weighted fusion with min-max normalisation → `src/retrieval/hybrid.py`
  (with the normalisation edge-case fixed).
* Evidence exclusion (`exclude=used`) and entity carry-over → `EvidenceManager`,
  `RetrieverAgent`.
* Planner → retrieve → reason → critic loop with hop budget → `Orchestrator`.
* Deterministic stub backend so the whole pipeline runs without a key → `MockProvider`
  (with the extractive heuristics kept as the *mock*, not as post-processing of real
  LLM output).
* EM / F1 / supporting-fact recall & precision → `src/evaluation/answer_metrics.py`,
  `retrieval_metrics.py`.
* Gain attribution, hop-budget sweep, failure taxonomy, trace inspection →
  `src/evaluation/`, `experiments/`.
* The "Notes for the IEEE write-up" cell – its guidance (report cost, report
  variance, label stub/synthetic) is implemented as hard requirements in the
  evaluator (every result row carries `dataset_kind`, `backend`, `model`, `seed`).

## 5. Proposed improvements (implemented in this repository)

1. Package layout `src/` with typed dataclasses / Pydantic models.
2. `LLMProvider` abstraction: `mock`, `anthropic`, `openai` (+ any OpenAI-compatible
   base URL), token/cost accounting, disk cache, retries.
3. Real HotpotQA loader (HuggingFace parquet mirror, public, no credentials) with
   schema validation and explicit `DatasetKind.REAL | SYNTHETIC` tagging.
4. Dense retrieval via `sentence-transformers` (configurable model), embedding cache,
   optional cross-encoder reranker, alpha/top-k sweeps.
5. Standalone retrieval metrics: Recall@k, Precision@k, MRR, nDCG, SF-P/R/F1.
6. Structured JSON agents (planner / reasoner / critic) with prompt versioning;
   heuristic fallbacks only inside the mock provider.
7. Explicit `OrchestratorState`, budgets (hops, calls, tokens, latency, USD), six
   termination reasons, full JSON traces.
8. Eleven baselines + ablation grid + hop sweep + multi-seed (42/43/44) with mean ± std,
   bootstrap CIs and paired tests (Wilcoxon signed-rank / paired permutation).
9. Thirteen-class failure taxonomy with representative examples.
10. FastAPI `/query` + static HTML demo UI.
11. `pytest` suite, five research notebooks importing from `src/`, docs.

## 6. Research risks

* **Mock-backend numbers are not LLM numbers.** Without an API key the pipeline runs
  end-to-end but the agents are extractive heuristics; every table/figure produced
  this way is labelled `backend=mock`. Only live-model runs support claims about
  multi-agent *reasoning*.
* **CPU-only dense retrieval** limits the evaluation size (embedding ~40 sentences ×
  10 paragraphs per question); the default config uses a small MiniLM model.
* **Multi-agent cost**: 3–8× more LLM calls than vanilla RAG. If the accuracy gain is
  small the honest conclusion is a negative result; the framework reports this.
* **Distractor setting is easier than full-wiki**; results are not comparable to
  open-retrieval papers.
* **Answer-length bias**: EM strongly penalises verbose LLM answers; prompts request
  short spans, but this remains a confound between reasoner variants.

## 7. Implementation roadmap (executed in this order)

1. Audit (this document).
2. Repository scaffold; port BM25/hybrid/metrics/agents/orchestrator with tests.
3. HotpotQA loader + validation + synthetic generator.
4. Dense/hybrid/reranker retrieval + retrieval evaluation.
5. LLM providers + prompts + structured agents.
6. Orchestrator state/policies/budgets/traces.
7. Baselines, ablations, hop sweep, multi-seed, statistics, failure analysis.
8. Figures/tables, API + UI, notebooks, documentation.
9. Smoke run (synthetic + mock), real-data run (HotpotQA dev subset), final report.
