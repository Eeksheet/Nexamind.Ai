# Experimental protocol

## Research questions

1. Does multi-agent orchestration (planner + iterative retrieval + critic) improve
   multi-hop QA over single-pass RAG and single-agent iterative RAG at matched
   retrieval settings?
2. Which components matter (planner, critic, query rewriting, entity carry-over,
   iteration, reranker)?
3. How do accuracy and cost trade off with the hop budget?
4. Which retrieval method (BM25 / dense / hybrid / hybrid+rerank) is best for
   sentence-level distractor retrieval, and which α?

## Dataset

HotpotQA distractor setting, validation split (the test split has no public labels).
Subsets are drawn with a fixed seed so every system sees the same questions per seed
(`size ∈ {smoke=20, 100, 500, 1000, full=7 404}`). One malformed row in the HF
mirror is dropped during validation.

## Systems

See README §Baselines. All systems share the retrieval settings of the run unless
the grid entry overrides them; the proposed system is `multi_agent_full`.

## Seeds and statistics

* Seeds **42, 43, 44**; each (system, seed) is a separate run with its own manifest.
* Tables report mean ± std over seeds.
* Paired comparisons vs. the reference system use aligned per-question arrays
  *within* each seed: exact McNemar for binary metrics (EM, SP-EM, joint EM),
  Wilcoxon signed-rank for continuous metrics (F1, SP-F1), with Holm–Bonferroni
  across the family of comparisons. 95 % bootstrap CIs of the mean difference are
  reported alongside. p-values are only produced from data; when two systems are
  identical the test returns p = 1.
* A reported "significant" difference requires Holm-corrected p < 0.05.

## Metrics

Answer EM/F1 (official normalisation), supporting-fact EM/P/R/F1, joint EM/F1,
Recall@{1,3,5,10}/MRR/nDCG for retrieval, LLM calls, tokens, estimated cost,
retrieval calls, latency, hops, termination reasons, failure taxonomy, EM by
question type (bridge/comparison) and difficulty (easy/medium/hard).

## Commands

```bash
# 1. retrieval only (no LLM) — genuine numbers with any backend
python experiments/run_retrieval_eval.py --size 500

# 2. full protocol
python experiments/run_full_evaluation.py --size 500 --seeds 42,43,44 --provider anthropic --model claude-sonnet-4-5

# 3. individual stages
python experiments/run_baselines.py --size 500 --seeds 42,43,44
python experiments/run_ablation.py  --size 500 --seeds 42,43,44
python experiments/run_hop_sweep.py --size 500 --seeds 42,43,44
```

Cost guard: `orchestrator.budget.max_cost_usd` bounds each question; the grid
driver logs cumulative cost after every (system, seed).

## Reporting rules

* Never report `backend=mock` or `dataset.kind=synthetic` numbers as benchmark results;
  the tables and figures carry the warning automatically.
* Report the exact `manifest.json` fields (model, prompt version, α, top-k, hops,
  seeds, dataset size, git revision) with every table.
* Cost is an estimate from `configs/models.yaml`; state the price date when quoting it.
