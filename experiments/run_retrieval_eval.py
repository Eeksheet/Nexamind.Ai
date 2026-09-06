#!/usr/bin/env python
"""LLM-free retrieval evaluation: BM25 / TF-IDF / dense / hybrid / hybrid+reranker,
alpha sweep and top-k table on the supporting-fact retrieval task."""
import json

import _bootstrap  # noqa: F401

from src.datasets.hotpotqa import DatasetKind, load_dataset
from src.evaluation.retrieval_eval import KS, alpha_sweep, compare_methods, topk_table
from src.experiments.figures import alpha_curve, retrieval_recall_curves
from src.experiments.runner import common_parser, config_from_args, load_experiments_yaml
from src.retrieval.hybrid import SharedRetrievalResources
from src.utils.config import resolve_path
from src.utils.logging import get_logger

log = get_logger("retrieval_eval")


def main() -> None:
    p = common_parser(__doc__)
    p.add_argument("--no-dense", action="store_true", help="skip dense/hybrid/reranker (no model download)")
    args = p.parse_args()
    cfg = config_from_args(args)
    ds = load_dataset(cfg.dataset)
    out = resolve_path(cfg.experiment.results_dir) / "retrieval"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    prov = f"{ds.name} ({ds.kind.value}, n={len(ds.examples)}) · {cfg.retrieval.embedding_model}"
    if ds.kind != DatasetKind.REAL:
        prov = "NOT A BENCHMARK RESULT — " + prov
    shared = SharedRetrievalResources(cfg.retrieval)
    methods = ("bm25", "tfidf") if args.no_dense else ("bm25", "tfidf", "dense", "hybrid")
    res = compare_methods(ds, cfg.retrieval, shared, methods=methods, include_rerank=not args.no_dense)
    sweep = {} if args.no_dense else alpha_sweep(ds, cfg.retrieval, load_experiments_yaml()["alpha_sweep"], shared)
    payload = {"dataset": {"name": ds.name, "kind": ds.kind.value, "n": len(ds.examples), "version": ds.version},
               "embedding_model": cfg.retrieval.embedding_model, "reranker_model": cfg.retrieval.reranker_model,
               "methods": res, "alpha_sweep": {str(k): v for k, v in sweep.items()}, "topk_table": topk_table(res)}
    if ds.kind != DatasetKind.REAL:
        payload["WARNING"] = "SYNTHETIC DATA – not HotpotQA benchmark results"
    (out / "tables" / "retrieval_metrics.json").write_text(json.dumps(payload, indent=2))
    header = "| method | " + " | ".join(f"R@{k}" for k in KS) + " | MRR | nDCG@10 | SP-P@5 |"
    md = [f"# Retrieval evaluation\n\n{prov}\n", header,
          "|---|" + "---|" * (len(KS) + 3)]
    for m, r in res.items():
        md.append(f"| {m} | " + " | ".join(f"{r[f'recall@{k}'] * 100:.1f}" for k in KS)
                  + f" | {r['mrr'] * 100:.1f} | {r['ndcg@10'] * 100:.1f} | {r['precision@5'] * 100:.1f} |")
    if sweep:
        md += ["", "## Alpha sweep (hybrid)", "", "| alpha | R@5 | R@10 | MRR |", "|---|---|---|---|"]
        for a, r in sweep.items():
            md.append(f"| {a:.1f} | {r['recall@5'] * 100:.1f} | {r['recall@10'] * 100:.1f} | {r['mrr'] * 100:.1f} |")
    (out / "tables" / "retrieval.md").write_text("\n".join(md) + "\n")
    retrieval_recall_curves(res, out / "figures" / "recall_at_k.png", prov)
    if sweep:
        alpha_curve(sweep, out / "figures" / "alpha_sweep.png", prov)
    log.info("retrieval eval done → %s", out)
    print("\n".join(md))


if __name__ == "__main__":
    main()
