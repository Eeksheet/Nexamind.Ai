#!/usr/bin/env python
"""Fast end-to-end smoke test: every system kind on a handful of questions, one seed.

Uses whatever backend/data is available (mock + synthetic if nothing else), so it
runs offline in seconds.  Its numbers are a control-flow check, never a result.
"""
import _bootstrap  # noqa: F401

from src.experiments.figures import standard_figures
from src.experiments.runner import common_parser, config_from_args, run_grid
from src.utils.logging import get_logger

log = get_logger("smoke")

GRID = {
    "closed_book": {"system": "closed_book"},
    "bm25_rag": {"system": "single_pass", "retrieval": {"method": "bm25"}},
    "single_agent_iterative": {"system": "single_agent_iterative", "retrieval": {"method": "bm25"}},
    "multi_agent_full": {"system": "multi_agent", "retrieval": {"method": "bm25"}},
}


def main() -> None:
    p = common_parser(__doc__)
    p.add_argument("--dense", action="store_true", help="also exercise dense/hybrid retrieval (downloads model)")
    args = p.parse_args()
    args.size = args.size or "smoke"
    args.seeds = args.seeds or "42"
    cfg = config_from_args(args)
    grid = dict(GRID)
    if args.dense:
        grid["hybrid_rerank_multi"] = {"system": "multi_agent", "retrieval": {"method": "hybrid", "reranker": True}}
    gr = run_grid(cfg, grid, "smoke", only=args.only.split(",") if args.only else None)
    figs = standard_figures(gr, gr.out_dir / "figures")
    log.info("smoke OK — tables in %s, %d figures", gr.out_dir / "tables", len(figs))
    for name, by_seed in gr.results.items():
        r = next(iter(by_seed.values()))
        assert r.n > 0 and all(row["termination"] != "unknown" for row in r.rows), name
        assert not any(row["category"] == "error" for row in r.rows), f"{name}: system raised errors"


if __name__ == "__main__":
    main()
