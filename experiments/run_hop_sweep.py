#!/usr/bin/env python
"""Sweep the hop budget of the full system: accuracy and cost vs max_hops."""
import json

import _bootstrap  # noqa: F401

from src.evaluation.statistics import mean_std
from src.experiments.figures import _prov, hop_curve
from src.experiments.runner import common_parser, config_from_args, load_experiments_yaml, run_grid
from src.utils.logging import get_logger

log = get_logger("hop_sweep")


def main() -> None:
    p = common_parser(__doc__)
    p.add_argument("--hops", default=None, help="comma-separated hop budgets (default from experiments.yaml)")
    args = p.parse_args()
    cfg = config_from_args(args)
    hops = [int(h) for h in args.hops.split(",")] if args.hops else load_experiments_yaml()["hop_sweep"]["max_hops"]
    grid = {f"max_hops_{h}": {"system": "multi_agent", "orchestrator": {"budget": {"max_hops": h}}} for h in hops}
    gr = run_grid(cfg, grid, "hop_sweep")
    points = {}
    for h in hops:
        bs = gr.results[f"max_hops_{h}"]
        points[h] = {"em": mean_std([r.summary["answer"]["em"] for r in bs.values()]),
                     "f1": mean_std([r.summary["answer"]["f1"] for r in bs.values()]),
                     "llm_calls": mean_std([r.summary["cost"]["llm_calls"] for r in bs.values()]),
                     "hops": mean_std([r.summary["cost"]["hops"] for r in bs.values()])}
    (gr.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (gr.out_dir / "tables" / "hop_sweep.json").write_text(json.dumps(points, indent=2))
    hop_curve(points, gr.out_dir / "figures" / "hop_sweep.png", _prov(gr))
    log.info("hop sweep done → %s", gr.out_dir)


if __name__ == "__main__":
    main()
