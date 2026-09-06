#!/usr/bin/env python
"""Run all baselines (configs/experiments.yaml: baselines) over the configured seeds."""
import _bootstrap  # noqa: F401

from src.experiments.figures import standard_figures
from src.experiments.runner import common_parser, config_from_args, load_grid, run_grid


def main() -> None:
    args = common_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    gr = run_grid(cfg, load_grid("baselines"), "baselines", only=args.only.split(",") if args.only else None)
    standard_figures(gr, gr.out_dir / "figures")


if __name__ == "__main__":
    main()
