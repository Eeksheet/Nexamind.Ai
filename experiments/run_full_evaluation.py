#!/usr/bin/env python
"""Main reproducibility entry point: retrieval eval + baselines + ablations + hop sweep,
multi-seed, with tables/figures/traces under results/<run-name>/.

    python experiments/run_full_evaluation.py --size 100 --seeds 42,43,44
"""
import subprocess
import sys
import time

import _bootstrap  # noqa: F401

from src.experiments.runner import common_parser
from src.utils.logging import get_logger

log = get_logger("full_eval")

STAGES = ("run_retrieval_eval.py", "run_baselines.py", "run_ablation.py", "run_hop_sweep.py")


def main() -> None:
    p = common_parser(__doc__)
    p.add_argument("--skip", default="", help="comma-separated stages to skip (e.g. run_hop_sweep.py)")
    args, extra = p.parse_known_args()
    skip = set(filter(None, args.skip.split(",")))
    passthrough = []
    for k, v in vars(args).items():
        if k in ("skip", "only") or v in (None, False):
            continue
        flag = "--" + k.replace("_", "-")
        passthrough += [flag] if v is True else [flag, str(v)]
    t0 = time.time()
    for stage in STAGES:
        if stage in skip:
            continue
        cmd = [sys.executable, str(_bootstrap.ROOT / "experiments" / stage), *passthrough, *extra]
        log.info(">>> %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    log.info("full evaluation finished in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
