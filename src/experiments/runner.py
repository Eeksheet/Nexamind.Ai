"""Shared experiment driver: grid of (system config × seed) → results, tables, tests.

Every table/figure produced here is tagged with the dataset kind and LLM
backend so synthetic / mock runs can never be mistaken for benchmark results.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import yaml

from ..datasets.hotpotqa import Dataset, DatasetKind, load_dataset
from ..evaluation.evaluator import EvalResult, Evaluator
from ..evaluation.statistics import format_mean_std, holm_bonferroni, mean_std, paired_test
from ..llm.backend import LLMBackend
from ..orchestration.systems import build_system
from ..retrieval.hybrid import SharedRetrievalResources
from ..utils.config import Config, load_config, resolve_path
from ..utils.logging import get_logger

log = get_logger(__name__)

EXPERIMENTS_YAML = "configs/experiments.yaml"


# ------------------------------------------------------------------ CLI helpers
def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="extra YAML merged over configs/default.yaml")
    p.add_argument("--size", default=None, help="smoke | 100 | 500 | 1000 | full | <int>")
    p.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 42,43,44")
    p.add_argument("--provider", default=None, help="mock | anthropic | openai (default: auto)")
    p.add_argument("--model", default=None)
    p.add_argument("--retrieval", default=None, help="bm25 | dense | hybrid | tfidf")
    p.add_argument("--max-hops", type=int, default=None)
    p.add_argument("--out", default=None, help="results root (default configs.experiment.results_dir)")
    p.add_argument("--synthetic", action="store_true", help="force synthetic data (testing only)")
    p.add_argument("--no-traces", action="store_true")
    p.add_argument("--only", default=None, help="comma-separated subset of grid entries")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    ov: Dict = {}
    if args.size:
        ov.setdefault("dataset", {})["size"] = args.size
    if args.synthetic:
        ov.setdefault("dataset", {})["force_synthetic"] = True
    if args.provider:
        ov.setdefault("llm", {})["provider"] = args.provider
    if args.model:
        ov.setdefault("llm", {})["model"] = args.model
    if args.retrieval:
        ov.setdefault("retrieval", {})["method"] = args.retrieval
    if args.max_hops:
        ov.setdefault("orchestrator", {}).setdefault("budget", {})["max_hops"] = args.max_hops
    if args.seeds:
        ov.setdefault("experiment", {})["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.out:
        ov.setdefault("experiment", {})["results_dir"] = args.out
    if args.no_traces:
        ov.setdefault("experiment", {})["save_traces"] = False
    return load_config(args.config, ov)


def load_grid(section: str) -> Dict[str, Dict]:
    with open(resolve_path(EXPERIMENTS_YAML), encoding="utf-8") as fh:
        y = yaml.safe_load(fh)
    return y[section]


def load_experiments_yaml() -> Dict:
    with open(resolve_path(EXPERIMENTS_YAML), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def apply_overrides(cfg: Config, overrides: Mapping) -> Config:
    """Return a new Config with a nested override dict applied (``system`` key ignored)."""
    return cfg.copy(**copy.deepcopy({k: v for k, v in overrides.items() if k != "system"}))


# ------------------------------------------------------------------- the grid
@dataclasses.dataclass
class GridResult:
    experiment: str
    dataset: Dataset
    results: Dict[str, Dict[int, EvalResult]]      # name -> seed -> result
    out_dir: Path

    def systems(self) -> List[str]:
        return list(self.results)


def run_grid(cfg: Config, grid: Mapping[str, Mapping], experiment: str, ds: Optional[Dataset] = None,
             seeds: Optional[Sequence[int]] = None, only: Optional[Sequence[str]] = None) -> GridResult:
    ds = ds or load_dataset(cfg.dataset)
    seeds = list(seeds or cfg.experiment.seeds)
    llm = LLMBackend.from_config(cfg.llm)
    shared_by_cfg: Dict[str, SharedRetrievalResources] = {}
    out_root = resolve_path(cfg.experiment.results_dir)
    results: Dict[str, Dict[int, EvalResult]] = {}
    _banner(ds, llm)
    for name, ov in grid.items():
        if only and name not in only:
            continue
        sys_cfg = apply_overrides(cfg, ov)
        key = json.dumps(dataclasses.asdict(sys_cfg.retrieval), sort_keys=True)
        shared = shared_by_cfg.setdefault(key, SharedRetrievalResources(sys_cfg.retrieval))
        evaluator = Evaluator(sys_cfg, llm, out_root)
        results[name] = {}
        for seed in seeds:
            system = build_system(ov.get("system", "multi_agent"), sys_cfg, llm, shared, name=name)
            results[name][seed] = evaluator.run(system, ds, name, seed, experiment)
    gr = GridResult(experiment, ds, results, out_root / experiment)
    write_tables(gr, reference=_reference(grid, only))
    return gr


def _reference(grid: Mapping[str, Mapping], only: Optional[Sequence[str]]) -> Optional[str]:
    for cand in ("multi_agent_full", "full"):
        if cand in grid and (not only or cand in only):
            return cand
    return None


def _banner(ds: Dataset, llm: LLMBackend) -> None:
    kind = "REAL HotpotQA" if ds.kind == DatasetKind.REAL else "*** SYNTHETIC DATA (not a benchmark) ***"
    be = f"{llm.provider_name}/{llm.model_name}" + ("  *** MOCK LLM ***" if llm.provider_name == "mock" else "")
    log.info("dataset: %s  n=%d  version=%s | backend: %s", kind, len(ds.examples), ds.version, be)


# ------------------------------------------------------------- tables / tests
TABLE_METRICS = (("em", "answer"), ("f1", "answer"), ("sp_f1", "supporting_facts"),
                 ("retrieval_recall", "supporting_facts"), ("llm_calls", "cost"), ("total_tokens", "cost"),
                 ("cost_usd", "cost"), ("latency_s", "cost"), ("hops", "cost"))


def provenance_note(gr: GridResult) -> str:
    any_res = next(iter(next(iter(gr.results.values())).values()))
    be = any_res.summary["backend"]
    lines = [f"dataset: **{gr.dataset.name}** ({gr.dataset.kind.value}, n={len(gr.dataset.examples)}, "
             f"version `{gr.dataset.version}`)", f"backend: **{be['provider']}/{be['model']}** "
             f"(cost is an estimate from configs/models.yaml)"]
    if gr.dataset.kind != DatasetKind.REAL:
        lines.insert(0, "> **WARNING: SYNTHETIC DATA — these are NOT HotpotQA benchmark results.**")
    if be["provider"] == "mock":
        lines.insert(0, "> **WARNING: MOCK LLM — heuristic backend; numbers verify control flow only.**")
    return "\n".join(lines)


def write_tables(gr: GridResult, reference: Optional[str] = None) -> Dict[str, Path]:
    tables_dir = gr.out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    # ---- main table: mean ± std over seeds
    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, by_seed in gr.results.items():
        agg[name] = {}
        for metric, section in TABLE_METRICS:
            agg[name][metric] = mean_std([r.summary[section][metric] for r in by_seed.values()])
    md = [f"# {gr.experiment}", "", provenance_note(gr), "",
          f"seeds: {sorted(next(iter(gr.results.values())).keys())} — values are mean ± std over seeds "
          "(×100 for EM/F1/recall)", "",
          "| system | EM | F1 | SP-F1 | Ret-Recall | LLM calls | tokens | cost $ | latency s | hops |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in agg.items():
        md.append(f"| {name} | {format_mean_std(m['em'])} | {format_mean_std(m['f1'])} | "
                  f"{format_mean_std(m['sp_f1'])} | {format_mean_std(m['retrieval_recall'])} | "
                  f"{format_mean_std(m['llm_calls'], 1, 2)} | {format_mean_std(m['total_tokens'], 1, 0)} | "
                  f"{format_mean_std(m['cost_usd'], 1, 4)} | {format_mean_std(m['latency_s'], 1, 2)} | "
                  f"{format_mean_std(m['hops'], 1, 2)} |")
    # ---- paired tests vs reference (per seed, then Holm correction)
    tests: List[Dict] = []
    if reference and reference in gr.results:
        for name, by_seed in gr.results.items():
            if name == reference:
                continue
            for seed, res in by_seed.items():
                ref = gr.results[reference][seed]
                for metric in ("em", "f1"):
                    t = paired_test(res.per_question(metric), ref.per_question(metric), metric, name, reference)
                    tests.append({"seed": seed, **t.to_dict()})
        if tests:
            flags = holm_bonferroni([t["p_value"] for t in tests])
            for t, f in zip(tests, flags):
                t["significant_holm"] = f
            md += ["", f"## Paired tests vs `{reference}` (per seed; McNemar for EM, Wilcoxon for F1; "
                   "Holm-corrected across all comparisons)", "",
                   "| system | seed | metric | Δ (ref − sys) | 95% CI | test | p | Holm sig. |",
                   "|---|---|---|---|---|---|---|---|"]
            for t in tests:
                md.append(f"| {t['system_a']} | {t['seed']} | {t['metric']} | {t['diff'] * 100:+.1f} | "
                          f"[{t['ci_lo'] * 100:+.1f}, {t['ci_hi'] * 100:+.1f}] | {t['test']} | {t['p_value']:.3g} | "
                          f"{'yes' if t['significant_holm'] else 'no'} |")
    # ---- breakdowns + terminations + failures for each system (seed-averaged counts)
    md += ["", "## Termination reasons (summed over seeds)", ""]
    for name, by_seed in gr.results.items():
        c: Dict[str, int] = {}
        for r in by_seed.values():
            for k, v in r.summary["terminations"].items():
                c[k] = c.get(k, 0) + v
        md.append(f"- **{name}**: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    first = next(iter(next(iter(gr.results.values())).values()))
    md += ["", "## Failure taxonomy (fraction of questions, seed-averaged)", "",
           "| system | " + " | ".join(first.summary["failures"]) + " |"]
    md.append("|---|" + "---|" * len(first.summary["failures"]))
    for name, by_seed in gr.results.items():
        fr = {}
        for r in by_seed.values():
            for k, v in r.summary["failures"].items():
                fr[k] = fr.get(k, 0.0) + v["fraction"] / len(by_seed)
        md.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in fr.values()) + " |")
    md += ["", "## EM by question type / level (seed-averaged)", ""]
    for name, by_seed in gr.results.items():
        bt, bl = {}, {}
        for r in by_seed.values():
            for k, v in r.summary["by_type"].items():
                bt[k] = bt.get(k, 0.0) + v["em"] / len(by_seed)
            for k, v in r.summary["by_level"].items():
                bl[k] = bl.get(k, 0.0) + v["em"] / len(by_seed)
        md.append(f"- **{name}**: type " + ", ".join(f"{k}={v * 100:.1f}" for k, v in bt.items())
                  + " | level " + ", ".join(f"{k}={v * 100:.1f}" for k, v in bl.items()))
    paths = {"markdown": tables_dir / "results.md", "json": tables_dir / "results.json",
             "csv": tables_dir / "results.csv", "tests": tables_dir / "paired_tests.json"}
    paths["markdown"].write_text("\n".join(md) + "\n", encoding="utf-8")
    paths["json"].write_text(json.dumps({"experiment": gr.experiment, "dataset": {
        "name": gr.dataset.name, "kind": gr.dataset.kind.value, "n": len(gr.dataset.examples),
        "version": gr.dataset.version}, "aggregate": agg, "tests": tests}, indent=2), encoding="utf-8")
    paths["tests"].write_text(json.dumps(tests, indent=2), encoding="utf-8")
    with open(paths["csv"], "w", encoding="utf-8") as fh:
        fh.write("system,seed,dataset_kind," + ",".join(m for m, _ in TABLE_METRICS) + "\n")
        for name, by_seed in gr.results.items():
            for seed, r in by_seed.items():
                fh.write(f"{name},{seed},{r.dataset_kind}," + ",".join(
                    f"{r.summary[s][m]:.6f}" for m, s in TABLE_METRICS) + "\n")
    log.info("tables → %s", tables_dir)
    return paths
