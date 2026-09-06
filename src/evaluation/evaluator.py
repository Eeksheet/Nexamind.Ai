"""End-to-end evaluator: runs a system over a dataset and writes provenance-tagged results.

Output layout (``results/<experiment>/<system>/seed_<s>/``)::

    manifest.json      reproducibility manifest (dataset kind/size, model, prompts, git rev …)
    summary.json       aggregate metrics (answer, supporting-fact, cost/latency, terminations)
    per_question.jsonl one row per question (metrics + answer + failure category)
    traces.jsonl       full execution traces (optional)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..datasets.hotpotqa import Dataset, DatasetKind, Example
from ..llm.backend import LLMBackend
from ..llm.prompts import PROMPT_VERSION
from ..orchestration.state import RunResult
from ..utils.config import Config
from ..utils.logging import get_logger
from ..utils.reproducibility import ExperimentManifest, set_seed
from .failure_analysis import breakdown, categorize, failure_table
from .metrics import answer_metrics, joint_metrics, paragraph_recall, supporting_fact_prf

log = get_logger(__name__)

System = Callable[[Example], RunResult]

ANSWER_KEYS = ("em", "f1", "precision", "recall")
SP_KEYS = ("sp_em", "sp_f1", "sp_precision", "sp_recall", "paragraph_recall", "retrieval_recall")
COST_KEYS = ("llm_calls", "retrieval_calls", "input_tokens", "output_tokens", "total_tokens",
             "cost_usd", "latency_s", "hops")


@dataclass
class EvalResult:
    system: str
    seed: int
    dataset_kind: str
    n: int
    summary: Dict
    rows: List[Dict]
    runs: List[Dict] = field(default_factory=list)
    out_dir: Optional[Path] = None

    def metric(self, name: str) -> float:
        return float(self.summary["answer"].get(name, self.summary.get("cost", {}).get(name, float("nan"))))

    def per_question(self, name: str) -> np.ndarray:
        return np.asarray([r[name] for r in self.rows], dtype=float)


def score_run(run: RunResult, ex: Example) -> Dict:
    ans = answer_metrics(run.answer, ex.answer)
    # supporting-fact prediction = units the reasoner cited (fallback: everything retrieved)
    cited = [tuple(u) for u in run.supporting_units] or run.evidence_units
    sp = supporting_fact_prf(cited, ex.supporting_facts)
    sp["paragraph_recall"] = paragraph_recall(run.evidence_units, ex.supporting_facts)
    sp["retrieval_recall"] = supporting_fact_prf(run.evidence_units, ex.supporting_facts)["sp_recall"]
    row = {"qid": ex.qid, "qtype": ex.qtype, "level": ex.level, "question": ex.question,
           "gold": ex.answer, "pred": run.answer, "confidence": run.confidence,
           "termination": run.termination, "system": run.system, **ans, **sp,
           **joint_metrics(ans, sp)}
    for k in COST_KEYS:
        row[k] = getattr(run, k)
    row["category"] = categorize(run.to_dict(), ex.answer, ex.supporting_facts)
    return row


def aggregate(rows: Sequence[Dict]) -> Dict:
    n = len(rows)

    def mean(k: str) -> float:
        return float(np.mean([r[k] for r in rows])) if n else float("nan")

    summary = {
        "n": n,
        "answer": {k: mean(k) for k in ANSWER_KEYS} | {"joint_em": mean("joint_em"), "joint_f1": mean("joint_f1")},
        "supporting_facts": {k: mean(k) for k in SP_KEYS},
        "cost": {k: mean(k) for k in COST_KEYS} | {
            "total_cost_usd": float(sum(r["cost_usd"] for r in rows)),
            "total_latency_s": float(sum(r["latency_s"] for r in rows)),
            "total_tokens_sum": int(sum(r["total_tokens"] for r in rows)),
            "p50_latency_s": float(np.median([r["latency_s"] for r in rows])) if n else float("nan"),
            "p95_latency_s": float(np.percentile([r["latency_s"] for r in rows], 95)) if n else float("nan")},
        "terminations": {},
        "failures": failure_table(rows),
        "by_type": breakdown(rows, "qtype"), "by_level": breakdown(rows, "level"),
        "by_type_f1": breakdown(rows, "qtype", "f1"), "by_level_f1": breakdown(rows, "level", "f1"),
    }
    for r in rows:
        summary["terminations"][r["termination"]] = summary["terminations"].get(r["termination"], 0) + 1
    return summary


class Evaluator:
    def __init__(self, cfg: Config, llm: LLMBackend, out_root: Optional[Path] = None,
                 save_traces: Optional[bool] = None) -> None:
        self.cfg = cfg
        self.llm = llm
        self.out_root = Path(out_root) if out_root else Path(cfg.experiment.results_dir)
        self.save_traces = cfg.experiment.save_traces if save_traces is None else save_traces

    def run(self, system: System, ds: Dataset, system_name: str, seed: int, experiment: str = "default",
            notes: str = "", progress_every: int = 10) -> EvalResult:
        set_seed(seed)
        rows, runs = [], []
        t0 = time.perf_counter()
        for i, ex in enumerate(ds.examples, 1):
            r = system(ex)
            rows.append(score_run(r, ex))
            runs.append(r.to_dict())
            if progress_every and i % progress_every == 0:
                em = np.mean([x["em"] for x in rows])
                log.info("[%s seed=%d] %d/%d  EM=%.3f  %.1fs", system_name, seed, i, len(ds.examples), em,
                         time.perf_counter() - t0)
        summary = aggregate(rows)
        summary["dataset"] = {"kind": ds.kind.value, "name": ds.name, "version": ds.version, "size": len(ds.examples),
                              "is_real_benchmark": ds.kind == DatasetKind.REAL}
        summary["wall_time_s"] = time.perf_counter() - t0
        summary["backend"] = {"provider": self.llm.provider_name, "model": self.llm.model_name,
                              "cost_is_estimate": True, "pricing_known": self.llm.pricing_known}
        if ds.kind != DatasetKind.REAL:
            summary["WARNING"] = "SYNTHETIC DATA – these are NOT HotpotQA benchmark results"
        if self.llm.provider_name == "mock":
            summary["WARNING_LLM"] = "MOCK LLM – heuristic backend; numbers are control-flow checks, not model quality"

        out_dir = self.out_root / experiment / system_name / f"seed_{seed}"
        res = EvalResult(system_name, seed, ds.kind.value, len(rows), summary, rows, runs, out_dir)
        self._save(res, ds, experiment, notes)
        return res

    def _save(self, res: EvalResult, ds: Dataset, experiment: str, notes: str) -> None:
        d = res.out_dir
        assert d is not None
        d.mkdir(parents=True, exist_ok=True)
        rc = self.cfg.retrieval
        manifest = ExperimentManifest(
            experiment=experiment, seed=res.seed, dataset_name=ds.name, dataset_kind=ds.kind.value,
            dataset_version=ds.version, dataset_size=res.n, backend=self.llm.provider_name,
            model=self.llm.model_name, embedding_model=rc.embedding_model, retrieval_method=rc.method,
            alpha=rc.alpha, top_k=rc.top_k, reranker=rc.reranker, max_hops=self.cfg.orchestrator.budget.max_hops,
            prompt_version=PROMPT_VERSION, config=self.cfg.to_dict(),
            notes=notes + (" | SYNTHETIC DATA" if ds.kind != DatasetKind.REAL else ""))
        manifest.save(d / "manifest.json")
        with open(d / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(res.summary, fh, indent=2, default=str)
        with open(d / "per_question.jsonl", "w", encoding="utf-8") as fh:
            for r in res.rows:
                fh.write(json.dumps(r, default=str) + "\n")
        if self.save_traces:
            with open(d / "traces.jsonl", "w", encoding="utf-8") as fh:
                for r in res.runs:
                    fh.write(json.dumps(r, default=str) + "\n")
        log.info("saved %s → %s  EM=%.3f F1=%.3f SP-F1=%.3f cost=$%.4f", res.system, d,
                 res.summary["answer"]["em"], res.summary["answer"]["f1"],
                 res.summary["supporting_facts"]["sp_f1"], res.summary["cost"]["total_cost_usd"])


def load_eval_result(d: Path) -> EvalResult:
    d = Path(d)
    summary = json.loads((d / "summary.json").read_text())
    rows = [json.loads(ln) for ln in (d / "per_question.jsonl").read_text().splitlines() if ln.strip()]
    manifest = json.loads((d / "manifest.json").read_text())
    return EvalResult(d.parent.name, manifest["seed"], manifest["dataset_kind"], len(rows), summary, rows, [], d)
