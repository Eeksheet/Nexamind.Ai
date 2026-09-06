"""Publication-style figures (matplotlib, headless).  Every figure carries a provenance
subtitle (dataset kind / backend) so synthetic or mock plots are self-labelling."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..datasets.hotpotqa import DatasetKind  # noqa: E402
from ..evaluation.statistics import mean_std  # noqa: E402
from .runner import GridResult  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def _prov(gr: GridResult) -> str:
    r = next(iter(next(iter(gr.results.values())).values()))
    be = r.summary["backend"]
    tag = f"{gr.dataset.name} ({gr.dataset.kind.value}, n={len(gr.dataset.examples)}) · {be['provider']}/{be['model']}"
    if gr.dataset.kind != DatasetKind.REAL or be["provider"] == "mock":
        tag = "NOT A BENCHMARK RESULT — " + tag
    return tag


def _agg(gr: GridResult, metric: str, section: str) -> Dict[str, Dict[str, float]]:
    return {n: mean_std([r.summary[section][metric] for r in bs.values()]) for n, bs in gr.results.items()}


def bar_metric(gr: GridResult, metric: str, section: str, out: Path, title: Optional[str] = None,
               scale: float = 100.0) -> Path:
    a = _agg(gr, metric, section)
    names = list(a)
    means = np.array([a[n]["mean"] for n in names]) * scale
    stds = np.array([a[n]["std"] for n in names]) * scale
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(names) + 2), 3.4))
    ax.bar(range(len(names)), means, yerr=stds, capsize=3, color="#4C72B0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel(f"{metric.upper()} ({'%' if scale == 100 else 'raw'})")
    ax.set_title(title or f"{metric.upper()} by system (mean ± std over seeds)")
    fig.text(0.5, -0.02, _prov(gr), ha="center", fontsize=7, color="gray")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def accuracy_vs_cost(gr: GridResult, out: Path, cost_metric: str = "total_tokens") -> Path:
    em = _agg(gr, "em", "answer")
    cost = _agg(gr, cost_metric, "cost")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for n in gr.results:
        ax.errorbar(cost[n]["mean"], em[n]["mean"] * 100, yerr=em[n]["std"] * 100, fmt="o", capsize=3)
        ax.annotate(n, (cost[n]["mean"], em[n]["mean"] * 100), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel(f"{cost_metric.replace('_', ' ')} per question")
    ax.set_ylabel("EM (%)")
    ax.set_title("Accuracy vs. cost")
    fig.text(0.5, -0.02, _prov(gr), ha="center", fontsize=7, color="gray")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def hop_curve(points: Mapping[int, Mapping[str, Dict[str, float]]], out: Path, prov: str) -> Path:
    """points: max_hops -> {"em": mean_std, "f1": mean_std, "llm_calls": mean_std}"""
    hs = sorted(points)
    fig, ax1 = plt.subplots(figsize=(5, 3.4))
    for m, c in (("em", "#4C72B0"), ("f1", "#55A868")):
        ax1.errorbar(hs, [points[h][m]["mean"] * 100 for h in hs], yerr=[points[h][m]["std"] * 100 for h in hs],
                     marker="o", capsize=3, label=m.upper(), color=c)
    ax1.set_xlabel("max hops")
    ax1.set_ylabel("score (%)")
    ax2 = ax1.twinx()
    ax2.plot(hs, [points[h]["llm_calls"]["mean"] for h in hs], "s--", color="gray", label="LLM calls")
    ax2.set_ylabel("LLM calls / question")
    ax1.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="lower right", fontsize=7)
    ax1.set_title("Effect of hop budget")
    fig.text(0.5, -0.02, prov, ha="center", fontsize=7, color="gray")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def retrieval_recall_curves(res: Mapping[str, Mapping[str, float]], out: Path, prov: str,
                            ks: Sequence[int] = (1, 3, 5, 10)) -> Path:
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for m, r in res.items():
        ax.plot(ks, [r[f"recall@{k}"] * 100 for k in ks], marker="o", label=m)
    ax.set_xlabel("k")
    ax.set_ylabel("supporting-fact recall@k (%)")
    ax.set_xticks(list(ks))
    ax.legend(fontsize=7)
    ax.set_title("Retrieval recall@k")
    fig.text(0.5, -0.02, prov, ha="center", fontsize=7, color="gray")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def alpha_curve(sweep: Mapping[float, Mapping[str, float]], out: Path, prov: str, k: int = 5) -> Path:
    al = sorted(sweep)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(al, [sweep[a][f"recall@{k}"] * 100 for a in al], marker="o", label=f"recall@{k}")
    ax.plot(al, [sweep[a]["mrr"] * 100 for a in al], marker="s", label="MRR")
    ax.set_xlabel("alpha (weight of BM25 in hybrid fusion)")
    ax.set_ylabel("%")
    ax.legend(fontsize=7)
    ax.set_title("Hybrid fusion alpha sweep")
    fig.text(0.5, -0.02, prov, ha="center", fontsize=7, color="gray")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def failure_stacked(gr: GridResult, out: Path) -> Path:
    cats: List[str] = list(next(iter(next(iter(gr.results.values())).values())).summary["failures"])
    names = list(gr.results)
    data = np.zeros((len(cats), len(names)))
    for j, n in enumerate(names):
        for r in gr.results[n].values():
            for i, c in enumerate(cats):
                data[i, j] += r.summary["failures"][c]["fraction"] / len(gr.results[n])
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(names) + 2), 3.6))
    bottom = np.zeros(len(names))
    for i, c in enumerate(cats):
        if data[i].sum() == 0:
            continue
        ax.bar(names, data[i], bottom=bottom, label=c)
        bottom += data[i]
    ax.set_ylabel("fraction of questions")
    ax.set_title("Outcome / failure taxonomy")
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.text(0.5, -0.02, _prov(gr), ha="center", fontsize=7, color="gray")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def standard_figures(gr: GridResult, fig_dir: Path) -> List[Path]:
    outs = [bar_metric(gr, "em", "answer", fig_dir / f"{gr.experiment}_em.png"),
            bar_metric(gr, "f1", "answer", fig_dir / f"{gr.experiment}_f1.png"),
            bar_metric(gr, "sp_f1", "supporting_facts", fig_dir / f"{gr.experiment}_sp_f1.png"),
            accuracy_vs_cost(gr, fig_dir / f"{gr.experiment}_em_vs_tokens.png"),
            failure_stacked(gr, fig_dir / f"{gr.experiment}_failures.png")]
    return outs
