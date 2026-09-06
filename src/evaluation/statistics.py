"""Statistical utilities: mean±std over seeds, bootstrap CIs and paired tests.

Nothing here invents numbers: every function is a pure computation over the
per-question scores you pass in.  Paired tests require *aligned* per-question
arrays (same questions, same order) for the two systems.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size)}


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    return {"mean": float(a.mean()), "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2))}


@dataclass
class PairedTest:
    metric: str
    system_a: str
    system_b: str
    n: int
    mean_a: float
    mean_b: float
    diff: float
    ci_lo: float
    ci_hi: float
    test: str
    statistic: float
    p_value: float
    effect_size: float          # Cohen's d on paired differences
    significant_05: bool

    def to_dict(self) -> Dict:
        return asdict(self)


def paired_test(a: Sequence[float], b: Sequence[float], metric: str = "", system_a: str = "A",
                system_b: str = "B", test: str = "auto", n_boot: int = 2000, seed: int = 0) -> PairedTest:
    """Paired comparison of per-question scores.

    * ``test="ttest"``     paired t-test
    * ``test="wilcoxon"``  Wilcoxon signed-rank
    * ``test="mcnemar"``   exact McNemar (binary metrics such as EM)
    * ``test="auto"``      McNemar if both arrays are binary, else Wilcoxon
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired_test needs aligned arrays of equal length")
    n = int(x.size)
    d = y - x
    binary = bool(np.isin(x, (0, 1)).all() and np.isin(y, (0, 1)).all())
    if test == "auto":
        test = "mcnemar" if binary else "wilcoxon"
    stat, p = float("nan"), float("nan")
    if n >= 2:
        if test == "ttest":
            if np.allclose(d, d[0]):
                stat, p = 0.0, 1.0
            else:
                r = stats.ttest_rel(y, x)
                stat, p = float(r.statistic), float(r.pvalue)
        elif test == "wilcoxon":
            if np.all(d == 0):
                stat, p = 0.0, 1.0
            else:
                r = stats.wilcoxon(y, x, zero_method="wilcox")
                stat, p = float(r.statistic), float(r.pvalue)
        elif test == "mcnemar":
            b01 = int(((x == 0) & (y == 1)).sum())
            b10 = int(((x == 1) & (y == 0)).sum())
            disc = b01 + b10
            stat = float(min(b01, b10))
            p = 1.0 if disc == 0 else float(stats.binomtest(b01, disc, 0.5).pvalue)
        else:
            raise ValueError(f"unknown test {test!r}")
    ci = bootstrap_ci(d, n_boot=n_boot, seed=seed) if n else {"lo": math.nan, "hi": math.nan}
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    effect = float(d.mean() / sd) if sd > 0 else 0.0
    return PairedTest(metric, system_a, system_b, n, float(x.mean()) if n else math.nan,
                      float(y.mean()) if n else math.nan, float(d.mean()) if n else math.nan,
                      ci["lo"], ci["hi"], test, stat, p, effect, bool(p < 0.05) if not math.isnan(p) else False)


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> Sequence[bool]:
    """Holm step-down correction; returns per-hypothesis rejection flags."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject.tolist()


def format_mean_std(ms: Dict[str, float], scale: float = 100.0, digits: int = 1) -> str:
    if ms.get("n", 0) == 0 or math.isnan(ms["mean"]):
        return "n/a"
    if ms.get("n", 0) == 1:
        return f"{ms['mean'] * scale:.{digits}f}"
    return f"{ms['mean'] * scale:.{digits}f} ± {ms['std'] * scale:.{digits}f}"


def seed_summary(per_seed: Dict[int, float]) -> Optional[Dict[str, float]]:
    """mean±std of a metric across seeds (dict seed→value)."""
    return mean_std(list(per_seed.values())) if per_seed else None
