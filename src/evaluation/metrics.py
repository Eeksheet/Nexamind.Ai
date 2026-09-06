"""Answer metrics (official HotpotQA normalisation), retrieval metrics and cost metrics."""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

Unit = Tuple[str, int]


# ----------------------------------------------------------------- answer
def normalize_answer(s: str) -> str:
    """Official HotpotQA / SQuAD normalisation (lower, strip punct/articles/space)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    # official HotpotQA special-case: yes/no/noanswer must match exactly
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        return float(p == g)
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(pt), ns / len(gt)
    return 2 * prec * rec / (prec + rec)


def precision_recall_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0, 0.0, 0.0
    prec, rec = ns / len(p), ns / len(g)
    return prec, rec, 2 * prec * rec / (prec + rec)


def answer_metrics(pred: str, gold: str) -> Dict[str, float]:
    prec, rec, _ = precision_recall_f1(pred, gold)
    return {"em": exact_match(pred, gold), "f1": f1_score(pred, gold),
            "precision": prec, "recall": rec}


# --------------------------------------------------------------- retrieval
def _hits(ranked: Sequence[Unit], gold: Iterable[Unit]) -> List[int]:
    gs = set(gold)
    return [1 if u in gs else 0 for u in ranked]


def recall_at_k(ranked: Sequence[Unit], gold: Iterable[Unit], k: int) -> float:
    gs = set(gold)
    if not gs:
        return 0.0
    return len(gs & set(ranked[:k])) / len(gs)


def precision_at_k(ranked: Sequence[Unit], gold: Iterable[Unit], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(gold) & set(ranked[:k])) / k


def mrr(ranked: Sequence[Unit], gold: Iterable[Unit]) -> float:
    for i, h in enumerate(_hits(ranked, gold), 1):
        if h:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[Unit], gold: Iterable[Unit], k: int) -> float:
    gs = set(gold)
    if not gs:
        return 0.0
    dcg = sum(h / math.log2(i + 1) for i, h in enumerate(_hits(ranked[:k], gs), 1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gs), k) + 1))
    return dcg / ideal if ideal else 0.0


def supporting_fact_prf(retrieved: Iterable[Unit], gold: Iterable[Unit]) -> Dict[str, float]:
    r, g = set(retrieved), set(gold)
    tp = len(r & g)
    p = tp / len(r) if r else 0.0
    rc = tp / len(g) if g else 0.0
    f = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"sp_precision": p, "sp_recall": rc, "sp_f1": f, "sp_em": float(r == g and bool(g))}


def paragraph_recall(retrieved: Iterable[Unit], gold: Iterable[Unit]) -> float:
    gt = {t for t, _ in gold}
    rt = {t for t, _ in retrieved}
    return len(gt & rt) / len(gt) if gt else 0.0


def retrieval_metrics(ranked: Sequence[Unit], gold: Iterable[Unit],
                      ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    gold = list(gold)
    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        out[f"precision@{k}"] = precision_at_k(ranked, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
    out["mrr"] = mrr(ranked, gold)
    return out


# --------------------------------------------------------------------- joint
def joint_metrics(ans: Dict[str, float], sp: Dict[str, float]) -> Dict[str, float]:
    """HotpotQA joint EM/F1: product of answer and supporting-fact scores."""
    jp = ans["precision"] * (sp["sp_precision"])
    jr = ans["recall"] * (sp["sp_recall"])
    jf = 2 * jp * jr / (jp + jr) if jp + jr else 0.0
    return {"joint_em": ans["em"] * sp["sp_em"], "joint_f1": jf}
