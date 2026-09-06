import math

import numpy as np
import pytest

from src.evaluation.failure_analysis import categorize
from src.evaluation.metrics import (
    answer_metrics,
    exact_match,
    f1_score,
    joint_metrics,
    mrr,
    ndcg_at_k,
    normalize_answer,
    precision_at_k,
    recall_at_k,
    retrieval_metrics,
    supporting_fact_prf,
)
from src.evaluation.statistics import bootstrap_ci, holm_bonferroni, mean_std, paired_test


def test_normalize_and_em():
    assert normalize_answer("The Beatles!") == "beatles"
    assert exact_match("the Beatles", "Beatles") == 1.0
    assert exact_match("Beatles", "Rolling Stones") == 0.0


def test_f1_hotpot_special_cases():
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("yes it is", "yes") == 0.0   # official: yes/no must match exactly
    assert abs(f1_score("New York City", "New York") - 0.8) < 1e-9


def test_retrieval_metrics():
    ranked = [("A", 0), ("B", 1), ("C", 2), ("A", 1)]
    gold = [("B", 1), ("A", 1)]
    assert recall_at_k(ranked, gold, 1) == 0.0
    assert recall_at_k(ranked, gold, 2) == 0.5
    assert recall_at_k(ranked, gold, 4) == 1.0
    assert precision_at_k(ranked, gold, 2) == 0.5
    assert mrr(ranked, gold) == 0.5
    assert 0 < ndcg_at_k(ranked, gold, 4) < 1
    assert ndcg_at_k([("B", 1), ("A", 1)], gold, 2) == pytest.approx(1.0)
    m = retrieval_metrics(ranked, gold)
    assert set(m) >= {"recall@1", "precision@5", "ndcg@10", "mrr"}


def test_supporting_fact_prf_and_joint():
    sp = supporting_fact_prf([("A", 0), ("B", 1)], [("B", 1), ("C", 0)])
    assert sp["sp_precision"] == 0.5 and sp["sp_recall"] == 0.5 and sp["sp_em"] == 0.0
    ans = answer_metrics("Paris", "Paris")
    j = joint_metrics(ans, sp)
    assert j["joint_em"] == 0.0 and 0 < j["joint_f1"] < 1


def test_mean_std_and_bootstrap():
    ms = mean_std([0.5, 0.6, 0.7])
    assert ms["mean"] == pytest.approx(0.6) and ms["std"] == pytest.approx(0.1) and ms["n"] == 3
    assert mean_std([])["n"] == 0 and math.isnan(mean_std([])["mean"])
    ci = bootstrap_ci([1, 1, 1, 0, 0], seed=0)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]


def test_paired_tests_do_not_fabricate_significance():
    a = np.zeros(20)
    t = paired_test(a, a, "em")
    assert t.p_value == 1.0 and not t.significant_05 and t.diff == 0.0
    t2 = paired_test([0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], "em")
    assert t2.test == "mcnemar" and t2.p_value < 0.05 and t2.diff == 1.0
    t3 = paired_test([0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], "f1")
    assert t3.test == "wilcoxon" and 0 <= t3.p_value <= 1
    with pytest.raises(ValueError):
        paired_test([1, 2], [1], "em")


def test_holm():
    assert holm_bonferroni([0.001, 0.04, 0.5]) == [True, False, False]


def test_failure_categories():
    gold_units = [("A", 0), ("B", 0)]
    ev = [{"title": "A", "sent_idx": 0, "sentence": "x Paris"}, {"title": "B", "sent_idx": 0, "sentence": "y"}]
    assert categorize({"answer": "Paris"}, "Paris", gold_units) == "correct"
    assert categorize({"answer": "Rome", "error": "boom"}, "Paris", gold_units) == "error"
    assert categorize({"answer": "Rome", "evidence": []}, "Paris", gold_units) == "retrieval_miss"
    assert categorize({"answer": "Rome", "evidence": ev[:1]}, "Paris", gold_units) == "partial_retrieval"
    assert categorize({"answer": "Paris France", "evidence": ev}, "Paris", gold_units) == "answer_format"
    run = {"answer": "Rome", "evidence": ev, "termination": "critic_approved",
           "trace": [{"agent": "reasoner", "answer": "Paris"}]}
    assert categorize(run, "Paris", gold_units) == "critic_false_reject"
    run = {"answer": "Rome", "evidence": ev, "termination": "max_hops", "trace": []}
    assert categorize(run, "Paris", gold_units) == "budget_exhausted"
    run = {"answer": "Rome", "evidence": ev, "termination": "critic_approved", "trace": []}
    assert categorize(run, "Paris", gold_units) == "unsupported_answer"
