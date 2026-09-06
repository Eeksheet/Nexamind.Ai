"""Integration: dataset -> retrieval -> agents -> orchestrator -> evaluator -> tables/figures."""
import json

from src.evaluation.evaluator import Evaluator, load_eval_result
from src.experiments.figures import standard_figures
from src.experiments.runner import run_grid


def test_pipeline_end_to_end(cfg, llm, synthetic, tmp_path):
    ev = Evaluator(cfg, llm, out_root=tmp_path)
    from src.orchestration.systems import build_system
    res = ev.run(build_system("multi_agent", cfg, llm), synthetic, "multi_agent", seed=42, experiment="e2e")
    assert res.n == len(synthetic.examples)
    s = res.summary
    assert 0 <= s["answer"]["em"] <= 1 and s["cost"]["llm_calls"] > 0
    assert s["dataset"]["is_real_benchmark"] is False and "WARNING" in s   # synthetic must be flagged
    d = res.out_dir
    assert (d / "manifest.json").exists() and (d / "per_question.jsonl").exists() and (d / "traces.jsonl").exists()
    man = json.loads((d / "manifest.json").read_text())
    for k in ("seed", "dataset_kind", "dataset_size", "backend", "model", "embedding_model", "retrieval_method",
              "alpha", "top_k", "reranker", "max_hops", "prompt_version", "timestamp"):
        assert k in man
    assert man["dataset_kind"] == "synthetic" and man["backend"] == "mock"
    loaded = load_eval_result(d)
    assert loaded.n == res.n and loaded.summary["answer"]["em"] == s["answer"]["em"]


def test_grid_multi_seed_tables_and_figures(cfg, llm, synthetic, tmp_path):
    cfg2 = cfg.copy(experiment={"results_dir": str(tmp_path), "seeds": [42, 43]})
    grid = {"bm25_rag": {"system": "single_pass"}, "multi_agent_full": {"system": "multi_agent"},
            "no_critic": {"system": "multi_agent", "orchestrator": {"use_critic": False}}}
    gr = run_grid(cfg2, grid, "grid", ds=synthetic)
    assert set(gr.results) == set(grid) and set(gr.results["multi_agent_full"]) == {42, 43}
    md = (gr.out_dir / "tables" / "results.md").read_text()
    assert "SYNTHETIC" in md and "MOCK" in md and "±" in md and "Paired tests" in md
    tests = json.loads((gr.out_dir / "tables" / "paired_tests.json").read_text())
    assert tests and all(0 <= t["p_value"] <= 1 for t in tests)
    figs = standard_figures(gr, tmp_path / "figs")
    assert all(p.exists() and p.stat().st_size > 0 for p in figs)
