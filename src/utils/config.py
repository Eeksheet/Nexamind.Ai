"""Configuration management.

All tunable knobs live in typed dataclasses that can be loaded from YAML
(``configs/*.yaml``) and overridden programmatically.  Nothing in ``src/``
reads YAML directly except :func:`load_config`.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"


@dataclass
class DatasetConfig:
    name: str = "hotpotqa"
    split: str = "validation"
    setting: str = "distractor"
    size: str = "smoke"           # smoke | 100 | 500 | 1000 | full | <int>
    allow_synthetic: bool = True  # fall back to synthetic if real data unavailable
    force_synthetic: bool = False
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    seed: int = 42


@dataclass
class RetrievalConfig:
    method: str = "hybrid"        # bm25 | dense | hybrid | tfidf
    top_k: int = 4
    alpha: float = 0.6            # weight of BM25 in hybrid fusion
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20
    cache_dir: str = "data/processed/cache"
    normalize: str = "minmax"     # minmax | zscore | none
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class LLMConfig:
    provider: str = "auto"        # auto | mock | anthropic | openai | gemini
    model: str = "claude-sonnet-4-5"
    temperature: float = 0.0
    max_tokens: int = 400
    timeout_s: float = 90.0
    max_retries: int = 3
    cache: bool = True
    cache_dir: str = "data/processed/llm_cache"
    base_url: Optional[str] = None
    prompt_version: str = "v2"


@dataclass
class BudgetConfig:
    max_hops: int = 3
    max_llm_calls: int = 12
    max_retrieval_calls: int = 8
    max_tokens: int = 20_000
    max_latency_s: float = 120.0
    max_cost_usd: Optional[float] = None
    confidence_threshold: float = 0.9


@dataclass
class OrchestratorConfig:
    use_planner: bool = True
    use_critic: bool = True
    use_query_rewriting: bool = True
    use_entity_carryover: bool = True
    iterative: bool = True
    budget: BudgetConfig = field(default_factory=BudgetConfig)


@dataclass
class ExperimentConfig:
    name: str = "default"
    seeds: list = field(default_factory=lambda: [42, 43, 44])
    results_dir: str = "results"
    save_traces: bool = True
    n_workers: int = 1


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def copy(self, **overrides: Any) -> "Config":
        """Deep-copy with nested overrides, e.g. ``cfg.copy(retrieval={"top_k": 8})``."""
        d = self.to_dict()
        _deep_update(d, overrides)
        return Config.from_dict(d)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Config":
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
        return cls(
            dataset=_build(DatasetConfig, d.get("dataset", {})),
            retrieval=_build(RetrievalConfig, d.get("retrieval", {})),
            llm=_build(LLMConfig, d.get("llm", {})),
            orchestrator=OrchestratorConfig(
                **{k: v for k, v in d.get("orchestrator", {}).items() if k != "budget"},
                budget=_build(BudgetConfig, d.get("orchestrator", {}).get("budget", {})),
            ),
            experiment=_build(ExperimentConfig, d.get("experiment", {})),
        )


def _build(cls: type, d: Mapping[str, Any]) -> Any:
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**d)


def _deep_update(base: Dict[str, Any], upd: Mapping[str, Any]) -> None:
    for k, v in upd.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def load_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: Optional[str | os.PathLike[str]] = None,
                overrides: Optional[Mapping[str, Any]] = None) -> Config:
    """Load ``configs/default.yaml`` merged with an optional extra YAML and overrides."""
    base = load_yaml(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else {}
    if path is not None:
        _deep_update(base, load_yaml(path))
    if overrides:
        _deep_update(base, dict(overrides))
    return Config.from_dict(base)


def resolve_path(p: str | os.PathLike[str]) -> Path:
    """Resolve a repo-relative path against the project root."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path
