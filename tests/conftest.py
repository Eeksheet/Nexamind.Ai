import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.hotpotqa import synthetic_multihop  # noqa: E402
from src.llm.backend import LLMBackend  # noqa: E402
from src.utils.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    # bm25 keeps the suite offline / fast; dense paths are covered by an opt-in test
    return load_config(overrides={"retrieval": {"method": "bm25"}, "llm": {"provider": "mock"},
                                  "dataset": {"force_synthetic": True}})


@pytest.fixture(scope="session")
def llm(cfg):
    return LLMBackend.from_config(cfg.llm, env={})


@pytest.fixture(scope="session")
def synthetic():
    return synthetic_multihop(12, seed=1)


@pytest.fixture
def example(synthetic):
    return synthetic.examples[0]
