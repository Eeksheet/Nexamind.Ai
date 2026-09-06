import pytest

from src.datasets.hotpotqa import (
    Dataset,
    DatasetKind,
    DatasetValidationError,
    Example,
    load_dataset,
    synthetic_multihop,
    validate_dataset,
    validate_example,
)
from src.utils.config import DatasetConfig


def test_synthetic_is_labelled_and_valid():
    ds = synthetic_multihop(15, seed=3)
    assert ds.kind == DatasetKind.SYNTHETIC and len(ds.examples) == 15
    assert all(not validate_example(e) for e in ds.examples)
    assert all(e.supporting_facts for e in ds.examples)


def test_synthetic_is_deterministic():
    a, b = synthetic_multihub_pair()
    assert [e.question for e in a.examples] == [e.question for e in b.examples]


def synthetic_multihub_pair():
    return synthetic_multihop(5, seed=7), synthetic_multihop(5, seed=7)


def test_validation_drops_bad_examples():
    good = synthetic_multihop(3, seed=0).examples
    bad = Example(qid="bad", question="?", answer="", paragraphs={"T": ["s"]}, supporting_facts=[("T", 9)])
    ds = Dataset(examples=good + [bad], kind=DatasetKind.SYNTHETIC, name="x", version="v")
    stats = validate_dataset(ds)
    assert stats["invalid"] == 1 and stats["n"] == 4
    with pytest.raises(DatasetValidationError):
        validate_dataset(Dataset(examples=[bad], kind=DatasetKind.SYNTHETIC, name="x", version="v"), strict=True)


def test_load_dataset_force_synthetic_never_claims_real(tmp_path):
    cfg = DatasetConfig(size="7", force_synthetic=True, raw_dir=str(tmp_path / "raw"),
                        processed_dir=str(tmp_path / "proc"))
    ds = load_dataset(cfg, download=False)
    assert ds.kind == DatasetKind.SYNTHETIC and len(ds.examples) == 7
    assert "synthetic" in ds.name.lower() or ds.kind == DatasetKind.SYNTHETIC


def test_load_dataset_without_data_and_no_fallback_raises(tmp_path):
    cfg = DatasetConfig(size="smoke", allow_synthetic=False, raw_dir=str(tmp_path / "raw"),
                        processed_dir=str(tmp_path / "proc"))
    with pytest.raises((FileNotFoundError, RuntimeError)):
        load_dataset(cfg, download=False)


@pytest.mark.real_data
def test_real_hotpotqa_if_present():
    from src.utils.config import resolve_path
    raw = resolve_path("data/raw")
    if not any(raw.rglob("*.parquet")) and not any(raw.glob("*.json")):
        pytest.skip("real HotpotQA not downloaded")
    ds = load_dataset(DatasetConfig(size="20", allow_synthetic=False), download=False)
    assert ds.kind == DatasetKind.REAL and len(ds.examples) == 20
    ex = ds.examples[0]
    assert ex.qtype in ("bridge", "comparison") and ex.level in ("easy", "medium", "hard")
    assert all(t in ex.paragraphs and 0 <= i < len(ex.paragraphs[t]) for t, i in ex.supporting_facts)
