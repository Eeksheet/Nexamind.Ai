"""HotpotQA (distractor setting) loading, validation and synthetic fallback.

Two dataset *kinds* exist and are never mixed:

* :attr:`DatasetKind.REAL` – HotpotQA downloaded from the public HuggingFace
  mirror (``hotpotqa/hotpot_qa``, parquet) or from a local ``hotpot_dev_distractor_v1.json``.
* :attr:`DatasetKind.SYNTHETIC` – the deterministic 2-hop bridge generator carried
  over from the prototype, used for unit tests and offline smoke runs.

Every :class:`Dataset` carries its ``kind`` and ``version`` so downstream tables can
never silently report synthetic numbers as benchmark numbers.
"""
from __future__ import annotations

import enum
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..utils.config import DatasetConfig, resolve_path
from ..utils.logging import get_logger

log = get_logger(__name__)

HF_REPO = "hotpotqa/hotpot_qa"
HF_FILES = {
    ("distractor", "validation"): "distractor/validation-00000-of-00001.parquet",
    ("distractor", "train"): "distractor/train-00000-of-00002.parquet",
}
SIZE_PRESETS: Dict[str, Optional[int]] = {
    "smoke": 20, "100": 100, "500": 500, "1000": 1000, "full": None,
}

SupportingFact = Tuple[str, int]


class DatasetKind(str, enum.Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"


@dataclass
class Example:
    """One multi-hop QA instance in the distractor setting."""

    qid: str
    question: str
    answer: str
    paragraphs: Dict[str, List[str]]          # title -> sentences
    supporting_facts: List[SupportingFact]    # (title, sentence_idx)
    qtype: str = "unknown"                    # bridge | comparison
    level: str = "unknown"                    # easy | medium | hard

    @property
    def gold_titles(self) -> List[str]:
        seen: List[str] = []
        for t, _ in self.supporting_facts:
            if t not in seen:
                seen.append(t)
        return seen

    @property
    def n_sentences(self) -> int:
        return sum(len(s) for s in self.paragraphs.values())

    def to_dict(self) -> dict:
        return dict(qid=self.qid, question=self.question, answer=self.answer,
                    paragraphs=self.paragraphs,
                    supporting_facts=[list(x) for x in self.supporting_facts],
                    qtype=self.qtype, level=self.level)

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        return cls(qid=d["qid"], question=d["question"], answer=d["answer"],
                   paragraphs=d["paragraphs"],
                   supporting_facts=[tuple(x) for x in d["supporting_facts"]],  # type: ignore[misc]
                   qtype=d.get("qtype", "unknown"), level=d.get("level", "unknown"))


@dataclass
class Dataset:
    examples: List[Example]
    kind: DatasetKind
    name: str
    version: str
    split: str = "validation"
    setting: str = "distractor"
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]

    @property
    def is_synthetic(self) -> bool:
        return self.kind is DatasetKind.SYNTHETIC

    @property
    def label(self) -> str:
        """Human-readable provenance label used in every table/figure."""
        return f"{self.name}/{self.split}/{self.setting} [{self.kind.value}, n={len(self)}]"

    def subset(self, n: Optional[int], seed: int = 42, shuffle: bool = True) -> "Dataset":
        """Deterministic subset of ``n`` examples (shuffled with ``seed``)."""
        ex = list(self.examples)
        if shuffle:
            random.Random(seed).shuffle(ex)
        if n is not None:
            ex = ex[:n]
        return Dataset(ex, self.kind, self.name, self.version, self.split, self.setting,
                       stats=compute_stats(ex))

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(dict(kind=self.kind.value, name=self.name, version=self.version,
                           split=self.split, setting=self.setting,
                           examples=[e.to_dict() for e in self.examples]), fh)
        return p

    @classmethod
    def from_json(cls, path: str | Path) -> "Dataset":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        ex = [Example.from_dict(x) for x in d["examples"]]
        return cls(ex, DatasetKind(d["kind"]), d["name"], d["version"], d["split"],
                   d["setting"], stats=compute_stats(ex))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class DatasetValidationError(ValueError):
    pass


def validate_example(ex: Example) -> List[str]:
    """Return a list of problems (empty when valid)."""
    problems: List[str] = []
    if not ex.question.strip():
        problems.append("empty question")
    if not ex.answer.strip():
        problems.append("empty answer")
    if not ex.paragraphs:
        problems.append("no paragraphs")
    for title, sents in ex.paragraphs.items():
        if not isinstance(sents, list) or not sents:
            problems.append(f"paragraph '{title}' has no sentences")
    if not ex.supporting_facts:
        problems.append("no supporting facts")
    for title, idx in ex.supporting_facts:
        if title not in ex.paragraphs:
            problems.append(f"supporting fact title '{title}' not in context")
        elif not (0 <= idx < len(ex.paragraphs[title])):
            problems.append(f"supporting fact ({title}, {idx}) out of range")
    if ex.qtype not in ("bridge", "comparison", "unknown"):
        problems.append(f"unknown question type '{ex.qtype}'")
    return problems


def validate_dataset(ds: Dataset, strict: bool = False) -> Dict[str, int]:
    """Validate every example. Returns a report; raises when ``strict`` and invalid."""
    report = {"n": len(ds), "invalid": 0, "duplicate_ids": 0}
    seen: set[str] = set()
    for ex in ds:
        if ex.qid in seen:
            report["duplicate_ids"] += 1
        seen.add(ex.qid)
        probs = validate_example(ex)
        if probs:
            report["invalid"] += 1
            log.warning("invalid example %s: %s", ex.qid, "; ".join(probs))
    if strict and (report["invalid"] or report["duplicate_ids"]):
        raise DatasetValidationError(f"dataset failed validation: {report}")
    return report


def compute_stats(examples: Sequence[Example]) -> dict:
    if not examples:
        return {"n": 0}
    n = len(examples)
    types: Dict[str, int] = {}
    levels: Dict[str, int] = {}
    for e in examples:
        types[e.qtype] = types.get(e.qtype, 0) + 1
        levels[e.level] = levels.get(e.level, 0) + 1
    return {
        "n": n,
        "avg_paragraphs": sum(len(e.paragraphs) for e in examples) / n,
        "avg_sentences": sum(e.n_sentences for e in examples) / n,
        "avg_supporting_facts": sum(len(e.supporting_facts) for e in examples) / n,
        "avg_gold_titles": sum(len(e.gold_titles) for e in examples) / n,
        "avg_question_tokens": sum(len(e.question.split()) for e in examples) / n,
        "avg_answer_tokens": sum(len(e.answer.split()) for e in examples) / n,
        "yes_no_share": sum(e.answer.strip().lower() in ("yes", "no") for e in examples) / n,
        "types": types,
        "levels": levels,
    }


# --------------------------------------------------------------------------- #
# Synthetic generator (carried over from the prototype, seeded)
# --------------------------------------------------------------------------- #
def synthetic_multihop(n: int = 200, seed: int = 42) -> Dataset:
    """Deterministic 2-hop bridge questions with distractors, mirroring HotpotQA."""
    rng = random.Random(seed)
    people = ["Ada Merrin", "Boris Kaltag", "Cira Nwosu", "Dmitri Falk", "Elena Rooke",
              "Farid Osman", "Greta Lindqvist", "Hiro Tanabe", "Ivo Petran", "Junia Barros"]
    fields = ["marine biology", "cryptography", "ceramics", "hydrology", "astrophysics",
              "linguistics", "metallurgy", "epidemiology", "cartography", "musicology"]
    insts = ["Ravenhill College", "Oster Institute", "Calderon University", "Thornfield Academy",
             "Meridian Polytechnic", "Bexley School of Science", "Larkspur University"]
    cities = ["Vantor", "Kessle", "Aramoor", "Dunhollow", "Peskaya", "Norbeck", "Tashiro"]
    years = list(range(1902, 1989))
    rows: List[Example] = []
    for i in range(n):
        p, f = rng.choice(people), rng.choice(fields)
        inst, city, yr = rng.choice(insts), rng.choice(cities), rng.choice(years)
        pa = [f"{p} is a researcher known for work in {f}.",
              f"{p} completed doctoral study at {inst} before joining the national survey.",
              f"{p} published three monographs on the subject."]
        pb = [f"{inst} is a research institution located in {city}.",
              f"{inst} was founded in {yr} and specialises in the applied sciences.",
              f"The campus of {inst} was expanded twice in the last century."]
        paras: Dict[str, List[str]] = {p: pa, inst: pb}
        for _ in range(8):
            dp, df = rng.choice(people), rng.choice(fields)
            di = rng.choice(insts)
            if dp in paras or di in paras:
                continue
            paras[dp] = [f"{dp} is a researcher known for work in {df}.",
                         f"{dp} taught at {di} for several years.",
                         f"{dp} received a regional award for that work."]
        if rng.random() < 0.5:
            q = f"In which city is the institution where {p} completed doctoral study located?"
            ans, sup = city, [(p, 1), (inst, 0)]
        else:
            q = f"In what year was the institution where {p} completed doctoral study founded?"
            ans, sup = str(yr), [(p, 1), (inst, 1)]
        items = list(paras.items())
        rng.shuffle(items)
        rows.append(Example(qid=f"synth-{i}", question=q, answer=ans, paragraphs=dict(items),
                            supporting_facts=sup, qtype="bridge", level="hard"))
    return Dataset(rows, DatasetKind.SYNTHETIC, "synthetic-2hop", f"seed{seed}",
                   split="generated", setting="distractor", stats=compute_stats(rows))


# --------------------------------------------------------------------------- #
# Real HotpotQA
# --------------------------------------------------------------------------- #
def _example_from_official_json(ex: dict) -> Example:
    paras = {t: list(s) for t, s in ex["context"]}
    return Example(qid=str(ex.get("_id", ex.get("id"))), question=ex["question"],
                   answer=ex["answer"], paragraphs=paras,
                   supporting_facts=[(t, int(i)) for t, i in ex.get("supporting_facts", [])],
                   qtype=ex.get("type", "unknown"), level=ex.get("level", "unknown"))


def _example_from_hf_row(row: dict) -> Example:
    ctx, sf = row["context"], row["supporting_facts"]
    paras = {t: list(s) for t, s in zip(ctx["title"], ctx["sentences"])}
    return Example(qid=str(row["id"]), question=row["question"], answer=row["answer"],
                   paragraphs=paras,
                   supporting_facts=list(zip(sf["title"], [int(i) for i in sf["sent_id"]])),
                   qtype=row.get("type", "unknown"), level=row.get("level", "unknown"))


def download_hotpotqa(raw_dir: Path, setting: str = "distractor",
                      split: str = "validation") -> Path:
    """Download the HF parquet shard for ``(setting, split)`` into ``raw_dir``."""
    from huggingface_hub import hf_hub_download

    fname = HF_FILES[(setting, split)]
    target = raw_dir / fname
    if target.exists():
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s/%s -> %s", HF_REPO, fname, target)
    path = hf_hub_download(HF_REPO, fname, repo_type="dataset", local_dir=str(raw_dir))
    return Path(path)


def _find_local_json(raw_dir: Path, setting: str, split: str) -> Optional[Path]:
    want = {"validation": "dev", "train": "train"}[split]
    for p in sorted(raw_dir.glob("*.json")):
        n = p.name.lower()
        if want in n and setting in n:
            return p
    return None


def load_real_hotpotqa(raw_dir: Path, setting: str = "distractor", split: str = "validation",
                       download: bool = True) -> Dataset:
    """Load HotpotQA from a local official JSON or the HF parquet mirror."""
    local = _find_local_json(raw_dir, setting, split)
    if local is not None:
        log.info("loading official HotpotQA json %s", local)
        with open(local, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        ex = [_example_from_official_json(r) for r in raw]
        version = f"official-json:{local.name}"
    else:
        parquet = raw_dir / HF_FILES[(setting, split)]
        if not parquet.exists():
            if not download:
                raise FileNotFoundError(f"HotpotQA not found in {raw_dir} and download=False")
            parquet = download_hotpotqa(raw_dir, setting, split)
        import pyarrow.parquet as pq

        table = pq.read_table(parquet)
        ex = [_example_from_hf_row(r) for r in table.to_pylist()]
        version = f"hf:{HF_REPO}@{HF_FILES[(setting, split)]}"
    return Dataset(ex, DatasetKind.REAL, "hotpotqa", version, split, setting,
                   stats=compute_stats(ex))


def resolve_size(size: str | int | None) -> Optional[int]:
    if size is None:
        return None
    if isinstance(size, int):
        return size
    if size in SIZE_PRESETS:
        return SIZE_PRESETS[size]
    return int(size)


def load_dataset(cfg: DatasetConfig, download: bool = True) -> Dataset:
    """Main entry point: returns a validated, size-limited, provenance-tagged dataset.

    Order of preference: processed cache → local JSON → HF download → synthetic
    (only when ``cfg.allow_synthetic``; the fallback is logged loudly).
    """
    n = resolve_size(cfg.size)
    raw_dir, proc_dir = resolve_path(cfg.raw_dir), resolve_path(cfg.processed_dir)

    if cfg.force_synthetic:
        ds = synthetic_multihop(n or 200, seed=cfg.seed)
        validate_dataset(ds, strict=True)
        return ds

    cache = proc_dir / f"{cfg.name}_{cfg.setting}_{cfg.split}_full.json"
    try:
        if cache.exists():
            ds = Dataset.from_json(cache)
        else:
            ds = load_real_hotpotqa(raw_dir, cfg.setting, cfg.split, download=download)
            ds.to_json(cache)
    except Exception as exc:  # network / missing file
        if not cfg.allow_synthetic:
            raise
        log.warning("REAL DATASET UNAVAILABLE (%s). Falling back to SYNTHETIC data. "
                    "Results must NOT be reported as HotpotQA numbers.", exc)
        ds = synthetic_multihop(n or 200, seed=cfg.seed)
        validate_dataset(ds, strict=True)
        return ds

    report = validate_dataset(ds)
    if report["invalid"]:
        ds = Dataset([e for e in ds if not validate_example(e)], ds.kind, ds.name, ds.version,
                     ds.split, ds.setting)
        log.warning("dropped %d invalid examples", report["invalid"])
    return ds.subset(n, seed=cfg.seed)
