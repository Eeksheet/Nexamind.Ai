"""Seeding and experiment-manifest utilities."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every RNG the project touches (python, numpy, torch if present)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional (only needed for dense retrieval)
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def stable_hash(obj: Any, length: int = 12) -> str:
    """Deterministic hash of any JSON-serialisable object."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # pragma: no cover
        return None


@dataclass
class ExperimentManifest:
    """Everything needed to reproduce one experiment run.

    Written next to every results table as ``manifest.json``.
    """

    experiment: str
    seed: int
    dataset_name: str
    dataset_kind: str            # "real" | "synthetic"
    dataset_version: str
    dataset_size: int
    backend: str                 # "mock" | "anthropic" | "openai"
    model: str
    embedding_model: str
    retrieval_method: str
    alpha: float
    top_k: int
    reranker: bool
    max_hops: int
    prompt_version: str
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")
    git_revision: Optional[str] = field(default_factory=git_revision)
    python: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    config: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | os.PathLike[str]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return p
