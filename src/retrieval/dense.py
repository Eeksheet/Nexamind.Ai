"""Dense (embedding) retrieval with a disk-backed embedding cache.

The encoder is loaded lazily and shared process-wide so that thousands of
per-question indexes reuse one model.  Embeddings are cached by
``sha256(model_name + text)`` in a memory-mapped ``.npy`` store per model.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)

_MODEL_LOCK = threading.Lock()
_MODELS: Dict[str, object] = {}


def get_encoder(model_name: str):
    """Return a (cached) sentence-transformers model."""
    with _MODEL_LOCK:
        if model_name not in _MODELS:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model %s", model_name)
            _MODELS[model_name] = SentenceTransformer(model_name, device="cpu")
        return _MODELS[model_name]


class EmbeddingCache:
    """Append-only key → vector cache persisted as ``<dir>/<model>.npz``."""

    def __init__(self, cache_dir: Optional[Path], model_name: str) -> None:
        self.path = (cache_dir / (model_name.replace("/", "__") + ".npz")) if cache_dir else None
        self._mem: Dict[str, np.ndarray] = {}
        self._dirty = 0
        if self.path and self.path.exists():
            try:
                with np.load(self.path, allow_pickle=False) as z:
                    keys = z["keys"].tolist()
                    vecs = z["vecs"]
                self._mem = {k: vecs[i] for i, k in enumerate(keys)}
            except Exception as exc:  # corrupt cache: start fresh
                log.warning("could not read embedding cache %s (%s)", self.path, exc)

    @staticmethod
    def key(model_name: str, text: str) -> str:
        return hashlib.sha256((model_name + "\x00" + text).encode("utf-8")).hexdigest()[:24]

    def get(self, k: str) -> Optional[np.ndarray]:
        return self._mem.get(k)

    def put(self, k: str, v: np.ndarray) -> None:
        self._mem[k] = v.astype(np.float32)
        self._dirty += 1

    def flush(self, min_dirty: int = 1) -> None:
        if not self.path or self._dirty < min_dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._mem.keys()))
        vecs = np.stack(list(self._mem.values())) if self._mem else np.zeros((0, 0), np.float32)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, vecs=vecs)
        tmp.replace(self.path)
        self._dirty = 0

    def __len__(self) -> int:
        return len(self._mem)


_CACHES: Dict[str, EmbeddingCache] = {}


def get_cache(cache_dir: Optional[Path], model_name: str) -> EmbeddingCache:
    key = f"{cache_dir}|{model_name}"
    if key not in _CACHES:
        _CACHES[key] = EmbeddingCache(cache_dir, model_name)
    return _CACHES[key]


class DenseEncoder:
    """Batch-embeds texts with caching; returns L2-normalised float32 matrices."""

    def __init__(self, model_name: str, cache_dir: Optional[Path] = None,
                 batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache = get_cache(cache_dir, model_name)
        self.n_encoded = 0

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        keys = [self.cache.key(self.model_name, t) for t in texts]
        out: List[Optional[np.ndarray]] = [self.cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            model = get_encoder(self.model_name)
            vecs = model.encode([texts[i] for i in missing], batch_size=self.batch_size,
                                normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)
            self.n_encoded += len(missing)
            for i, v in zip(missing, vecs):
                out[i] = np.asarray(v, dtype=np.float32)
                self.cache.put(keys[i], out[i])
        return np.stack(out)  # type: ignore[arg-type]

    def flush(self) -> None:
        self.cache.flush()


class DenseIndex:
    """Cosine-similarity index over one corpus."""

    def __init__(self, texts: Sequence[str], encoder: DenseEncoder) -> None:
        self.encoder = encoder
        self.M = encoder.encode(texts)

    def score(self, query: str) -> np.ndarray:
        q = self.encoder.encode([query], is_query=True)[0]
        return self.M @ q

    def top_k(self, query: str, k: int) -> List[int]:
        return list(np.argsort(-self.score(query), kind="stable")[:k])
