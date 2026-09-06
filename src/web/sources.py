"""Source normalization and prompt-safe web evidence."""
from __future__ import annotations

from typing import Iterable, List, Dict

from .search import SearchResult


def normalize_sources(results: Iterable[SearchResult], limit: int = 5) -> List[Dict[str, str]]:
    out = []
    seen = set()
    for r in results:
        if not r.url or r.url in seen:
            continue
        seen.add(r.url)
        out.append({"title": r.title, "url": r.url, "domain": r.domain, "snippet": r.snippet})
        if len(out) >= limit:
            break
    return out


def format_web_evidence(sources: List[Dict[str, str]], fetched: Dict[str, str] | None = None,
                        max_chars: int = 14000) -> str:
    chunks = []
    total = 0
    fetched = fetched or {}
    for i, s in enumerate(sources):
        content = fetched.get(s["url"], "")
        block = f"[{i}] {s['title']} ({s['domain']})\nURL: {s['url']}\nSNIPPET: {s['snippet']}"
        if content:
            block += f"\nPAGE TEXT: {content[:5000]}"
        if total + len(block) > max_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n\n".join(chunks) if chunks else "(no web evidence)"
