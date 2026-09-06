"""Pluggable web-search providers for Nexamind.Ai.

DuckDuckGo HTML is the zero-key development default. Google Programmable Search
(JSON API) is supported when GOOGLE_API_KEY and GOOGLE_CSE_ID are configured.
"""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import os
from typing import List, Protocol
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    domain: str


class SearchProvider(Protocol):
    name: str
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]: ...


class _DDGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: List[SearchResult] = []
        self._title = ""
        self._url = ""
        self._snippet = ""
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")
        if tag == "a" and "result__a" in cls:
            self._finalize()
            self._in_title = True
            self._title = ""
            self._url = self._unwrap(a.get("href", ""))
        elif "result__snippet" in cls and tag in ("a", "div"):
            self._in_snippet = True
            self._snippet = ""

    def handle_endtag(self, tag):
        if self._in_title and tag == "a":
            self._in_title = False
        if self._in_snippet and tag in ("a", "div"):
            self._in_snippet = False

    def handle_data(self, data: str):
        if self._in_title:
            self._title += data
        elif self._in_snippet:
            self._snippet += data

    def close(self):
        super().close()
        self._finalize()

    @staticmethod
    def _unwrap(href: str) -> str:
        if not href:
            return ""
        try:
            q = parse_qs(urlparse(href).query)
            if q.get("uddg"):
                return unquote(q["uddg"][0])
        except Exception:
            pass
        return href

    def _finalize(self):
        url = self._url.strip()
        title = " ".join(self._title.split())
        if not title or not url.startswith(("http://", "https://")):
            return
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        if not any(r.url == url for r in self.results):
            self.results.append(SearchResult(title[:300], url, " ".join(self._snippet.split())[:500], domain))


class DuckDuckGoProvider:
    name = "duckduckgo"

    def __init__(self, timeout_s: float = 12.0) -> None:
        self.timeout_s = timeout_s
        self.headers = {"user-agent": "Nexamind.Ai/1.0"}

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not query.strip():
            return []
        url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
        with httpx.Client(timeout=self.timeout_s, headers=self.headers, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
        parser = _DDGParser()
        parser.feed(response.text)
        return parser.results[:max_results]


class GoogleProgrammableSearchProvider:
    name = "google"
    endpoint = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str, cx: str, timeout_s: float = 12.0) -> None:
        self.api_key = api_key
        self.cx = cx
        self.timeout_s = timeout_s

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not query.strip():
            return []
        params = {"key": self.api_key, "cx": self.cx, "q": query, "num": min(max_results, 10)}
        with httpx.Client(timeout=self.timeout_s, headers={"user-agent": "Nexamind.Ai/1.0"}) as client:
            response = client.get(self.endpoint, params=params)
            response.raise_for_status()
        data = response.json()
        out: List[SearchResult] = []
        for item in data.get("items", []):
            url = str(item.get("link", ""))
            if not url.startswith(("http://", "https://")):
                continue
            out.append(SearchResult(
                title=str(item.get("title", ""))[:300],
                url=url,
                snippet=str(item.get("snippet", ""))[:500],
                domain=urlparse(url).netloc.lower().removeprefix("www."),
            ))
        return out[:max_results]


def get_search_provider() -> SearchProvider | None:
    provider = os.environ.get("WEB_SEARCH_PROVIDER", "duckduckgo").strip().lower()
    if provider in ("", "none", "disabled", "off"):
        return None
    timeout = float(os.environ.get("WEB_SEARCH_TIMEOUT_S", "12"))
    if provider in ("duckduckgo", "ddg"):
        return DuckDuckGoProvider(timeout)
    if provider in ("google", "google_cse", "google_programmable_search"):
        key = os.environ.get("GOOGLE_API_KEY", "").strip()
        cx = os.environ.get("GOOGLE_CSE_ID", "").strip()
        if not key or not cx:
            raise RuntimeError("GOOGLE_API_KEY and GOOGLE_CSE_ID are required for WEB_SEARCH_PROVIDER=google")
        return GoogleProgrammableSearchProvider(key, cx, timeout)
    raise ValueError(f"unsupported WEB_SEARCH_PROVIDER={provider!r}; use duckduckgo, google, or none")
