"""Safe, bounded page fetching and readable-text extraction."""
from __future__ import annotations

from html.parser import HTMLParser
import ipaddress
import socket
from typing import Optional
from urllib.parse import urlparse

import httpx


class _TextExtractor(HTMLParser):
    BLOCK = {"p", "div", "article", "section", "li", "h1", "h2", "h3", "h4", "br"}
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}

    def __init__(self) -> None:
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK and self.skip == 0:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK and self.skip == 0:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip == 0 and data.strip():
            self.parts.append(data)


def _public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
        for info in infos:
            addr = ipaddress.ip_address(info[4][0])
            if not addr.is_global:
                return False
        return True
    except (ValueError, OSError):
        return False


def validate_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("only http/https URLs are allowed")
    if p.username or p.password:
        raise ValueError("URLs with embedded credentials are not allowed")
    if p.hostname.lower() in {"localhost", "localhost.localdomain"} or not _public_host(p.hostname):
        raise ValueError("private or local destinations are not allowed")


def fetch_text(url: str, *, timeout_s: float = 10.0, max_bytes: int = 750_000) -> str:
    validate_url(url)
    with httpx.Client(timeout=timeout_s, follow_redirects=True, headers={"user-agent": "Nexamind.Ai/1.0"}) as client:
        response = client.get(url)
        response.raise_for_status()
        final = str(response.url)
        validate_url(final)
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ""
        raw = response.content[:max_bytes]
        if "text/plain" in content_type:
            return raw.decode(response.encoding or "utf-8", errors="replace")[:12000]
        parser = _TextExtractor()
        parser.feed(raw.decode(response.encoding or "utf-8", errors="replace"))
        text = " ".join(" ".join("".join(parser.parts).split()).split())
        return text[:12000]
