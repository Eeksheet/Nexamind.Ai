"""Small in-process conversation memory suitable for local development."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List
import uuid


@dataclass
class Conversation:
    id: str
    messages: List[Dict[str, str]] = field(default_factory=list)


class ConversationMemory:
    def __init__(self, max_conversations: int = 100, max_messages: int = 20) -> None:
        self.max_conversations = max_conversations
        self.max_messages = max_messages
        self._items: "OrderedDict[str, Conversation]" = OrderedDict()
        self._lock = Lock()

    def create(self) -> Conversation:
        with self._lock:
            cid = uuid.uuid4().hex
            c = Conversation(cid)
            self._items[cid] = c
            self._trim()
            return c

    def get_or_create(self, cid: str | None) -> Conversation:
        with self._lock:
            if cid and cid in self._items:
                c = self._items.pop(cid)
                self._items[cid] = c
                return c
            cid2 = uuid.uuid4().hex
            c = Conversation(cid2)
            self._items[cid2] = c
            self._trim()
            return c

    def add(self, cid: str, role: str, content: str) -> None:
        with self._lock:
            c = self._items.get(cid)
            if not c:
                return
            c.messages.append({"role": role, "content": content})
            c.messages = c.messages[-self.max_messages:]
            self._items.move_to_end(cid)

    def history(self, cid: str) -> List[Dict[str, str]]:
        with self._lock:
            c = self._items.get(cid)
            return list(c.messages) if c else []

    def recent(self, limit: int = 20) -> List[Dict[str, str]]:
        with self._lock:
            return [{"id": c.id, "title": next((m["content"] for m in c.messages if m["role"] == "user"), "New chat")[:80]}
                    for c in list(self._items.values())[-limit:][::-1]]

    def _trim(self) -> None:
        while len(self._items) > self.max_conversations:
            self._items.popitem(last=False)
