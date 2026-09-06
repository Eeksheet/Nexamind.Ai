"""Structured logging helpers."""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Mapping, Optional


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line (machine-readable experiment logs)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, Mapping):
            payload.update(extra)
        return json.dumps(payload, default=str)


def get_logger(name: str, level: int = logging.INFO, json_format: bool = False,
               stream: Optional[Any] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stderr)
        fmt: logging.Formatter
        if json_format:
            fmt = JsonFormatter()
        else:
            fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                                    "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def log_event(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Log ``msg`` with structured ``fields`` attached."""
    logger.info(msg, extra={"extra_fields": fields})
