"""Structured logging.

Two sinks: a human-readable console stream, and a JSONL file under the run
directory. The JSONL sink is the durable one — every arbitration decision,
replanning trigger and state transition lands there in machine-readable form, so
the audit log can be reconstructed and analysed after the fact without re-running
anything.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

_EVENT_KEY = "event"

# LogRecord attributes present on every record; anything else was passed by the
# caller via `extra=` and is part of the structured payload.
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class JSONLFormatter(logging.Formatter):
    """One JSON object per line, carrying any ``extra=`` fields verbatim."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(run_dir: Path | None = None, level: int = logging.INFO) -> None:
    """Configure the root logger. Idempotent — safe to call per run."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = RichHandler(rich_tracebacks=True, show_path=False, log_time_format="%H:%M:%S")
    console.setLevel(level)
    root.addHandler(console)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "events.jsonl", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONLFormatter())
        root.addHandler(file_handler)


def log_event(logger: logging.Logger, event: str, message: str = "", /, **fields: Any) -> None:
    """Emit a structured event.

    ``event`` is the machine-readable discriminator (``"replan.triggered"``,
    ``"state.updated"``); ``message`` is for the human reading the console.
    Analysis filters on ``event``, so keep the names stable.
    """
    logger.info(message or event, extra={_EVENT_KEY: event, **fields})
