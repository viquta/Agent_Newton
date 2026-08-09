"""Run directories.

One directory per run, holding the manifest, the JSONL event log, and the
metrics the analysis stage reads back. Run ids sort chronologically and carry
the config hash, so a directory listing is already a readable experiment log.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agent_newton.config import Config


def make_run_id(config: Config) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}_{config.run_name}_{config.arm}_{config.content_hash()[:8]}"


def new_run_dir(config: Config, run_id: str | None = None) -> tuple[str, Path]:
    """Create and return this run's directory."""
    run_id = run_id or make_run_id(config)
    run_dir = config.paths.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir
