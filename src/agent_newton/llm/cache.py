"""On-disk response cache.

Keyed by everything that could change a reply: provider, model, system prompt,
user prompt, and the requested schema. Two consequences the project depends on:

* **Re-running an analysis costs nothing and returns the same numbers.** A
  cohort takes hours; re-deriving a figure from stored results must not.
* **A repeated call is genuinely repeated.** Sampling at temperature zero is not
  a guarantee across versions or hosts, so the cache is what actually makes a
  run reproducible.

The cache is deliberately *not* invalidated by time. An entry is wrong only if
one of its key components changed, in which case the key changed too.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pydantic import BaseModel

from agent_newton.llm.base import Completion, LLMProvider

log = logging.getLogger(__name__)


def cache_key(
    provider: str, model: str, prompt: str, system: str | None, schema: type[BaseModel]
) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "system": system,
            "prompt": prompt,
            # The schema is part of the request: the same prompt asking for a
            # different shape is a different call.
            "schema": schema.model_json_schema(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class CachedProvider:
    """Wraps a provider, storing every reply on disk.

    A decorator rather than a mixin, so caching is composable and every backend
    gets it without implementing anything.
    """

    def __init__(self, inner: LLMProvider, cache_dir: Path) -> None:
        self._inner = inner
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @property
    def label(self) -> str:
        return self._inner.label

    def _path(self, key: str) -> Path:
        # Sharded, because a cohort produces tens of thousands of entries and
        # one flat directory becomes slow to list.
        return self._dir / key[:2] / f"{key}.json"

    def generate(
        self, prompt: str, schema: type[BaseModel], system: str | None
    ) -> Completion:
        provider, _, model = self.label.partition("/")
        key = cache_key(provider, model, prompt, system, schema)
        path = self._path(key)

        if path.exists():
            self.hits += 1
            stored = json.loads(path.read_text())
            return Completion(
                text=stored["text"], model=stored["model"], provider=stored["provider"]
            )

        self.misses += 1
        completion = self._inner.generate(prompt, schema, system)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "text": completion.text,
                    "model": completion.model,
                    "provider": completion.provider,
                    # Stored for auditing, never read back as key material.
                    "prompt": prompt,
                    "system": system,
                },
                indent=2,
            )
        )
        return completion

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
