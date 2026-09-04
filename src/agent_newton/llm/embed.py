"""Embeddings, and the cache that makes them reproducible.

The same discipline the response cache follows, for the same reason: a run must
cost nothing to repeat and must return what it returned before. Keyed on the
model *and* the text, so a different model is a different entry rather than a
silently reused one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from agent_newton.llm.base import ProviderError, ProviderTimeout


class OllamaEmbedder:
    """A locally served embedding model."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._host = host
        self._timeout = timeout
        self._client: Any = None

    @property
    def label(self) -> str:
        return f"ollama/{self._model}"

    def _connect(self) -> Any:
        if self._client is None:
            try:
                import ollama
            except ImportError as exc:  # pragma: no cover
                raise ProviderError("the 'ollama' package is not installed") from exc
            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._host:
                kwargs["host"] = self._host
            self._client = ollama.Client(**kwargs)
        return self._client

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return ()
        try:
            reply = self._connect().embed(model=self._model, input=list(texts))
        except Exception as exc:
            if any("timeout" in k.__name__.lower() for k in type(exc).__mro__):
                raise ProviderTimeout(
                    f"{self._model} did not embed within {self._timeout:.0f}s"
                ) from exc
            raise ProviderError(f"ollama embed failed for {self._model}: {exc}") from exc
        vectors = reply.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise ProviderError(
                f"{self._model} returned {len(vectors or ())} vectors for "
                f"{len(texts)} texts"
            )
        return vectors


class CachedEmbedder:
    """Wraps an embedder, storing every vector on disk.

    A decorator rather than a mixin, so caching is composable and any backend
    gets it — the shape :class:`~agent_newton.llm.cache.CachedProvider` already
    uses.

    ⚠️ Cached per *text*, not per batch. A corpus grows by one utterance between
    one sitting and the next, and a batch-keyed cache would re-embed the whole
    thing every time — which would make an "affordable" exact scan cost a model
    call per stored word, forever.
    """

    def __init__(self, inner, cache_dir: Path) -> None:  # noqa: ANN001
        self._inner = inner
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @property
    def label(self) -> str:
        return self._inner.label

    def _path(self, text: str) -> Path:
        key = hashlib.sha256(f"{self.label}|{text}".encode()).hexdigest()
        # Sharded, like the response cache: a learner's whole history plus every
        # query is a lot of small files in one directory otherwise.
        return self._dir / key[:2] / f"{key}.json"

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        found: dict[int, Sequence[float]] = {}
        missing: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            path = self._path(text)
            if path.exists():
                self.hits += 1
                found[index] = json.loads(path.read_text())
            else:
                self.misses += 1
                missing.append((index, text))

        if missing:
            fresh = self._inner.embed([text for _, text in missing])
            for (index, text), vector in zip(missing, fresh):
                found[index] = vector
                path = self._path(text)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(list(vector)))

        return [found[index] for index in range(len(texts))]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
