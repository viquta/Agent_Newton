"""Ollama backend — the default, and the only one needing no API key.

Structured output is requested through Ollama's ``format`` parameter, which
constrains decoding to the JSON schema rather than merely asking for it in the
prompt. That removes most malformed replies at source; the repair loop in
:mod:`agent_newton.llm.base` covers what remains.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent_newton.llm.base import Completion, ProviderError

#: Deterministic decoding. Not a reproducibility guarantee on its own — that is
#: what the cache is for — but it removes the obvious source of variance.
TEMPERATURE = 0.0


class OllamaProvider:
    """A locally served model."""

    def __init__(self, model: str, host: str | None = None, timeout: float = 120.0) -> None:
        self._model = model
        self._timeout = timeout
        self._host = host
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
            self._client = ollama.Client(host=self._host) if self._host else ollama.Client()
        return self._client

    @retry(
        retry=retry_if_exception_type(ProviderError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def generate(
        self, prompt: str, schema: type[BaseModel], system: str | None
    ) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._connect().chat(
                model=self._model,
                messages=messages,
                # Constrained decoding: the reply is shaped by the schema rather
                # than merely requested to match it.
                format=schema.model_json_schema(),
                options={"temperature": TEMPERATURE},
            )
        except Exception as exc:  # the client raises a variety of errors
            raise ProviderError(f"ollama call failed for {self._model}: {exc}") from exc

        content = response.get("message", {}).get("content", "")
        if not content:
            raise ProviderError(f"ollama returned an empty reply for {self._model}")
        return Completion(text=content, model=self._model, provider="ollama")
