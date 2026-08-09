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

#: Hard cap on tokens generated per call.
#:
#: Every reply here is a small JSON object — a label and a confidence, a concept
#: and a sentence, a two-sentence hint — so this is roughly five times the
#: longest legitimate reply. It exists because constrained decoding bounds the
#: *shape* of a reply and not its length: a schema with an array or a free
#: string permits an arbitrarily long one, and a model that starts repeating
#: itself will run to the context limit. One observed diagnosis decoded 8,485
#: tokens before it was interrupted, against a 30-token reply for the same
#: question elsewhere.
#:
#: A truncated reply fails schema validation and is handled by the repair loop
#: as a malformed response, which is counted and visible. That is the right
#: trade: a call that fails in seconds and is recorded beats one that stalls an
#: overnight run and is not.
MAX_TOKENS = 512

#: Seconds allowed for one call, including queueing behind another request.
#: Generous, because a cold model must load first.
REQUEST_TIMEOUT = 120.0


class OllamaProvider:
    """A locally served model."""

    def __init__(
        self, model: str, host: str | None = None, timeout: float = REQUEST_TIMEOUT
    ) -> None:
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
            # The timeout is applied here rather than per call: the client owns
            # the connection, and a request that never returns would otherwise
            # hold a cohort run open indefinitely.
            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._host:
                kwargs["host"] = self._host
            self._client = ollama.Client(**kwargs)
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
                options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
            )
        except Exception as exc:  # the client raises a variety of errors
            raise ProviderError(f"ollama call failed for {self._model}: {exc}") from exc

        content = response.get("message", {}).get("content", "")
        if not content:
            raise ProviderError(f"ollama returned an empty reply for {self._model}")
        return Completion(text=content, model=self._model, provider="ollama")
