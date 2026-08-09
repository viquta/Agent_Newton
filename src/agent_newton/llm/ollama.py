"""Ollama backend — the default, and the only one needing no API key.

Structured output is requested through Ollama's ``format`` parameter, which
constrains decoding to the JSON schema rather than merely asking for it in the
prompt. That removes most malformed replies at source; the repair loop in
:mod:`agent_newton.llm.base` covers what remains.

**Reasoning models answer in two channels.** A model that deliberates writes to
``message.thinking`` and only fills ``message.content`` once it has finished. An
empty ``content`` is therefore not necessarily a failed call: it is what a
truncated deliberation looks like from outside. The two are distinguished here,
because they call for different handling — one is worth retrying, the other is
the same every time and belongs to the repair loop.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent_newton.llm.base import Completion, MalformedResponse, ProviderError

#: Deterministic decoding. Not a reproducibility guarantee on its own — that is
#: what the cache is for — but it removes the obvious source of variance.
TEMPERATURE = 0.0

#: Hard cap on tokens generated per call, thinking included.
#:
#: Every reply here is a small JSON object — a label and a confidence, a concept
#: and a sentence, a two-sentence hint — so this is many times the longest
#: legitimate answer. It exists because constrained decoding bounds the *shape*
#: of a reply and not its length: a reasoning model can deliberate without
#: bound before emitting anything at all, and one observed diagnosis decoded
#: 8,485 tokens without producing an answer.
#:
#: Exhausting the budget yields a reply that fails schema validation, which the
#: repair loop handles as a malformed response: counted, visible, and over in
#: seconds. That is the right trade — a call that fails quickly and is recorded
#: beats one that stalls an overnight run and is not.
MAX_TOKENS = 1024

#: Seconds allowed for one call, including queueing behind another request.
#: Generous, because a cold model must load first.
REQUEST_TIMEOUT = 120.0


class OllamaProvider:
    """A locally served model."""

    def __init__(
        self,
        model: str,
        host: str | None = None,
        timeout: float = REQUEST_TIMEOUT,
        think: bool | None = None,
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._host = host
        #: Whether the model deliberates before answering. ``None`` leaves the
        #: server's default alone, which is what a model with no reasoning mode
        #: needs.
        self._think = think
        self._client: Any = None

    @property
    def label(self) -> str:
        # The reasoning mode is part of the label because the response cache is
        # keyed on it. The same question asked of a deliberating model and a
        # direct one is not the same call, and a cached reply from one must not
        # be served for the other.
        if self._think is None:
            return f"ollama/{self._model}"
        return f"ollama/{self._model}+think={str(self._think).lower()}"

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
        # A malformed reply is excluded: temperature is zero, so asking the same
        # question again gives the same answer. Retrying it here would spend the
        # budget three times over before the repair loop — which changes the
        # prompt — ever sees it.
        retry=retry_if_exception_type(ProviderError)
        & retry_if_not_exception_type(MalformedResponse),
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

        call: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            # Constrained decoding: the reply is shaped by the schema rather
            # than merely requested to match it.
            "format": schema.model_json_schema(),
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        }
        if self._think is not None:
            call["think"] = self._think

        try:
            response = self._connect().chat(**call)
        except Exception as exc:  # the client raises a variety of errors
            raise ProviderError(f"ollama call failed for {self._model}: {exc}") from exc

        message = response.get("message", {}) or {}
        content = message.get("content", "")
        if content:
            return Completion(text=content, model=self._model, provider="ollama")

        # No answer. Which failure it is decides who handles it.
        if message.get("thinking") or response.get("done_reason") == "length":
            raise MalformedResponse(
                f"{self._model} spent its {MAX_TOKENS}-token budget without "
                f"producing an answer. Reasoning models deliberate before they "
                f"reply; set think=False for this role, or raise MAX_TOKENS."
            )
        raise ProviderError(f"ollama returned an empty reply for {self._model}")
