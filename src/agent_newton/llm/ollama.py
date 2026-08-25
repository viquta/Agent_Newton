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

**Three limits bound a call, and they are not interchangeable.** Making a
deliberating model viable needs all three moved together; raising one alone
relocates the failure rather than removing it.

===============  ====================================================
``num_predict``  tokens *generated* — the deliberation and the answer
``num_ctx``      the *context window* — prompt, deliberation and answer
                 together
``timeout``      wall-clock allowed for the call
===============  ====================================================

``num_ctx`` is the one that fails quietly. When a call exceeds it the server
drops the oldest context rather than refusing, so the model answers from a
prompt it was never fully shown — and there is nothing in the reply to say so.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agent_newton.llm.base import (
    Completion,
    MalformedResponse,
    ProviderError,
    ProviderTimeout,
)

log = logging.getLogger(__name__)

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

#: Characters per token, for the truncation warning only.
#:
#: A rough English average, and deliberately not a tokeniser: loading the
#: model's own vocabulary to decide whether to log a warning would cost more
#: than the warning is worth. It is used to decide when to *look*, never to
#: decide what to send.
_CHARS_PER_TOKEN = 4

#: Share of ``num_ctx`` a prompt may reach before it is worth warning about.
#:
#: Below 1.0 because the window holds the reply as well as the prompt: a prompt
#: at 90% of the window leaves a tenth of it for the deliberation and the answer,
#: which is a truncation that has not happened yet rather than one that has not
#: happened.
_CONTEXT_WARN_AT = 0.8


def _looks_like_a_timeout(exc: BaseException) -> bool:
    """Whether this exception is the call running out of time.

    Decided on the exception's class name rather than by importing the client's
    HTTP library. The ollama package has changed transport before, and a check
    that breaks silently when it changes again would put the retry back to three
    long waits without anything failing to say so.
    """
    return any("timeout" in klass.__name__.lower() for klass in type(exc).__mro__)


class OllamaProvider:
    """A locally served model."""

    def __init__(
        self,
        model: str,
        host: str | None = None,
        timeout: float | None = None,
        think: bool | None = None,
        max_tokens: int | None = None,
        context_tokens: int | None = None,
    ) -> None:
        self._model = model
        self._host = host
        #: Whether the model deliberates before answering. ``None`` leaves the
        #: server's default alone, which is what a model with no reasoning mode
        #: needs.
        self._think = think
        #: The three limits, kept as passed so :attr:`label` can tell "left
        #: alone" from "set to the same value". What the call actually sends is
        #: resolved below.
        self._configured_timeout = timeout
        self._configured_max_tokens = max_tokens
        self._configured_context = context_tokens

        self._timeout = REQUEST_TIMEOUT if timeout is None else timeout
        self._max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
        #: No default. Today nothing sets ``num_ctx``, so leaving it unset is
        #: what keeps every existing run byte-identical — the server applies its
        #: own, whatever that is for this model.
        self._context_tokens = context_tokens
        self._client: Any = None

    @property
    def label(self) -> str:
        """``provider/model``, plus whatever was configured about the call.

        The response cache keys on this, so anything that changes *what the
        model saw or how much it could say* has to appear here or a reply from
        one configuration will be served for another. The reasoning mode was the
        first: the same question asked of a deliberating model and a direct one
        is not the same call.

        ``num_predict`` and ``num_ctx`` join it for that reason. ``timeout`` does
        **not** — how long the caller was willing to wait does not change what
        came back, and putting it here would throw away every cached reply over a
        knob with no effect on content.

        Each part appears only when it was explicitly configured, so a provider
        built the way every previous run built one produces the label those runs
        recorded, and their cache entries stay valid.
        """
        label = f"ollama/{self._model}"
        if self._think is not None:
            label += f"+think={str(self._think).lower()}"
        if self._configured_max_tokens is not None:
            label += f"+max_tokens={self._configured_max_tokens}"
        if self._configured_context is not None:
            label += f"+ctx={self._configured_context}"
        return label

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

    def _warn_if_close_to_the_window(self, prompt: str, system: str | None) -> None:
        """Say so when a prompt is approaching the context window.

        Ollama drops the oldest context instead of refusing, so a prompt that
        does not fit produces a confident reply to a question the model was only
        partly shown. Nothing in the response says which part went missing, and
        the reply validates against the schema like any other — which is the
        silent-failure shape this project keeps finding.

        Only reached when ``num_ctx`` was configured. With the server's own
        default there is no number here to compare against, and inventing one
        would produce warnings that mean nothing.
        """
        if self._context_tokens is None:
            return
        estimated = (len(prompt) + len(system or "")) // _CHARS_PER_TOKEN
        if estimated < self._context_tokens * _CONTEXT_WARN_AT:
            return
        log.warning(
            "prompt for %s is roughly %d tokens against a %d-token context "
            "window; the window holds the reply too, and Ollama truncates "
            "silently rather than refusing",
            self._model,
            estimated,
            self._context_tokens,
            extra={
                "event": "llm.context_pressure",
                "model": self._model,
                "estimated_tokens": estimated,
                "num_ctx": self._context_tokens,
            },
        )

    @retry(
        # Two exclusions, and they are excluded for the same reason.
        #
        # A malformed reply: temperature is zero, so asking the same question
        # again gives the same answer. Retrying here would spend the budget three
        # times over before the repair loop — which changes the prompt — ever
        # sees it.
        #
        # A timeout: likewise deterministic. The identical question takes the
        # identical time, so three attempts turn one wait into three and end in
        # the same place. That is felt hardest exactly where the timeout is
        # longest, which is the deliberating-model case this budget exists for.
        retry=retry_if_exception_type(ProviderError)
        & retry_if_not_exception_type((MalformedResponse, ProviderTimeout)),
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

        self._warn_if_close_to_the_window(prompt, system)

        options: dict[str, Any] = {
            "temperature": TEMPERATURE,
            "num_predict": self._max_tokens,
        }
        if self._context_tokens is not None:
            # Sent only when configured. Passing the server's own default back
            # to it would be harmless but would make every run look as though it
            # had chosen a window, which is the sort of thing a manifest is read
            # for.
            options["num_ctx"] = self._context_tokens

        call: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            # Constrained decoding: the reply is shaped by the schema rather
            # than merely requested to match it.
            "format": schema.model_json_schema(),
            "options": options,
        }
        if self._think is not None:
            call["think"] = self._think

        try:
            response = self._connect().chat(**call)
        except Exception as exc:  # the client raises a variety of errors
            if _looks_like_a_timeout(exc):
                raise ProviderTimeout(
                    f"{self._model} did not answer within {self._timeout:.0f}s. "
                    f"Deliberation is the usual cause: raise timeout_seconds for "
                    f"this role, or set think=False."
                ) from exc
            raise ProviderError(f"ollama call failed for {self._model}: {exc}") from exc

        message = response.get("message", {}) or {}
        content = message.get("content", "")
        if content:
            return Completion(text=content, model=self._model, provider="ollama")

        # No answer. Which failure it is decides who handles it.
        if message.get("thinking") or response.get("done_reason") == "length":
            raise MalformedResponse(
                f"{self._model} spent its {self._max_tokens}-token budget without "
                f"producing an answer. Reasoning models deliberate before they "
                f"reply; set think=False for this role, or raise max_tokens — and "
                f"raise context_tokens with it, since the window bounds the "
                f"prompt and the deliberation together."
            )
        raise ProviderError(f"ollama returned an empty reply for {self._model}")
