"""Anthropic and OpenAI backends. vh comment: HAVE NOT TESTED, cause I don't have the API keys. 

Both SDKs are optional extras, so an offline install stays lean and a run that
names neither never imports them. Import failure is reported as a missing
extra rather than a stack trace, because "you did not install this" and "the
service is down" need different responses.

The escape hatch from local-model quality: switching a role to a hosted model is
a config edit, and the run manifest records which was used.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from agent_newton.llm.base import Completion, ProviderError

TEMPERATURE = 0.0
MAX_TOKENS = 2048


def _schema_instruction(schema: type[BaseModel]) -> str:
    import json

    return (
        "Reply with JSON matching this schema exactly, and nothing else:\n"
        f"{json.dumps(schema.model_json_schema())}"
    )


class AnthropicProvider:
    """Claude, via the Anthropic SDK. Install with the ``anthropic`` extra."""

    def __init__(self, model: str, timeout: float = 120.0) -> None:
        self._model = model
        self._timeout = timeout
        self._client: Any = None

    @property
    def label(self) -> str:
        return f"anthropic/{self._model}"

    def _connect(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # pyright: ignore[reportMissingImports]
            except ImportError as exc:
                raise ProviderError(
                    "the anthropic extra is not installed; "
                    "run `uv sync --extra anthropic`"
                ) from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ProviderError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic(timeout=self._timeout)
        return self._client

    def generate(
        self, prompt: str, schema: type[BaseModel], system: str | None
    ) -> Completion:
        instruction = _schema_instruction(schema)
        try:
            response = self._connect().messages.create(
                model=self._model,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=f"{system}\n\n{instruction}" if system else instruction,
                messages=[{"role": "user", "content": prompt}],
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"anthropic call failed for {self._model}: {exc}") from exc

        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        if not text:
            raise ProviderError(f"anthropic returned an empty reply for {self._model}")
        return Completion(text=text, model=self._model, provider="anthropic")


class OpenAIProvider:
    """GPT models, via the OpenAI SDK. Install with the ``openai`` extra."""

    def __init__(self, model: str, timeout: float = 120.0) -> None:
        self._model = model
        self._timeout = timeout
        self._client: Any = None

    @property
    def label(self) -> str:
        return f"openai/{self._model}"

    def _connect(self) -> Any:
        if self._client is None:
            try:
                import openai  # pyright: ignore[reportMissingImports]
            except ImportError as exc:
                raise ProviderError(
                    "the openai extra is not installed; run `uv sync --extra openai`"
                ) from exc
            if not os.environ.get("OPENAI_API_KEY"):
                raise ProviderError("OPENAI_API_KEY is not set")
            self._client = openai.OpenAI(timeout=self._timeout)
        return self._client

    def generate(
        self, prompt: str, schema: type[BaseModel], system: str | None
    ) -> Completion:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._connect().chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"openai call failed for {self._model}: {exc}") from exc

        text = response.choices[0].message.content or ""
        if not text:
            raise ProviderError(f"openai returned an empty reply for {self._model}")
        return Completion(text=text, model=self._model, provider="openai")
