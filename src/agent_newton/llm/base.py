"""The provider interface.

One protocol, several backends. Agents depend on this and never on a particular
vendor's client, so which model serves a role is a config edit rather than a
code change.

Every call is **schema-enforced**: a caller names a pydantic model and receives
an instance of it or an exception. Free-text parsing never reaches an agent,
which matters because a small local model produces malformed output often enough
that handling it per-agent would be repetitive and easy to get wrong.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

#: Attempts at a single call before giving up. The retry carries the validation
#: error back to the model, so a second attempt is a repair rather than a
#: repetition.
MAX_ATTEMPTS = 3


class ProviderError(RuntimeError):
    """The provider could not be reached, or refused the request."""


class MalformedResponse(ProviderError):
    """The model replied, but not with something matching the schema."""


@dataclass(frozen=True, slots=True)
class Completion:
    """One raw response, before schema validation."""

    text: str
    model: str
    provider: str


@runtime_checkable
class LLMProvider(Protocol):
    """A backend that can be asked for structured output."""

    @property
    def label(self) -> str:
        """``provider/model``, as recorded in the run manifest."""
        ...

    def generate(self, prompt: str, schema: type[BaseModel], system: str | None) -> Completion:
        """Return raw text. Validation and retry are handled by the caller."""
        ...


def repair_prompt(prompt: str, bad: str, error: str, schema: type[BaseModel]) -> str:
    """Ask again, showing what was wrong.

    Handing back the actual validation error is what makes the second attempt a
    correction rather than a re-roll of the same dice.
    """
    return (
        f"{prompt}\n\n"
        f"Your previous reply could not be used.\n"
        f"You replied:\n{bad}\n\n"
        f"The problem was:\n{error}\n\n"
        f"Reply with JSON matching this schema exactly, and nothing else:\n"
        f"{json.dumps(schema.model_json_schema())}"
    )


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may be wrapped in prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    return stripped[start : end + 1] if 0 <= start < end else stripped


def complete(
    provider: LLMProvider,
    prompt: str,
    schema: type[T],
    *,
    system: str | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> T:
    """Ask for a structured reply, repairing malformed ones.

    Raises :class:`MalformedResponse` when every attempt fails, rather than
    returning a default. A silently-defaulted diagnosis would be recorded as a
    real inference and would corrupt the accuracy measurement.
    """
    current = prompt
    last_error = ""
    last_text = ""

    for attempt in range(1, max_attempts + 1):
        completion = provider.generate(current, schema, system)
        last_text = completion.text
        try:
            return schema.model_validate_json(_extract_json(completion.text))
        except ValidationError as exc:
            last_error = str(exc)
            log.debug(
                "malformed reply from %s (attempt %d/%d)",
                provider.label,
                attempt,
                max_attempts,
                extra={"event": "llm.malformed", "provider": provider.label},
            )
            current = repair_prompt(prompt, completion.text, last_error, schema)

    raise MalformedResponse(
        f"{provider.label} did not produce valid {schema.__name__} in "
        f"{max_attempts} attempts. Last reply: {last_text[:300]!r}. "
        f"Last error: {last_error[:300]}"
    )
