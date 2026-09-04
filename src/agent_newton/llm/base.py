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


class BudgetExhausted(MalformedResponse):
    """The model spent its whole token budget without producing an answer.

    A subclass of :class:`MalformedResponse` because that is what it is — no
    reply came back that could be validated — but it is **not repaired**, and
    the difference is measured rather than assumed.

    ⚠️ The repair loop existed here on the reasoning that an exhausted budget
    "yields a reply that fails schema validation, counted, visible, and over in
    seconds". The first half is right and the last is not. One case measured at
    three budgets, with the context window kept clear each time:

    ========  =========  ==========
    budget    seconds    answer
    ========  =========  ==========
    4096      547        none
    8192      1114       none
    16384     2301       none
    ========  =========  ==========

    The cost doubles exactly with the budget — ratios 2.04 and 2.06 — and the
    outcome never changes. A model that deliberates without converging fills
    whatever room it is given, so asking again cannot help: decoding is
    deterministic, and the repair prompt is *longer*, leaving even less room
    than the attempt that just failed. Three attempts turned a thirteen-minute
    dead end into thirty-eight.

    Same reasoning as :class:`ProviderTimeout`, one layer up. That one is
    excluded from the transport retry; this is excluded from the repair loop.
    """


class ProviderTimeout(ProviderError):
    """The call did not finish inside the time allowed.

    Distinct from a plain :class:`ProviderError` for the same reason
    :class:`MalformedResponse` is: it decides whether asking again is worth
    anything. A dropped connection is worth retrying — the next attempt may find
    the server. A timeout is not: decoding is deterministic at temperature zero,
    so the identical question takes the identical time and fails the identical
    way, and three attempts turn one wait into three.

    That matters most where the wait is longest. A deliberating model given a
    generous ``timeout`` is exactly the case where a silent triple retry is felt
    as the system having hung.

    It stays a ``ProviderError`` so every existing handler keeps working:
    ``LLMDiagnostic`` counts it as a failure to infer, ``LLMTutor`` falls back to
    a fixed hint, and the sitting survives a dead backend. Only the retry
    predicate treats it specially.
    """


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
        try:
            completion = provider.generate(current, schema, system)
            last_text = completion.text
            return schema.model_validate_json(_extract_json(completion.text))
        except BudgetExhausted:
            # Before the repair clause, deliberately: this *is* a malformed
            # response and would be caught by it. Asking again cannot help — see
            # the class. Raised on rather than counted here, so the caller
            # records it exactly as it records every other failure to get an
            # answer, and only the wasted attempts are gone.
            raise
        except (ValidationError, MalformedResponse) as exc:
            # A provider may reject its own reply before this sees it — a
            # deliberation that never reached an answer produces no text to
            # validate. It is the same failure and takes the same repair.
            last_error = str(exc)
            log.debug(
                "malformed reply from %s (attempt %d/%d)",
                provider.label,
                attempt,
                max_attempts,
                extra={"event": "llm.malformed", "provider": provider.label},
            )
            current = repair_prompt(prompt, last_text, last_error, schema)

    raise MalformedResponse(
        f"{provider.label} did not produce valid {schema.__name__} in "
        f"{max_attempts} attempts. Last reply: {last_text[:300]!r}. "
        f"Last error: {last_error[:300]}"
    )
