"""Reply schemas, built per domain.

The valid answers are baked into the schema as a ``Literal`` over the domain's
own ids. With a backend that constrains decoding to the schema, a model
*cannot* return a label outside the catalogue or a concept outside the graph —
the failure mode is removed rather than validated after the fact.

That matters most for the diagnostic agent. A label outside the catalogue would
have to be discarded, and discarded predictions quietly bias an accuracy figure:
the cases a model finds hardest are exactly the ones where it invents something.

Schemas are cached per domain because building one walks the whole catalogue,
and a session asks for the same shape thousands of times.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, create_model

from agent_newton.domains.base import Domain

#: Returned when no catalogue entry fits. A real option, not a failure: a model
#: forced to choose always chooses, and a forced guess is worse than an
#: admission for both tutoring and measurement.
UNKNOWN = "unknown"

#: Candidates the diagnostic agent may list before choosing between them.
#:
#: Bounded, and that bound is load-bearing rather than tidiness. Constrained
#: decoding follows the schema's grammar, and an unbounded array is a grammar
#: that permits an arbitrarily long reply: a model that starts enumerating
#: labels has nothing to stop it, and one observed case decoded 8,485 tokens —
#: minutes of inference — before it was interrupted. A cap on the array makes
#: that unreachable; :data:`agent_newton.llm.ollama.MAX_TOKENS` is the backstop
#: for the fields a schema cannot bound.
MAX_CANDIDATES = 4


# Sized for one entry per concept plus the whole catalogue, since the label
# space offered is now per item. Too small a cache would rebuild a model class
# on nearly every diagnosis.
@lru_cache(maxsize=64)
def diagnosis_schema(domain_name: str, ids: tuple[str, ...]) -> type[BaseModel]:
    """Reply shape for the diagnostic agent, closed over the labels on offer."""
    labels = Literal[ids + (UNKNOWN,)]  # type: ignore[valid-type]
    return create_model(
        f"Diagnosis_{domain_name}",
        considered=(
            list[labels],  # type: ignore[valid-type]
            Field(
                default_factory=list,
                max_length=MAX_CANDIDATES,
                description=(
                    f"Up to {MAX_CANDIDATES} candidate misconceptions consistent "
                    f"with the step, before choosing between them."
                ),
            ),
        ),
        misconception_id=(
            labels,
            Field(description="The single best explanation, or 'unknown'."),
        ),
        confidence=(float, Field(ge=0.0, le=1.0, description="Confidence in [0, 1].")),
    )


@lru_cache(maxsize=8)
def plan_schema(domain_name: str, concept_ids: tuple[str, ...]) -> type[BaseModel]:
    """Reply shape for the planner, closed over this graph."""
    return create_model(
        f"Plan_{domain_name}",
        concept_id=(
            Literal[concept_ids],  # type: ignore[valid-type]
            Field(description="The concept to work on next."),
        ),
        reason=(str, Field(default="", description="One sentence of justification.")),
    )


class HintReply(BaseModel):
    """Reply shape for a hint. Domain-independent — it is prose."""

    text: str = Field(description="What to say to the learner. Two sentences at most.")


class LessonReply(BaseModel):
    """Reply shape for one turn of a lesson.

    ⚠️ Separate from :class:`HintReply` because of its *description*, which is
    the whole point. A field description goes into the JSON schema, and the
    schema is what Ollama constrains decoding against — so "Two sentences at
    most" is not documentation, it is an instruction the model obeys. A lesson
    turn asked for two or three sentences and a question, and the model stopped
    at "have you ever worked with the concept of", mid-sentence, because the
    schema told it to.

    That is the third instance of one defect. `_TUTOR_SYSTEM` once demanded two
    sentences globally while `WORKED_STEP` asked for the step to be worked
    through, and the fix then was to move the length budget from the system
    prompt to the level. It survived here, one layer further down, where nothing
    reads like a length budget at all.
    """

    text: str = Field(
        description=(
            "What to say to the student: a little, and then one question they "
            "can answer. A few sentences."
        )
    )

    @field_validator("text")
    @classmethod
    def _must_be_a_finished_thought(cls, text: str) -> str:
        """Reject a turn that stops mid-sentence.

        ⚠️ Observed, not hypothetical. One opening came back as *"...To start
        off, have you ever worked with the concept of "* — valid JSON, correct
        shape, schema-clean, and cut off mid-phrase. It reached the learner
        looking like a question that had been asked.

        Deterministic at temperature zero, so it recurred identically on every
        re-run: this is not noise that a retry outruns. It is one bad
        generation for one prompt, and the thing that makes it dangerous is that
        nothing downstream could tell — it is the silent-failure shape this
        project keeps finding, in the one place a learner reads directly.

        Raising here rather than checking at the call site is deliberate: a
        `ValidationError` is what `complete()`'s repair loop already handles, so
        the model is shown its own truncated reply and asked again, and a
        provider that cannot manage it falls back to the authored account
        through the existing `ProviderError` path. No new machinery, and the
        failure is counted where every other malformed reply is counted.

        Trailing quotes and brackets are stripped before the test, so a turn
        ending in a quoted phrase or a parenthesis is not called truncated.
        """
        trimmed = text.rstrip().rstrip(')"\'’”')
        if not trimmed.endswith((".", "?", "!", ":")):
            raise ValueError(
                f"the reply stops mid-sentence: ...{text.rstrip()[-40:]!r}. "
                f"Finish the sentence, and end with the question you are putting "
                f"to the student."
            )
        return text


class ClosingReply(LessonReply):
    """The last turn of a lesson. Must answer, and must not ask.

    ⚠️ Its own schema because the instruction alone did not hold. Asked to stop
    asking, the model asked anyway — the system prompt said "say a little and
    then ask" on every turn, and a rule a model can talk itself out of is not
    one. The prompt conflict is fixed; this is what makes the rule checkable
    rather than hoped for.

    Only a *trailing* question is refused. A closing turn may perfectly well
    contain one — "you asked what happens as it slides in: it settles on a
    single value" — and what must not happen is that it ends on something the
    learner has no way to reply to.
    """

    @field_validator("text")
    @classmethod
    def _must_not_ask(cls, text: str) -> str:
        if text.rstrip().endswith("?"):
            raise ValueError(
                "this is the last turn and it ends on a question the student "
                "cannot reply to. Answer what was left hanging instead, and "
                "stop."
            )
        return text


class ConfusionReply(BaseModel):
    """Whether a learner's own words say they do not understand the concept.

    Deliberately narrow, and the narrowness is the point. It is **not** asked
    whether the learner is struggling, whether they are frustrated, or whether
    they need help — those are judgements about a person. It is asked one
    question about a piece of text: does it say the concept itself is not
    understood, as opposed to showing a mistake in applying it?

    Those are different things and the distinction is what makes the trigger
    worth having. Someone who differentiates wrongly has met the concept and
    slipped; someone who writes "I don't understand what a limit is" has not met
    it, and no amount of correcting the slip will help them.
    """

    confused: bool = Field(
        description=(
            "True only if the student says they do not know what the concept "
            "is or means. False if they are attempting it and getting it wrong."
        )
    )
    quote: str = Field(
        default="",
        description="The words that say so, copied exactly. Empty if false.",
    )


def schemas_for(domain: Domain) -> dict[str, Any]:
    return {
        "diagnosis": diagnosis_schema(domain.name, tuple(domain.misconceptions.ids())),
        "plan": plan_schema(domain.name, tuple(domain.concepts.ids())),
        "hint": HintReply,
    }
