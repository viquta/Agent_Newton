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

from pydantic import BaseModel, Field, create_model

from agent_newton.domains.base import Domain

#: Returned when no catalogue entry fits. A real option, not a failure: a model
#: forced to choose always chooses, and a forced guess is worse than an
#: admission for both tutoring and measurement.
UNKNOWN = "unknown"


@lru_cache(maxsize=8)
def diagnosis_schema(domain_name: str, ids: tuple[str, ...]) -> type[BaseModel]:
    """Reply shape for the diagnostic agent, closed over this catalogue."""
    labels = Literal[ids + (UNKNOWN,)]  # type: ignore[valid-type]
    return create_model(
        f"Diagnosis_{domain_name}",
        considered=(
            list[labels],  # type: ignore[valid-type]
            Field(
                default_factory=list,
                description=(
                    "Candidate misconceptions consistent with the step, before "
                    "choosing between them."
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
    """Reply shape for the tutor. Domain-independent — it is prose."""

    text: str = Field(description="What to say to the learner. Two sentences at most.")


def schemas_for(domain: Domain) -> dict[str, Any]:
    return {
        "diagnosis": diagnosis_schema(domain.name, tuple(domain.misconceptions.ids())),
        "plan": plan_schema(domain.name, tuple(domain.concepts.ids())),
        "hint": HintReply,
    }
