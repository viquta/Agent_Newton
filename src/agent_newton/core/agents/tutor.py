"""The template tutor — hints without a model.

Text is assembled from the domain's misconception catalogue at the support
level the scaffolding rule asks for. The wording is plain; what matters for the
experiment is *what the hint targets*, since that is the only thing the learner
responds to.

The tutor is driven by the instructional rules rather than checked against them
afterwards: it asks what move is required and makes it, so the error-first
ordering holds by construction.
"""

from __future__ import annotations

from typing import Sequence

from agent_newton.core.agents.base import Diagnosis, Hint, StateView
from agent_newton.core.pedagogy import (
    HintLevel,
    TeachingStyle,
    TutorMove,
    hint_level,
    move_for,
)
from agent_newton.config import ScaffoldingPolicy, ZPDConfig
from agent_newton.domains.base import ConceptResource, Domain, Item


class TemplateTutor:
    """Model-free tutor. Support level from the rules, target from the diagnosis."""

    def __init__(
        self, band: ZPDConfig, policy: ScaffoldingPolicy = "banded"
    ) -> None:
        self._band = band
        self._policy: ScaffoldingPolicy = policy

    def explain(
        self,
        resource: ConceptResource,
        style: TeachingStyle,
        exchanges: Sequence[tuple[str, str]] = (),
    ) -> str:
        """The lesson as authored. ``style`` and ``exchanges`` are ignored.

        Deliberate, and the same reasoning as ``respond`` ignoring ``response``:
        the cohorts run this tutor, so its output must not vary with anything
        the model-backed one gained. A lesson whose wording moved with the style
        would make every measured number depend on something outside the
        manipulation. There is a test asserting it cannot vary.

        Ignoring the style is not a degraded lesson. ``PLAIN`` *is* the authored
        text, and the authored text is the one thing every domain offering
        resources is guaranteed to have.

        Nor is ignoring the conversation. This tutor has nothing to say back
        that it did not already say, so a dialogue driven by it would be the
        same paragraph repeated — which is the failure the account ceiling was
        added to prevent. It gives its one turn and the summary follows.
        """
        return resource.lesson()

    def respond(
        self,
        item: Item,
        diagnosis: Diagnosis,
        view: StateView,
        domain: Domain,
        *,
        response: str,
        mastery: float,
        prior_failures: int,
        moves_this_item: Sequence[TutorMove],
        said_this_item: Sequence[str] = (),
        explained: bool = False,
    ) -> Hint:
        # ``response`` and ``said_this_item`` are deliberately unused. This
        # tutor's text comes from the catalogue at the level the scaffolding
        # rule asks for, and the cohorts run it — so its output must not vary
        # with anything the model-backed tutor gained. There is a test
        # asserting exactly that.
        #
        # It therefore does repeat itself when the level does not move, which is
        # a property of a catalogue lookup rather than an oversight: the same
        # level on the same misconception *is* the same instruction. Only a
        # tutor that writes prose can say the same thing differently.
        del response, said_this_item

        # The session supplies the scaffolding rule's inputs. Reading them here
        # is what put the failure being responded to into both of them — see the
        # Tutor protocol.
        level = hint_level(mastery, prior_failures, self._band, policy=self._policy)

        move = move_for(
            level,
            moves_this_item,
            misconception_confirmed=diagnosis.named,
            already_explained=explained,
        )
        if move is TutorMove.REFLECT:
            return Hint(
                text=(
                    f"Before we fix it — look again at your step for "
                    f"'{item.prompt}'. Which part are you least sure of, and why?"
                ),
                move=TutorMove.REFLECT,
                level=level,
                # A reflective prompt deliberately targets nothing: it costs a
                # turn and remediates nothing, which is what makes the
                # error-first rule a real constraint rather than a free one.
                #
                # That also carries the whole of `HintLevel.NONE`: a learner the
                # model believes has the concept gets this turn and no other, so
                # the only move that teaches is withheld exactly where the
                # evidence says teaching is not what is missing.
                targets=None,
            )

        if not diagnosis.named:
            return Hint(
                text=f"That is not right yet. Check your working on '{item.prompt}'.",
                move=TutorMove.HINT,
                level=level,
                targets=None,
            )

        assert diagnosis.misconception_id is not None
        misconception = domain.misconceptions.get(diagnosis.misconception_id)
        return Hint(
            text=_text_for(level, misconception.description, item),
            move=TutorMove.REMEDIATE,
            level=level,
            targets=diagnosis.misconception_id,
        )


def _text_for(level: HintLevel, description: str, item: Item) -> str:
    """The catalogue's words, at the level the rule asked for.

    ⚠️ No level states ``item.answer``. The worked step used to, and a person
    reading one said what that is worth: the reply assembled the answer and the
    only thing left to do was type it back, which the learner model then records
    as knowing it. The answer is given when the item closes instead, where the
    next question on the concept carries different numbers.
    """
    description = " ".join(description.split())
    if level is HintLevel.NUDGE:
        return "Almost — one part of that step does not follow. Try it once more."
    if level is HintLevel.TARGETED:
        return f"Here is what went wrong: {description}"
    return (
        f"Here is what went wrong: {description} Start again from the rule "
        f"itself, and take '{item.prompt}' one step at a time."
    )
