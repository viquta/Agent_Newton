"""The rule engine: what a simulated learner actually does.

Deterministic given a seed. The engine decides whether a step is correct and,
if not, which misconception produced it; a surface renderer may later phrase
that decision in natural language, but it never changes it.

Randomness is drawn per decision from a hash of
``(seed, learner, item, repetition, attempt, misconception)`` rather than from a
running generator. Three consequences:

* A decision does not depend on how many rolls happened on *other* items, so
  reordering unrelated work cannot perturb it.
* **The same learner meeting the same item for the nth time draws the same
  number in both architectures.** Only the tutoring differs, so the comparison
  is not also absorbing the difference between two random streams. This is the
  common random numbers technique, and it lowers the variance of the paired
  comparison at no cost in bias.
* ``repetition`` is part of the key, so practising an item again is a fresh
  draw. Without it a repeated item reproduces its own past verbatim: a learner
  who once answered correctly would do so forever, mastering the concept
  without ever demonstrating anything, and one who erred could never stop.
  Common random numbers then survive only while the two arms have given the
  item the same number of times — which is exactly as long as their histories
  agree, and is the most the technique can offer once they genuinely diverge.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent_newton.config import SimulatorConfig
from agent_newton.core.simulator.profile import MisconceptionProfile, solidity
from agent_newton.domains.base import Domain, Item


@dataclass(frozen=True, slots=True)
class SimulatedStep:
    """One step, with the ground truth that produced it.

    ``fired`` is the injected label the diagnostic agent is scored against. It
    is never exposed to any agent, and it is ``None`` whenever no misconception
    produced the step — including every step by a human, who has no injected
    label at all.

    ``correct`` is what the *learner* believed; the verifier is what decides,
    and the session reads its verdict rather than this field.
    """

    response: str
    fired: str | None
    correct: bool

    @property
    def label(self) -> str:
        return self.fired or ("correct" if self.correct else "unlabelled-error")


@runtime_checkable
class Learner(Protocol):
    """Whoever is answering: a simulated profile, or a person.

    The session depends on this and not on the simulator, so a human sits in
    the same loop the cohorts run — the demo exercises the real system rather
    than a re-implementation of it.
    """

    @property
    def learner_id(self) -> str: ...

    def answer(self, item: Item, attempt: int = 0, repetition: int = 0) -> SimulatedStep: ...

    def receive_hint(self, targeted_misconception: str | None) -> bool: ...

    def reflect(self, item: Item, prompt: str) -> str | None:
        """Answer a reflective prompt in prose, or ``None`` to say nothing.

        A reflection is **not** an answer attempt. It is not verified, it costs
        no attempt, and it is not counted as an unmeasurable step — the tutor
        asked a question in words, and a person replying in words has not
        failed to solve anything.
        """
        ...

    def discuss(self, concept_id: str, prompt: str) -> str | None:
        """Reply to the tutor while a concept is being explained, or ``None``.

        Half of a conversation rather than an answer to anything. A lesson is
        about a *concept* and may happen with no question in front of the
        learner at all, which is why this takes a concept id where
        :meth:`reflect` takes an item.

        ⚠️ That difference is the whole reason this is a separate method rather
        than a reuse of :meth:`reflect`. Reflecting through here would need a
        synthetic ``Item``, whose id would flow into the recorded utterance and
        from there into ``said_about`` — which filters by concept *and* item. A
        sitting once had the tutor ask a learner to review their derivative of
        ``u^4`` on a question containing no ``u^4``, and that is the same defect
        being reintroduced on purpose.

        ``None`` ends the conversation, and a simulated learner always returns
        it. That is what makes a dialogue structurally unreachable in a cohort:
        not a flag that is off, but a learner who cannot talk.
        """
        ...

    def show_working(
        self, item: Item, response: str, required: bool = False
    ) -> str | None:
        """The steps taken to reach ``response``, or ``None`` to show none.

        Volunteered rather than asked for, which is what separates it from
        :meth:`reflect`. A person working on paper has reasoning the answer
        alone does not carry, and a tutor that never sees it can only guess at
        where the reasoning went wrong. Prose on the same terms as a
        reflection: never verified, never an attempt.

        ``required`` is set when the step did not come out right, and it is the
        session saying that this is the moment the reasoning is worth having:
        the answer alone cannot distinguish a method that was wrong from
        arithmetic that slipped, and an unreadable answer says nothing at all.
        A front end may insist; whether it does is its business, and returning
        ``None`` under it is a refusal rather than an error — the session
        records that it was asked and declined.

        A simulated learner returns ``None``. It has no reasoning to show —
        its answer comes from a buggy rule, not from steps — and inventing
        prose for it would put words into the tutor's prompt that no cohort
        result should depend on.
        """
        ...

    def remediation_ratio(self) -> float | None:
        """How far the learner's misconceptions have been reduced.

        ``None`` when there is no ground-truth profile to measure against,
        which is the case for a person. Reported as unavailable rather than as
        zero: a zero would be read as "nothing was remediated".
        """
        ...


def _roll(
    seed: int,
    learner_id: str,
    item_id: str,
    repetition: int,
    attempt: int,
    misconception_id: str,
) -> float:
    """A stable number in [0, 1) for one specific decision."""
    key = f"{seed}|{learner_id}|{item_id}|{repetition}|{attempt}|{misconception_id}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class SimulatedLearner:
    """A learner defined by a misconception profile and a domain's buggy rules."""

    def __init__(
        self,
        profile: MisconceptionProfile,
        domain: Domain,
        config: SimulatorConfig,
    ) -> None:
        self._profile = profile
        self._domain = domain
        self._config = config

    @property
    def profile(self) -> MisconceptionProfile:
        """Ground truth. For the evaluation harness only — never for an agent."""
        return self._profile

    @property
    def learner_id(self) -> str:
        return self._profile.learner_id

    def remediation_ratio(self) -> float | None:
        return self._profile.remediation_ratio()

    def answer(self, item: Item, attempt: int = 0, repetition: int = 0) -> SimulatedStep:
        """Produce a step for this item.

        Misconceptions this learner holds *and* which the item can elicit are
        considered in a fixed order; the first to fire produces the response. If
        none fires, the learner answers correctly.

        ``attempt`` counts steps within one visit; ``repetition`` counts how
        many times the item has been given before. Both enter the draw, so
        neither a retry nor a later revisit repeats an earlier outcome.
        """
        applicable = sorted(m for m in item.probes if self._profile.holds(m))

        for misconception_id in applicable:
            probability = self._profile.probability(misconception_id)
            if _roll(
                self._profile.seed,
                self._profile.learner_id,
                item.id,
                repetition,
                attempt,
                misconception_id,
            ) >= probability:
                continue

            rule = self._domain.buggy_rule(misconception_id)
            if rule is None:
                continue
            wrong = rule.apply(item)
            if wrong is None:
                # The item declares it probes this misconception but the rule
                # cannot act on it. `domain validate` rejects that combination,
                # so this is a guard rather than an expected path.
                continue

            return SimulatedStep(response=wrong, fired=misconception_id, correct=False)

        return SimulatedStep(response=item.answer, fired=None, correct=True)

    def reflect(self, item: Item, prompt: str) -> str | None:  # noqa: ARG002
        """A simulated learner has nothing to say.

        The reflective prompt still costs it a turn — that is what gives the
        error-first rule a price — but it produces no prose.
        """
        return None

    def discuss(self, concept_id: str, prompt: str) -> str | None:  # noqa: ARG002
        """Nothing to say. A rule engine cannot hold a conversation.

        Returning ``None`` ends the dialogue at its first turn, so a lesson
        collapses to the opening and its written summary however
        ``teaching.lesson_turns`` is set. That is the guarantee worth having: a
        cohort cannot be taught conversationally because its learner cannot
        converse, which is an inability rather than a configuration — and there
        is a test asserting *that* rather than only asserting the numbers did
        not move.
        """
        return None

    def show_working(
        self, item: Item, response: str, required: bool = False
    ) -> str | None:  # noqa: ARG002
        """Nothing to show. Its answer comes from a rule, not from steps.

        Keeping this empty is what keeps the channel out of the cohorts: no
        measured result can depend on prose that was never written — including
        under ``required``, which a rule has no more to say about than it had
        before.
        """
        return None

    def receive_hint(self, targeted_misconception: str | None) -> bool:
        """Apply a hint. Returns whether it changed anything.

        A hint weakens a misconception only if it names one the learner actually
        holds. Aiming at the wrong one — or at nothing — leaves the learner
        exactly as they were, which is the mechanism that gives diagnostic
        accuracy its consequences for learning outcomes.

        Under ``simulator.prerequisite_dependence`` it also takes less well when
        the learner's foundations under that concept are shaky. That is the one
        thing this generator did not represent: order of instruction changed
        nothing at all, so an architecture judged on sequencing was being
        credited for something the simulator could not reward. Zero leaves the
        old behaviour untouched, exactly.
        """
        if targeted_misconception is None:
            return False
        return self._profile.remediate(
            targeted_misconception, self._efficacy(targeted_misconception)
        )

    def _efficacy(self, misconception_id: str) -> float:
        """The multiplier a correct hint applies, given the foundations under it.

        ``1 - (1 - factor) * (1 - k * (1 - solidity))``. At ``k = 0`` this is the
        configured factor and nothing is computed, so the untouched path is
        untouched — the byte-identical baseline is what the whole sweep is read
        against.

        At ``k = 1`` on a concept with nothing solid beneath it the multiplier is
        1.0: the hint lands, is correctly aimed, and does nothing, because there
        was nothing underneath to attach it to. At full solidity it is the
        configured factor whatever ``k`` is, so a well-founded learner is never
        penalised for the dial existing.

        Deterministic and drawn from no random source, so a seeded cohort stays
        exactly reproducible at every point on the sweep.
        """
        factor = self._config.remediation_factor
        dependence = self._config.prerequisite_dependence
        if dependence <= 0.0:
            return factor
        try:
            concept_id = self._domain.misconceptions.get(misconception_id).concept_id
        except Exception:
            # A label outside the catalogue cannot be placed in the graph. It
            # also cannot be one the learner holds, so `remediate` is about to
            # decline it anyway.
            return factor
        sound = solidity(
            concept_id, self._profile, self._domain.misconceptions, self._domain.concepts
        )
        return 1.0 - (1.0 - factor) * (1.0 - dependence * (1.0 - sound))
