"""A person, in the same loop the cohorts run.

The demo drives the real :class:`~agent_newton.core.orchestration.session.Session`
rather than a copy of it, so what a person sees is the system that produced the
numbers — the same planner, verifier, arbitration policy and blackboard. A
separate interactive loop would eventually drift from the measured one, and the
demo would stop being evidence about anything.

**A person has no injected label.** ``SimulatedStep.fired`` is always ``None``,
which has consequences the rest of the system has to respect:

* The oracle and noised-oracle diagnostics read that label, so with a human they
  would diagnose nothing at all, no hint would ever target a misconception, and
  no remediation could occur. The configuration is rejected at load rather than
  producing a session that looks like it ran — see ``config.py``.
* Diagnostic accuracy cannot be computed, because there is nothing to score
  against. It is reported as unavailable, never as zero.
* ``remediation_ratio`` is likewise ``None``: there is no ground-truth profile
  to measure a reduction against.
"""

from __future__ import annotations

from typing import Callable

from agent_newton.core.simulator.engine import SimulatedStep
from agent_newton.domains.base import Item


class HumanLearner:
    """Reads answers from a person.

    ``ask`` is injected rather than reading stdin directly, so the loop can be
    driven by a terminal prompt, a test, or any other front end without this
    class knowing which.
    """

    def __init__(
        self,
        ask: Callable[[Item, int], str],
        learner_id: str = "human",
        on_hint: Callable[[str | None], None] | None = None,
        ask_reflection: Callable[[Item, str], str] | None = None,
    ) -> None:
        self._ask = ask
        self._ask_reflection = ask_reflection
        self._learner_id = learner_id
        self._on_hint = on_hint
        #: Every response given, in order. The demo shows it back at the end;
        #: nothing in the loop reads it.
        self.responses: list[tuple[str, str]] = []

    @property
    def learner_id(self) -> str:
        return self._learner_id

    def answer(self, item: Item, attempt: int = 0, repetition: int = 0) -> SimulatedStep:
        response = self._ask(item, attempt)
        self.responses.append((item.id, response))
        # `fired` is None because there is no injected label, and `correct` is
        # not consulted by the loop — the verifier decides, which is the whole
        # reason correctness here does not depend on a model.
        return SimulatedStep(response=response, fired=None, correct=False)

    def reflect(self, item: Item, prompt: str) -> str | None:
        """Take the person's reply to a reflective prompt, in their own words.

        The tutor asked a question in words; before this existed the reply went
        to the symbolic verifier, came back unreadable, and cost an attempt. A
        reflection is not an answer.
        """
        if self._ask_reflection is None:
            return None
        said = self._ask_reflection(item, prompt).strip()
        return said or None

    def receive_hint(self, targeted_misconception: str | None) -> bool:
        """A person is not adjusted by a hint; they read it.

        Returns False always: nothing measurable changed, and claiming
        otherwise would put a number into the outcome that means nothing.
        """
        if self._on_hint is not None:
            self._on_hint(targeted_misconception)
        return False

    def remediation_ratio(self) -> float | None:
        """Unavailable. There is no profile to measure a reduction against."""
        return None
