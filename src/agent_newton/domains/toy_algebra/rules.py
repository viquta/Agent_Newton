"""Buggy rules: how a simulated learner errs under each misconception.

Each rule is deterministic and computes from ``item.params`` rather than parsing
``item.prompt`` — so rewording a prompt never changes what a learner does, and a
seeded cohort stays exactly reproducible.

Returning ``None`` means the item cannot elicit this misconception. The domain
validator checks the converse: every misconception an item claims to probe must
actually yield a response the verifier marks incorrect.
"""

from __future__ import annotations

from typing import Sequence

from agent_newton.domains.base import BuggyRule, Item


class DistributeFirstTermOnly:
    """``3(x + 4)`` -> ``3x + 4``: the factor reaches only the first term."""

    misconception_id = "distribute_first_term_only"

    def apply(self, item: Item) -> str | None:
        a, c = item.params.get("a"), item.params.get("c")
        if a is None or c is None:
            return None
        return f"{a}x + {c}"


class CombineUnlikeTerms:
    """``4x + 5`` -> ``9x``: a variable term and a constant added as like terms."""

    misconception_id = "combine_unlike_terms"

    def apply(self, item: Item) -> str | None:
        a, c = item.params.get("a"), item.params.get("c")
        if a is None or c is None:
            return None
        return f"{int(a) + int(c)}x"


class SignErrorMovingTerm:
    """``x + 5 = 12`` -> ``17``: the term crosses the equals sign unnegated."""

    misconception_id = "sign_error_moving_term"

    def apply(self, item: Item) -> str | None:
        if item.params.get("form") != "add":
            return None
        c, rhs = item.params.get("c"), item.params.get("rhs")
        if c is None or rhs is None:
            return None
        return str(int(rhs) + int(c))


class DropCoefficientWhenSolving:
    """``3x = 12`` -> ``12``: the constant is read off without dividing."""

    misconception_id = "drop_coefficient_when_solving"

    def apply(self, item: Item) -> str | None:
        if item.params.get("form") != "mul":
            return None
        rhs = item.params.get("rhs")
        return None if rhs is None else str(int(rhs))


RULES: Sequence[BuggyRule] = (
    DistributeFirstTermOnly(),
    CombineUnlikeTerms(),
    SignErrorMovingTerm(),
    DropCoefficientWhenSolving(),
)
