"""A deliberately non-CAS verifier.

The calculus domain checks symbolic equivalence with sympy. This one compares
canonicalised term multisets instead — no algebra system involved.

That is the point. If both domains verified through sympy, the ``Verifier``
Protocol would only have been shown to work for CAS-checkable subjects, and the
claim that the architecture is domain-agnostic would be untested. A future
programming domain would verify by running unit tests; this stands in for that
shape of implementation.
"""

from __future__ import annotations

import re

from agent_newton.domains.base import Item, VerificationResult, Verdict

_TERM = re.compile(r"[+-]?[^+-]+")
_LEGAL = frozenset("0123456789x+-/.")


class UnparseableResponse(ValueError):
    """The response could not be read at all."""


def canonical(text: str) -> tuple[str, ...]:
    """Sorted multiset of signed terms.

    ``"3x + 12"``, ``"12 + 3x"`` and ``"12+3*x"`` all canonicalise alike, so a
    correct answer written in a different order is not scored wrong.
    """
    compact = text.strip().lower().replace(" ", "").replace("*", "")
    if not compact:
        raise UnparseableResponse("empty response")

    illegal = sorted(set(compact) - _LEGAL)
    if illegal:
        raise UnparseableResponse(f"unexpected characters: {illegal}")

    matches = _TERM.findall(compact)
    # Every character must belong to some term. Without this, a trailing sign
    # ("3x+") silently parses as "3x" and is scored as a wrong answer rather
    # than an unreadable one — which would feed a bogus error event into the
    # learner model.
    if "".join(matches) != compact:
        raise UnparseableResponse(f"could not read all of {text!r}")

    terms: list[str] = []
    for raw in matches:
        signed = raw if raw[0] in "+-" else f"+{raw}"
        sign, body = signed[0], signed[1:]
        if not body:
            raise UnparseableResponse(f"dangling sign in {text!r}")
        # "x" and "1x" are the same term; normalise so they compare equal.
        if body == "x":
            body = "1x"
        terms.append(sign + body)

    if not terms:
        raise UnparseableResponse(f"no terms in {text!r}")
    return tuple(sorted(terms))


class NormalizingVerifier:
    """Model-independent correctness for toy_algebra."""

    def verify(self, item: Item, response: str) -> VerificationResult:
        try:
            student = canonical(response)
        except UnparseableResponse as exc:
            # Distinct from INCORRECT on purpose: an unreadable response is a
            # measurement failure, not evidence about the learner, and must not
            # update the learner model.
            return VerificationResult(Verdict.UNPARSEABLE, item.answer, str(exc))

        expected = canonical(item.answer)
        verdict = Verdict.CORRECT if student == expected else Verdict.INCORRECT
        return VerificationResult(verdict, item.answer)
