"""Symbolic verification for calculus answers.

Correctness is **equivalence**, not string match: ``2*sin(x)*cos(x)`` and
``sin(2*x)`` are the same answer, and a learner who writes the second must not
be marked wrong.

Three stages, cheapest first. The ordering matters because this runs after every
student step, thousands of times per cohort:

1. **Structural** — sympy's own equality. Instant, catches identical forms.
2. **Numeric screen** — evaluate both at random points. A single disagreement
   proves inequivalence, so most wrong answers are rejected here in
   microseconds, without ever calling ``simplify``.
3. **Symbolic confirmation** — only for answers that survive the screen.
   ``simplify`` can run unboundedly long on innocuous-looking input, so it is
   wrapped in a hard timeout.

If the timeout fires, agreement at every sampled point is accepted as
equivalent. That is a deliberate trade: a false accept needs two expressions
that agree at many random points yet differ symbolically, which is vanishingly
rare for the expression classes here, whereas a hung ``simplify`` would stall an
overnight run. Such cases are recorded in ``detail`` so they can be audited.
"""

from __future__ import annotations

import contextlib
import random
import signal
import threading
from typing import Iterator

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from agent_newton.domains.base import Item, VerificationResult, Verdict

#: Seconds allowed for one symbolic simplification.
SIMPLIFY_TIMEOUT = 2.0

#: Points sampled by the numeric screen, and the tolerance for agreement.
SAMPLE_POINTS = 12
TOLERANCE = 1e-7

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,  # "2x" -> 2*x
    convert_xor,  # "x^2" -> x**2
)

# The only names a response may use. parse_expr evaluates the parsed tree, so
# the namespace is closed explicitly rather than inheriting sympy's globals —
# a response is untrusted input.
_ALLOWED = {
    name: getattr(sympy, name)
    for name in (
        "sin cos tan sec csc cot asin acos atan sinh cosh tanh "
        "exp log ln sqrt Abs pi E Integral Derivative"
    ).split()
    if hasattr(sympy, name)
}
_ALLOWED["ln"] = sympy.log
_SYMBOLS = {s: sympy.Symbol(s) for s in ("x", "y", "h", "t", "u", "a", "b", "n", "C")}
_NAMESPACE = {**_ALLOWED, **_SYMBOLS}

# parse_expr compiles the transformed source, and that source calls sympy
# constructors by name — "10" becomes Integer(10). Those constructors must be
# reachable, so the global namespace is these and nothing else: no builtins, no
# import machinery, no way out.
_GLOBALS = {
    **_NAMESPACE,
    "Integer": sympy.Integer,
    "Float": sympy.Float,
    "Rational": sympy.Rational,
    "Symbol": sympy.Symbol,
}


class UnparseableResponse(ValueError):
    """The response could not be read as an expression."""


@contextlib.contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Abort the block after ``seconds``.

    SIGALRM is only deliverable on the main thread; off it, this degrades to no
    timeout rather than raising, so a threaded caller still gets an answer.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _fire(signum, frame):  # noqa: ANN001
        raise TimeoutError("symbolic simplification exceeded its budget")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def parse(text: str) -> sympy.Expr:
    """Parse one expression from student notation."""
    cleaned = text.strip().replace("−", "-").replace("·", "*")
    if not cleaned:
        raise UnparseableResponse("empty response")
    # dy/dx is notation, not division by a symbol named dx.
    cleaned = cleaned.replace("dy/dx", "Derivative(y, x)")
    try:
        expr = parse_expr(
            cleaned,
            local_dict=_NAMESPACE,
            global_dict=_GLOBALS,
            transformations=_TRANSFORMS,
            evaluate=True,
        )
    except Exception as exc:  # sympy raises a wide variety here
        raise UnparseableResponse(f"could not parse {text!r}: {exc}") from exc
    if not isinstance(expr, (sympy.Expr, sympy.Basic)):
        raise UnparseableResponse(f"{text!r} is not an expression")

    # Implicit multiplication will happily read "no idea" as no*idea, inventing
    # a symbol per word. Prose must come back UNPARSEABLE, not INCORRECT — the
    # learner said nothing mathematical, and scoring it wrong would feed a false
    # error event into the learner model.
    unknown = sorted(str(s) for s in expr.free_symbols if str(s) not in _SYMBOLS)
    if unknown:
        raise UnparseableResponse(f"unknown symbol(s) in {text!r}: {', '.join(unknown)}")

    return expr


def parse_answer(text: str) -> tuple[sympy.Expr, ...]:
    """Parse an answer that may name several values, e.g. ``"0, 2"``.

    Solution sets are compared unordered, so ``"2, 0"`` is the same answer —
    order carries no meaning when the question asks for all roots.
    """
    parts = [p for p in text.split(",") if p.strip()]
    if not parts:
        raise UnparseableResponse("empty response")
    return tuple(parse(p) for p in parts)


def _free_symbols(*exprs: sympy.Expr) -> list[sympy.Symbol]:
    found: set[sympy.Symbol] = set()
    for expr in exprs:
        found |= expr.free_symbols
    return sorted(found, key=str)


def _numerically_disagrees(a: sympy.Expr, b: sympy.Expr, rng: random.Random) -> bool:
    """True when the two provably differ at some point.

    Sampling failures (poles, complex results, undefined points) are skipped
    rather than counted: they say nothing about equivalence.
    """
    symbols = _free_symbols(a, b)
    difference = a - b
    checked = 0

    for _ in range(SAMPLE_POINTS):
        # Irrational-ish points avoid the small integers where distinct
        # expressions coincide by accident.
        substitution = {s: sympy.Float(rng.uniform(0.35, 2.65)) for s in symbols}
        try:
            value = complex(difference.subs(substitution).evalf())
        except (TypeError, ValueError, ZeroDivisionError, AttributeError):
            continue
        if value != value or abs(value) == float("inf"):  # NaN / pole
            continue
        checked += 1
        if abs(value) > TOLERANCE * max(1.0, abs(complex(a.subs(substitution).evalf() or 0))):
            return True

    return False if checked else False


def _equivalent(a: sympy.Expr, b: sympy.Expr, rng: random.Random) -> tuple[bool, str]:
    if a == b:
        return True, ""

    if _numerically_disagrees(a, b, rng):
        return False, ""

    try:
        with _time_limit(SIMPLIFY_TIMEOUT):
            return bool(sympy.simplify(a - b) == 0), ""
    except TimeoutError:
        # Survived every sample but could not be confirmed symbolically.
        return True, "accepted on numeric agreement; simplify timed out"
    except Exception as exc:
        return False, f"simplification failed: {exc}"


class SymbolicVerifier:
    """Equivalence-checking verifier for the calculus domain."""

    def __init__(self, seed: int = 0) -> None:
        # Seeded so a given (item, response) pair always samples the same
        # points — a verdict must not vary between runs.
        self._seed = seed

    def verify(self, item: Item, response: str) -> VerificationResult:
        try:
            expected = parse_answer(item.answer)
        except UnparseableResponse as exc:
            # The item is malformed, not the learner. `domain validate` exists
            # to catch this before a run starts.
            return VerificationResult(
                Verdict.UNPARSEABLE, item.answer, f"item {item.id!r} has a bad answer: {exc}"
            )

        try:
            given = parse_answer(response)
        except UnparseableResponse as exc:
            return VerificationResult(Verdict.UNPARSEABLE, item.answer, str(exc))

        if len(given) != len(expected):
            return VerificationResult(
                Verdict.INCORRECT,
                item.answer,
                f"expected {len(expected)} value(s), got {len(given)}",
            )

        rng = random.Random(f"{self._seed}:{item.id}:{response}")
        remaining = list(expected)
        notes: list[str] = []

        for candidate in given:
            for index, target in enumerate(remaining):
                same, note = _equivalent(candidate, target, rng)
                if same:
                    if note:
                        notes.append(note)
                    remaining.pop(index)
                    break
            else:
                return VerificationResult(Verdict.INCORRECT, item.answer)

        return VerificationResult(Verdict.CORRECT, item.answer, "; ".join(notes))
