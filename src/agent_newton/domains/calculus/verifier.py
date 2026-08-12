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
from dataclasses import dataclass
from typing import Iterator, cast

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

#: Sample points that must actually have been evaluable before numeric
#: agreement may stand in for a symbolic proof. Below this, a timed-out
#: simplification means the answer was not verified at all.
MIN_SAMPLES_TO_ACCEPT = 4

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
# The constant of integration is written both ways. It is mapped onto the *same*
# symbol rather than admitted as a second one: two distinct symbols would make
# "x**3 + c" and "x**3 + C" inequivalent, which would turn a response the
# verifier merely could not read into one it scores as wrong.
_SYMBOLS["c"] = _SYMBOLS["C"]
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
    # Expr only, not Basic. Basic is the superclass, so admitting it also
    # admits things that are not expressions at all — a relational like "x > 2"
    # parses to a Boolean, which would then be subtracted from an expression
    # downstream and yield nonsense rather than a verdict.
    if not isinstance(expr, sympy.Expr):
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


def _subtract(a: sympy.Expr, b: sympy.Expr) -> sympy.Expr:
    """``a - b``.

    Wrapped because sympy's arithmetic operators are built by decorators that
    erase their signatures, so a static checker cannot see that ``Expr``
    supports ``-``. Keeping the cast here documents that once, instead of
    scattering suppressions through the module.
    """
    return cast(sympy.Expr, a - b)


def _substitute(expr: sympy.Expr, point: dict[sympy.Symbol, sympy.Expr]) -> sympy.Expr:
    """``expr`` with every symbol replaced by a value from ``point``.

    Cast for the same reason as :func:`_subtract`: sympy's ``subs`` is typed as
    returning ``Basic``, but substituting numbers into an expression yields an
    expression, and the caller needs ``evalf``.

    Pairs rather than the dict itself, which sympy's stub does not describe.
    The two forms differ only in that pairs substitute sequentially — immaterial
    here, since every replacement is a number and no symbol can reappear.
    """
    return cast(sympy.Expr, expr.subs(list(point.items())))


def _free_symbols(*exprs: sympy.Expr) -> list[sympy.Symbol]:
    found: set[sympy.Symbol] = set()
    for expr in exprs:
        # Typed as set[Basic] by sympy; every member is a Symbol at runtime,
        # which is what subs() needs as a key.
        found |= cast("set[sympy.Symbol]", expr.free_symbols)
    return sorted(found, key=str)


def _as_finite_complex(expr: sympy.Expr) -> complex | None:
    """Numeric value of ``expr``, or None where it is not usable.

    Poles, NaNs and non-numeric results all mean "this point tells us nothing",
    and every one of them must be caught here — a conversion that escapes would
    abort the whole session over one unlucky sample point.
    """
    try:
        value = complex(expr.evalf())
    except (TypeError, ValueError, ZeroDivisionError, AttributeError, OverflowError):
        return None
    if value != value or abs(value) == float("inf"):  # NaN or pole
        return None
    return value


@dataclass(frozen=True, slots=True)
class _Screen:
    """Outcome of the numeric screen.

    ``checked`` is load-bearing, not diagnostic: agreement across zero evaluable
    points is not evidence of anything, and callers must not treat it as such.
    """

    disagrees: bool
    checked: int


def _numeric_screen(a: sympy.Expr, b: sympy.Expr, rng: random.Random) -> _Screen:
    """Sample both expressions; one clear disagreement proves inequivalence."""
    symbols = _free_symbols(a, b)
    difference = _subtract(a, b)
    checked = 0

    for _ in range(SAMPLE_POINTS):
        # Irrational-ish points avoid the small integers where distinct
        # expressions coincide by accident.
        substitution: dict[sympy.Symbol, sympy.Expr] = {
            s: sympy.Float(rng.uniform(0.35, 2.65)) for s in symbols
        }

        gap = _as_finite_complex(_substitute(difference, substitution))
        if gap is None:
            continue

        # Scale the tolerance by the operand's own magnitude so floating-point
        # error in a large expression is not mistaken for a real difference.
        # Where that magnitude is unusable, fall back to an absolute tolerance
        # rather than skipping the point.
        magnitude = _as_finite_complex(_substitute(a, substitution))
        scale = max(1.0, abs(magnitude)) if magnitude is not None else 1.0

        checked += 1
        if abs(gap) > TOLERANCE * scale:
            return _Screen(disagrees=True, checked=checked)

    return _Screen(disagrees=False, checked=checked)


class VerificationUnavailable(Exception):
    """Equivalence could not be decided either numerically or symbolically."""


def _equivalent(a: sympy.Expr, b: sympy.Expr, rng: random.Random) -> tuple[bool, str]:
    if a == b:
        return True, ""

    screen = _numeric_screen(a, b, rng)
    if screen.disagrees:
        return False, ""

    try:
        with _time_limit(SIMPLIFY_TIMEOUT):
            return bool(sympy.simplify(_subtract(a, b)) == 0), ""
    except TimeoutError:
        if screen.checked >= MIN_SAMPLES_TO_ACCEPT:
            return True, f"accepted on agreement at {screen.checked} points; simplify timed out"
        # Nothing was evaluable and nothing was proved. Reporting this as
        # correct would score an answer that was never checked, and reporting it
        # as incorrect would blame the learner for our own failure to measure.
        raise VerificationUnavailable(
            f"simplify timed out and only {screen.checked} sample point(s) "
            f"could be evaluated"
        ) from None
    except Exception as exc:
        raise VerificationUnavailable(f"simplification failed: {exc}") from exc


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

        if item.params.get("up_to_constant"):
            # An antiderivative is determined only up to a constant, so two of
            # them agree exactly when their derivatives do. Comparing the
            # expressions instead charges an error to a learner who multiplied
            # out and dropped a constant the question told them to omit — a
            # correct answer scored as a mistake, which then writes an error
            # event and aims a hint at a misconception nobody held.
            #
            # Opt-in per item, and deliberately **not** set on the
            # antiderivative items: there the constant is the whole point, and
            # ignoring it would make the misconception being probed invisible.
            try:
                expected = tuple(sympy.diff(e, _SYMBOLS["x"]) for e in expected)
                given = tuple(sympy.diff(g, _SYMBOLS["x"]) for g in given)
            except Exception as exc:  # pragma: no cover - sympy internals
                return VerificationResult(
                    Verdict.UNPARSEABLE, item.answer, f"could not differentiate: {exc}"
                )

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
                try:
                    same, note = _equivalent(candidate, target, rng)
                except VerificationUnavailable as exc:
                    # We failed to measure, so we know nothing about this
                    # response. UNPARSEABLE keeps it out of the learner model
                    # rather than charging our own failure to the learner.
                    return VerificationResult(
                        Verdict.UNPARSEABLE, item.answer, f"could not verify: {exc}"
                    )
                if same:
                    if note:
                        notes.append(note)
                    remaining.pop(index)
                    break
            else:
                return VerificationResult(Verdict.INCORRECT, item.answer)

        return VerificationResult(Verdict.CORRECT, item.answer, "; ".join(notes))
