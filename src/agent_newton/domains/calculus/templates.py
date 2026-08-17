"""Numeric variants of the calculus practice items.

A concept is worked until its posterior clears the band, and most concepts carry
one item — so without these the same question is asked verbatim until mastery.
A person who sat through that said what it is: memorising an answer rather than
learning a method.

Each template regenerates ``prompt``, ``answer`` and ``params`` from the same
draw, so the buggy rules — which compute against ``params`` — stay aligned with
the question actually asked. Draw 0 reproduces the item as written in the YAML,
which keeps that file the readable definition and keeps anything referring to an
item by id meaning what it meant. Both properties are checked by
``domain validate``, over the first several draws of every template.

The draw is the repetition count the session already tracks. Nothing here draws
from a generator, so reproducibility does not depend on call ordering.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from agent_newton.domains.base import Item, ItemTemplate


def _term(coefficient: int, exponent: int, symbol: str = "x") -> str:
    """A monomial in sympy notation, written the way a person would.

    ``2*x`` rather than ``2*x**1``, and ``3`` rather than ``3*x**0``. Cosmetic
    to sympy, which treats them as equal, but draw 0 has to reproduce the YAML
    string exactly and the YAML is written for a reader.
    """
    if exponent == 0:
        return str(coefficient)
    power = symbol if exponent == 1 else f"{symbol}**{exponent}"
    return power if coefficient == 1 else f"{coefficient}*{power}"


class _Template:
    """Shared plumbing: hold the id, rebuild the item from a draw."""

    item_id = ""

    def variant(self, base: Item, draw: int) -> Item:
        prompt, answer, params = self.parts(draw)
        return replace(base, prompt=prompt, answer=answer, params={**base.params, **params})

    def parts(self, draw: int) -> tuple[str, str, dict]:  # pragma: no cover - abstract
        raise NotImplementedError


class IndeterminateLimit(_Template):
    """lim(x -> a) of (x^2 - a^2)/(x - a), which is 2a once the form is resolved."""

    item_id = "ca_lim_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        a = 2 + draw
        return (
            f"Evaluate: lim(x -> {a}) of (x^2 - {a * a})/(x - {a})",
            str(2 * a),
            {"indeterminate": True},
        )


class DirectLimit(_Template):
    """lim(x -> a) of (x^2 + c). No indeterminate form: substitution settles it."""

    item_id = "ca_lim_p2"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        a, c = 1 + draw, 3 + draw
        return f"Evaluate: lim(x -> {a}) of (x^2 + {c})", str(a * a + c), {}


class AverageRate(_Template):
    """y = x^2 over [a, b]. The average rate is a + b; the difference is b^2 - a^2."""

    item_id = "ca_avg_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        a = 1 + draw
        b = a + 2 + draw % 3
        return (
            f"For y = x^2, find the average rate of change from x = {a} to x = {b}.",
            str(a + b),
            {"difference": b * b - a * a},
        )


class TangentAsLimit(_Template):
    """The tangent gradient to y = x^2 at x = a, which is 2a.

    The secant taken one unit along gives 2a + 1 — the answer of someone who
    computed a gradient but never took the limit.
    """

    item_id = "ca_tan_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        a = 2 + draw
        return (
            f"The tangent to y = x^2 at x = {a} is the limit of the gradients of "
            f"the secants joining that point to a nearby point on the curve, as "
            f"the nearby point approaches it. Find the tangent's gradient.",
            str(2 * a),
            {"secant_gradient": 2 * a + 1},
        )


class InstantaneousRate(_Template):
    """y = x^2 at x = a. The instantaneous rate is 2a, the average over one unit 2a + 1."""

    item_id = "ca_inst_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        a = 3 + draw
        return (
            f"For y = x^2, find the instantaneous rate of change at x = {a}.",
            str(2 * a),
            {"average_rate": 2 * a + 1},
        )


class BinomialExpansion(_Template):
    """(base + step)^2. Only the names change; the structure is the point.

    The pairs are drawn from the verifier's own symbol vocabulary rather than
    invented. That vocabulary is deliberately small — an unknown name is how
    prose is told apart from an expression — so a variant using a symbol outside
    it would produce a question whose correct answer the verifier cannot read.
    """

    item_id = "ca_fp_p1"
    _PAIRS: Sequence[tuple[str, str]] = (
        ("x", "h"),
        ("t", "h"),
        ("u", "h"),
        ("x", "a"),
        ("t", "a"),
        ("u", "a"),
        ("x", "b"),
        ("t", "b"),
        ("u", "b"),
    )

    def parts(self, draw: int) -> tuple[str, str, dict]:
        base, step = self._PAIRS[draw % len(self._PAIRS)]
        return (
            f"Expand ({base} + {step})^2.",
            f"{base}**2 + 2*{base}*{step} + {step}**2",
            {"base": base, "step": step, "exponent": 2},
        )


class PowerRulePlain(_Template):
    """y = x^n."""

    item_id = "ca_pow_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        n = 5 + draw
        return (
            f"Differentiate: y = x^{n}",
            _term(n, n - 1),
            {"coefficient": 1, "exponent": n},
        )


class PowerRuleWithCoefficient(_Template):
    """y = c·x^n."""

    item_id = "ca_pow_p2"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        c, n = 4 + draw, 3 + draw % 3
        return (
            f"Differentiate: y = {c}x^{n}",
            _term(c * n, n - 1),
            {"coefficient": c, "exponent": n},
        )


class ReciprocalPower(_Template):
    """y = c/x^n, whose derivative needs the exponent rewritten as negative."""

    item_id = "ca_neg_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        c, n = 2 + draw, 2 + draw % 3
        return (
            f"Differentiate: y = {c}/x^{n}",
            f"-{c * n}/x**{n + 1}",
            {"coefficient": c, "exponent": n},
        )


class PolynomialWithConstant(_Template):
    """y = x^3 - b·x^2 + k. The constant vanishes; that is what is being probed."""

    item_id = "ca_poly_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        b, k = 3 + draw, 4 + draw
        return (
            f"Differentiate: y = x^3 - {b}x^2 + {k}",
            f"3*x**2 - {2 * b}*x",
            {"constant": k},
        )


class StationaryPoints(_Template):
    """a·x^2 - a·r·x = 0, whose roots are 0 and r. Cancelling x loses the first."""

    item_id = "ca_stat_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        a, r = 3 + draw % 2, 2 + draw
        return (
            f"Solve {a}x^2 - {a * r}x = 0. Give all solutions.",
            f"0, {r}",
            {"nonzero_root": r},
        )


class ProductRule(_Template):
    """Four shapes of product, so neither factor's derivative is predictable.

    The first version moved only the second exponent — (x^2+1)(x^3+2),
    (x^2+1)(x^4+2), (x^2+1)(x^5+2) — leaving the first factor and its derivative
    fixed at every draw. Same defect as the quotient family, one factor along.
    """

    item_id = "ca_prod_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        shape = draw % 4
        step = draw // 4

        if shape == 0:
            # The item as written at draw 0.
            p, q = 2, 3 + step
            return (
                f"Differentiate: y = (x^{p} + 1)(x^{q} + 2)",
                f"{p}*x*(x**{q} + 2) + (x**{p} + 1)*{q}*x**{q - 1}",
                {
                    "f": f"x**{p} + 1", "g": f"x**{q} + 2",
                    "df": f"{p}*x", "dg": f"{q}*x**{q - 1}",
                },
            )

        if shape == 1:
            # A linear second factor: dg is a constant.
            p, c = 3 + step, 2 + step
            return (
                f"Differentiate: y = (x^{p} + 1)(x + {c})",
                f"{p}*x**{p - 1}*(x + {c}) + (x**{p} + 1)",
                {
                    "f": f"x**{p} + 1", "g": f"x + {c}",
                    "df": f"{p}*x**{p - 1}", "dg": "1",
                },
            )

        if shape == 2:
            # A bare monomial first factor: no constant term to carry.
            p, q = 2 + step, 3 + step
            return (
                f"Differentiate: y = x^{p}(x^{q} + 1)",
                f"{p}*x**{p - 1}*(x**{q} + 1) + x**{p}*{q}*x**{q - 1}",
                {
                    "f": f"x**{p}", "g": f"x**{q} + 1",
                    "df": f"{p}*x**{p - 1}", "dg": f"{q}*x**{q - 1}",
                },
            )

        # A subtraction, so the sign inside a factor is not always +.
        p, c = 2 + step, 3 + step
        return (
            f"Differentiate: y = (x^{p} - {c})(x + 1)",
            f"{p}*x**{p - 1}*(x + 1) + (x**{p} - {c})",
            {
                "f": f"x**{p} - {c}", "g": "x + 1",
                "df": f"{p}*x**{p - 1}", "dg": "1",
            },
        )



class QuotientRule(_Template):
    """x^2/(x + c) and three other shapes, so u' is not always 2x.

    The first version varied only the denominator constant: x^2/(x+1),
    x^2/(x+2), x^2/(x+3). Ten steps of that in one sitting, and a learner said
    what it looks like from the chair — "every new question just kept sequencing
    a +1 as you can see. This way, i never really learned the quotient rule very
    well." The numerator, and therefore u', never moved.

    Now the numerator's degree, the denominator's degree and their order all
    rotate, so neither u' nor v' can be carried over from the last question.
    """

    item_id = "ca_quot_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        shape = draw % 4
        step = draw // 4

        if shape == 0:
            # The item as written at draw 0.
            c = 1 + step
            return (
                f"Differentiate: y = x^2/(x + {c})",
                f"(2*x*(x + {c}) - x**2)/(x + {c})**2",
                {"f": "x**2", "g": f"x + {c}", "df": "2*x", "dg": "1"},
            )

        if shape == 1:
            # A cubic numerator, so u' is no longer 2x.
            c = 1 + step
            return (
                f"Differentiate: y = x^3/(x + {c})",
                f"(3*x**2*(x + {c}) - x**3)/(x + {c})**2",
                {"f": "x**3", "g": f"x + {c}", "df": "3*x**2", "dg": "1"},
            )

        if shape == 2:
            # A quadratic denominator, so v' is no longer 1.
            c = 2 + step
            return (
                f"Differentiate: y = x^2/(x^2 + {c})",
                f"(2*x*(x**2 + {c}) - x**2*2*x)/(x**2 + {c})**2",
                {"f": "x**2", "g": f"x**2 + {c}", "df": "2*x", "dg": "2*x"},
            )

        # Inverted: the linear factor on top, the power underneath.
        c = 1 + step
        return (
            f"Differentiate: y = (x + {c})/x^2",
            f"(x**2 - (x + {c})*2*x)/x**4",
            {"f": f"x + {c}", "g": "x**2", "df": "1", "dg": "2*x"},
        )



class ChainRuleCubed(_Template):
    """y = (x^m + 1)^3. The inner derivative is what gets dropped."""

    item_id = "ca_chain_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        m = 2 + draw
        return (
            f"Differentiate: y = (x^{m} + 1)^3",
            f"{_term(3 * m, m - 1)}*(x**{m} + 1)**2",
            {"outer_derivative": f"3*(x**{m} + 1)**2"},
        )


class ChainRulePower(_Template):
    """y = (a·x + 1)^4."""

    item_id = "ca_chain_p2"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        a, n = 3 + draw, 4
        inner = f"{a}*x + 1"
        return (
            f"Differentiate: y = ({a}x + 1)^{n}",
            f"{n * a}*({inner})**{n - 1}",
            {"outer_derivative": f"{n}*({inner})**{n - 1}"},
        )


class ImplicitRelation(_Template):
    """Relations whose dy/dx actually differs, so it has to be recomputed.

    The first version varied only the radius of ``x^2 + y^2 = r^2``, and its
    docstring stated the intent: *the radius changes; dy/dx does not, which is the
    lesson.* That is a real pedagogical point — the derivative is independent of
    the constant — and it had a side effect nobody looked for: **the answer is
    ``-x/y`` at every draw**, so after one exposure the rest are recall.

    Worse, `guessable_family` could not see it. That check reads the *numbers* in
    the answer, and ``-x/y`` has none, so it returned early and reported nothing.
    An answer identical across every draw is now the first thing it tests, and
    this family is what taught it to.

    The lesson survives as shape 0, which is also the item as written. The other
    three move the powers and a coefficient, so ``dy/dx`` genuinely changes and
    the chain rule has to be applied rather than remembered.

    Throughout, the misconception is the same one: ``y`` differentiated as a bare
    variable, so the ``dy/dx`` factor never appears and the ``y`` it would have
    put in the denominator is missing.
    """

    item_id = "ca_impl_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        shape = draw % 4
        step = draw // 4

        if shape == 0:
            # The item as written at draw 0, and the original lesson: dy/dx does
            # not depend on the radius.
            r = 5 + step
            return (
                f"Given x^2 + y^2 = {r * r}, find dy/dx.",
                "-x/y",
                {"without_dydx": "-x"},
            )

        if shape == 1:
            # A coefficient on y^2, which survives into the denominator.
            k, c = 2 + step, 12 + step
            return (
                f"Given x^2 + {k}y^2 = {c}, find dy/dx.",
                f"-x/({k}*y)",
                {"without_dydx": f"-x/{k}"},
            )

        if shape == 2:
            # A higher power in x, so the numerator is no longer -x. The power
            # moves with the step, or the answer would repeat every fourth draw:
            # the constant does not reach dy/dx, which is the whole point of
            # shape 0 and a defect everywhere else.
            n, c = 3 + step, 9 + step
            return (
                f"Given x^{n} + y^2 = {c}, find dy/dx.",
                f"-{n}*x**{n - 1}/(2*y)",
                {"without_dydx": f"-{n}*x**{n - 1}/2"},
            )

        # A higher power in y, so the denominator is no longer linear.
        m, c = 3 + step, 9 + step
        return (
            f"Given x^2 + y^{m} = {c}, find dy/dx.",
            f"-2*x/({m}*y**{m - 1})",
            {"without_dydx": f"-2*x/{m}"},
        )



class GeneralAntiderivative(_Template):
    """Antiderivatives whose *shape* rotates, so no surface rule answers them all.

    The first version generated only ``(n+1)·x^n``, chosen so the antiderivative's
    coefficient came out as 1 and ``+C`` was the salient thing. Optimising for
    that one feature gave everything else away: the answer to ``3x^2`` is
    ``x^3 + C`` and to ``4x^3`` is ``x^4 + C``, so **read the coefficient, that is
    your exponent** — a rule that needs no idea what an antiderivative is. A
    learner said as much at the keyboard, and the mastery estimate had believed
    them.

    More draws of that shape would not have helped; the *form* is what was being
    matched. So the draws now rotate over four shapes, each making a different
    thing the point:

    ``0`` unit coefficient, so ``+C`` is what is being tested — and it is the
          item as written, which draw 0 must always reproduce
    ``1`` a coefficient that must actually be divided by ``n+1``
    ``2`` a sum, which must be integrated term by term
    ``3`` a constant term, which is a term rather than decoration

    No single map from the prompt's numbers to the answer's numbers survives all
    four, which is what ``guessable_families`` in ``domains/validate.py`` checks.
    """

    item_id = "ca_anti_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        shape = draw % 4
        step = draw // 4

        if shape == 0:
            # Unit coefficient. Draw 0 is the item as written.
            n = 2 + step
            particular = f"x**{n + 1}"
            return (
                f"Find the general antiderivative of {n + 1}x^{n}.",
                f"{particular} + C",
                {"particular": particular},
            )

        if shape == 1:
            # The coefficient does not divide out, so it has to be divided.
            n = 3 + step
            k = 5 + step
            particular = f"{k}*x**{n + 1}/{n + 1}"
            return (
                f"Find the general antiderivative of {k}x^{n}.",
                f"{particular} + C",
                {"particular": particular},
            )

        if shape == 2:
            # Two terms. Neither coefficient predicts the other's exponent.
            n = 2 + step
            a, b = 6 + step, 4 + step
            particular = f"{a}*x**{n + 1}/{n + 1} + {b}*x**2/2"
            return (
                f"Find the general antiderivative of {a}x^{n} + {b}x.",
                f"{particular} + C",
                {"particular": particular},
            )

        # A constant term integrates to a linear one, which the surface rule
        # has no way to produce.
        n = 2 + step
        c = 7 + step
        particular = f"x**{n + 1} + {c}*x"
        return (
            f"Find the general antiderivative of {n + 1}x^{n} + {c}.",
            f"{particular} + C",
            {"particular": particular},
        )


class SubstitutionIntegral(_Template):
    """(kx + 1)^3, whose integral needs dividing by the inner derivative k."""

    item_id = "ca_usub_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        k = 2 + draw
        return (
            f"Integrate ({k}x + 1)^3 with respect to x. Omit the constant.",
            f"({k}*x + 1)**4/{4 * k}",
            {"without_du": f"({k}*x + 1)**4/4", "up_to_constant": True},
        )


TEMPLATES: Sequence[ItemTemplate] = (
    IndeterminateLimit(),
    DirectLimit(),
    AverageRate(),
    TangentAsLimit(),
    InstantaneousRate(),
    BinomialExpansion(),
    PowerRulePlain(),
    PowerRuleWithCoefficient(),
    ReciprocalPower(),
    PolynomialWithConstant(),
    StationaryPoints(),
    ProductRule(),
    QuotientRule(),
    ChainRuleCubed(),
    ChainRulePower(),
    ImplicitRelation(),
    GeneralAntiderivative(),
    SubstitutionIntegral(),
)
