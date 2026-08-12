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
    """y = (x^p + 1)(x^q + 2). Two polynomials, so nothing outside the graph."""

    item_id = "ca_prod_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        p, q = 2, 3 + draw
        df, dg = _term(p, p - 1), _term(q, q - 1)
        return (
            f"Differentiate: y = (x^{p} + 1)(x^{q} + 2)",
            f"{df}*(x**{q} + 2) + (x**{p} + 1)*{dg}",
            {"f": f"x**{p} + 1", "g": f"x**{q} + 2", "df": df, "dg": dg},
        )


class QuotientRule(_Template):
    """y = x^2/(x + c)."""

    item_id = "ca_quot_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        c = 1 + draw
        return (
            f"Differentiate: y = x^2/(x + {c})",
            f"(2*x*(x + {c}) - x**2)/(x + {c})**2",
            {"f": "x**2", "g": f"x + {c}", "df": "2*x", "dg": "1"},
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


class ImplicitCircle(_Template):
    """x^2 + y^2 = r^2. The radius changes; dy/dx does not, which is the lesson."""

    item_id = "ca_impl_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        r = 5 + draw
        return (
            f"Given x^2 + y^2 = {r * r}, find dy/dx.",
            "-x/y",
            {"without_dydx": "-x"},
        )


class GeneralAntiderivative(_Template):
    """(n+1)·x^n, chosen so the antiderivative's coefficient is 1 and +C is the point."""

    item_id = "ca_anti_p1"

    def parts(self, draw: int) -> tuple[str, str, dict]:
        n = 2 + draw
        return (
            f"Find the general antiderivative of {n + 1}x^{n}.",
            f"x**{n + 1} + C",
            {"particular": f"x**{n + 1}"},
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
    ImplicitCircle(),
    GeneralAntiderivative(),
    SubstitutionIntegral(),
)
