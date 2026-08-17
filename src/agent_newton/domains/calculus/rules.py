"""Buggy rules: how a simulated learner errs under each calculus misconception.

Each rule computes from ``item.params`` — the item's structured form — rather
than parsing ``item.prompt``. Rewording a prompt therefore never changes learner
behaviour, and a seeded cohort stays exactly reproducible.

Returning ``None`` means the item cannot elicit this misconception. The domain
validator checks the converse: every misconception an item claims to probe must
actually yield a response the verifier judges incorrect.
"""

from __future__ import annotations

from typing import Sequence

from agent_newton.domains.base import BuggyRule, Item


def _need(item: Item, *keys: str) -> tuple | None:
    """Fetch required params, or None if any is absent."""
    values = tuple(item.params.get(k) for k in keys)
    return None if any(v is None for v in values) else values


class BinomialMiddleTermLost:
    """(x + h)^2 expanded as x^2 + h^2 — the 2xh term is dropped."""

    misconception_id = "binomial_middle_term_lost"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "base", "step", "exponent")
        if got is None:
            return None
        base, step, exponent = got
        return f"{base}**{exponent} + {step}**{exponent}"


class CancelXLosesRoot:
    """Dividing a stationary-point equation through by x, discarding x = 0."""

    misconception_id = "cancel_x_loses_root"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "nonzero_root")
        return None if got is None else str(got[0])


class AverageForInstantaneousRate:
    """The average rate over [x0, x0 + h] reported as the rate at x0."""

    misconception_id = "average_for_instantaneous_rate"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "average_rate")
        return None if got is None else str(got[0])


class ReciprocalPowerNotRewritten:
    """d/dx (a/x^n) computed as though it were a·x^n."""

    misconception_id = "reciprocal_power_not_rewritten"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "coefficient", "exponent")
        if got is None:
            return None
        coefficient, exponent = got
        # Power rule applied with the exponent's sign never flipped.
        return f"{int(coefficient) * int(exponent)}*x**{int(exponent) - 1}"


class IndeterminateLimitReportedAsZero:
    """An indeterminate form reported as zero."""

    misconception_id = "indeterminate_limit_reported_as_zero"

    def apply(self, item: Item) -> str | None:
        return "0" if item.params.get("indeterminate") else None


class ChainRuleOmitsInner:
    """The outer derivative only — the inner function's derivative is dropped."""

    misconception_id = "chain_rule_omits_inner"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "outer_derivative")
        return None if got is None else str(got[0])


class ImplicitDiffOmitsDydx:
    """y differentiated as a bare variable, so no dy/dx factor is produced."""

    misconception_id = "implicit_diff_omits_dydx"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "without_dydx")
        return None if got is None else str(got[0])


class PowerRuleForgetsDecrement:
    """d/dx x^n given as n·x^n — the exponent is not decremented."""

    misconception_id = "power_rule_forgets_decrement"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "coefficient", "exponent")
        if got is None:
            return None
        coefficient, exponent = got
        return f"{int(coefficient) * int(exponent)}*x**{int(exponent)}"


class ProductRuleAsProductOfDerivatives:
    """(fg)' taken to be f'·g'."""

    misconception_id = "product_rule_as_product_of_derivatives"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "df", "dg")
        if got is None:
            return None
        df, dg = got
        return f"({df})*({dg})"


class QuotientRuleNumeratorNotDifferentiated:
    """(f/g)' with u' replaced by u in the first numerator term.

    Distinct from transposing the terms: the structure of the rule is right and
    one factor was simply not differentiated. A person made exactly this error and
    was told the terms were transposed, because that was the only label the
    concept had — so the diagnosis was wrong while the verdict was right.
    """

    misconception_id = "quotient_rule_numerator_not_differentiated"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "f", "g", "dg")
        if got is None:
            return None
        f, g, dg = got
        return f"(({f})*({g}) - ({f})*({dg}))/({g})**2"


class QuotientRuleOrderSwapped:
    """(f/g)' with the numerator terms transposed."""

    misconception_id = "quotient_rule_order_swapped"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "f", "g", "df", "dg")
        if got is None:
            return None
        f, g, df, dg = got
        return f"(({f})*({dg}) - ({df})*({g}))/({g})**2"


class UsubForgetsDu:
    """The antiderivative without the reciprocal of the inner derivative."""

    misconception_id = "usub_forgets_du"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "without_du")
        return None if got is None else str(got[0])


class AntiderivativeOmitsConstant:
    """A particular antiderivative given where the general one was asked for."""

    misconception_id = "antiderivative_omits_constant"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "particular")
        return None if got is None else str(got[0])


class RateAsDifferenceNotRatio:
    """f(b) - f(a) reported as the average rate, never divided by b - a."""

    misconception_id = "rate_as_difference_not_ratio"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "difference")
        return None if got is None else str(got[0])


class TangentAsSecant:
    """A secant gradient at a finite separation given as the tangent's."""

    misconception_id = "tangent_as_secant"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "secant_gradient")
        return None if got is None else str(got[0])


class ConstantTermDifferentiated:
    """The constant term carried into the derivative instead of vanishing."""

    misconception_id = "constant_term_differentiated"

    def apply(self, item: Item) -> str | None:
        got = _need(item, "constant")
        if got is None:
            return None
        # Built from the item's own answer rather than from a second copy of
        # the derivative in params. The ban in this module is on parsing
        # ``prompt``, which is prose; ``answer`` is already structured.
        return f"{item.answer} + ({got[0]})"


RULES: Sequence[BuggyRule] = (
    BinomialMiddleTermLost(),
    CancelXLosesRoot(),
    AverageForInstantaneousRate(),
    ReciprocalPowerNotRewritten(),
    IndeterminateLimitReportedAsZero(),
    ChainRuleOmitsInner(),
    ImplicitDiffOmitsDydx(),
    PowerRuleForgetsDecrement(),
    ProductRuleAsProductOfDerivatives(),
    QuotientRuleNumeratorNotDifferentiated(),
    QuotientRuleOrderSwapped(),
    UsubForgetsDu(),
    AntiderivativeOmitsConstant(),
    RateAsDifferenceNotRatio(),
    TangentAsSecant(),
    ConstantTermDifferentiated(),
)
