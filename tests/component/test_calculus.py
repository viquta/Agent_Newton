"""The calculus domain and its symbolic verifier.

The verifier is the one component whose output must not depend on a model, so
its behaviour is pinned here in detail: what counts as the same answer, what
counts as unreadable, and what a response cannot be allowed to do.
"""

from __future__ import annotations

import pytest

from agent_newton.domains import registry
from agent_newton.domains.base import Item, Verdict
from agent_newton.domains.calculus.verifier import (
    SymbolicVerifier,
    UnparseableResponse,
    parse,
)
from agent_newton.domains.validate import NEEDS_SOURCE, UNCONFIRMED_SOURCE, validate


@pytest.fixture(scope="module")
def calculus():
    return registry.load_domain("calculus")


def item(answer: str, item_id: str = "t") -> Item:
    return Item(item_id, "power_rule", "prompt", answer)


class TestEquivalence:
    """A correct answer written differently is still correct."""

    @pytest.mark.parametrize(
        ("answer", "response"),
        [
            ("2*x", "2x"),  # implicit multiplication
            ("x**2", "x^2"),  # caret notation
            ("2*x", "x*2"),  # commuted
            ("2*sin(x)*cos(x)", "sin(2*x)"),  # trig identity
            ("(x + 1)**2", "x**2 + 2*x + 1"),  # expanded
            ("x**2 - 1", "(x - 1)*(x + 1)"),  # factored
            ("-4/x**3", "-4*x**(-3)"),  # negative exponent
            ("1/2", "0.5"),  # rational vs decimal
            ("2*x", " 2 * x "),  # whitespace
            ("x**3 + C", "x**3 + c"),  # constant of integration, either case
        ],
    )
    def test_accepts_equivalent_forms(self, calculus, answer: str, response: str) -> None:
        assert calculus.verifier.verify(item(answer), response).verdict is Verdict.CORRECT

    @pytest.mark.parametrize(
        ("answer", "response"),
        [
            ("5*x**4", "5*x**5"),  # exponent not decremented
            ("2*x*cos(x**2)", "cos(x**2)"),  # inner derivative dropped
            ("2*x", "2*y"),  # wrong variable
            ("x**3 + C", "x**3"),  # constant of integration missing
            ("2*x", "2*x + 1"),  # off by a constant
        ],
    )
    def test_rejects_wrong_answers(self, calculus, answer: str, response: str) -> None:
        assert calculus.verifier.verify(item(answer), response).verdict is Verdict.INCORRECT


class TestSolutionSets:
    """Questions asking for all roots compare unordered."""

    def test_order_does_not_matter(self, calculus) -> None:
        assert calculus.verifier.verify(item("0, 2"), "2, 0").verdict is Verdict.CORRECT

    def test_a_missing_root_is_incorrect(self, calculus) -> None:
        # The documented error: dividing through by x discards x = 0.
        assert calculus.verifier.verify(item("0, 2"), "2").verdict is Verdict.INCORRECT

    def test_an_extra_root_is_incorrect(self, calculus) -> None:
        assert calculus.verifier.verify(item("0, 2"), "0, 2, 5").verdict is Verdict.INCORRECT

    def test_equivalent_forms_within_a_set(self, calculus) -> None:
        assert calculus.verifier.verify(item("0, 2"), "0, 4/2").verdict is Verdict.CORRECT


class TestUnreadableResponses:
    @pytest.mark.parametrize("response", ["", "   ", "no idea", "x +", "2 **/ x"])
    def test_are_not_scored_as_wrong(self, calculus, response: str) -> None:
        # UNPARSEABLE is a measurement failure, not evidence about the learner:
        # the learner model must not update on it.
        result = calculus.verifier.verify(item("2*x"), response)
        assert result.verdict is Verdict.UNPARSEABLE
        assert not result.is_evidence


class TestNonExpressions:
    """Only expressions get a verdict."""

    @pytest.mark.parametrize("response", ["x > 2", "x >= 2", "x < 2"])
    def test_relations_are_unreadable_not_wrong(self, calculus, response: str) -> None:
        # A relational parses to a Boolean, not an Expr. Admitting it would send
        # a Boolean into the subtraction and produce a verdict from nonsense.
        result = calculus.verifier.verify(item("2*x"), response)
        assert result.verdict is Verdict.UNPARSEABLE
        assert not result.is_evidence


class TestResponsesAreUntrusted:
    """A response is input from a model or a person, never code to run."""

    @pytest.mark.parametrize(
        "hostile",
        [
            "__import__('os').system('echo pwned')",
            "open('/etc/passwd').read()",
            "().__class__.__bases__[0].__subclasses__()",
            "exec('x=1')",
        ],
    )
    def test_hostile_input_cannot_execute(self, calculus, hostile: str) -> None:
        # Must be refused, not evaluated. Any verdict is acceptable except one
        # reached by running the payload.
        result = calculus.verifier.verify(item("2*x"), hostile)
        assert result.verdict in (Verdict.UNPARSEABLE, Verdict.INCORRECT)

    def test_builtins_are_not_reachable(self) -> None:
        with pytest.raises(UnparseableResponse):
            parse("__import__('os')")


class TestNumericScreenRobustness:
    """The screen runs on every step; it must never take the session down."""

    def test_survives_a_pole_in_one_operand(self, calculus) -> None:
        # 1/(x-1) + 1 against 1/(x-1): the difference is a constant and finite
        # everywhere, but the operands blow up near x = 1. Evaluating the
        # operand for tolerance scaling must not escape as an exception.
        result = calculus.verifier.verify(item("1/(x - 1) + 1"), "1/(x - 1)")
        assert result.verdict is Verdict.INCORRECT

    def test_accepts_equivalent_expressions_with_poles(self, calculus) -> None:
        result = calculus.verifier.verify(item("1/(x - 1)"), "(x + 1)/((x - 1)*(x + 1))")
        assert result.verdict is Verdict.CORRECT

    def test_reports_how_many_points_were_evaluable(self) -> None:
        import random

        import sympy

        from agent_newton.domains.calculus.verifier import _numeric_screen

        x = sympy.Symbol("x")
        screen = _numeric_screen(x**2, x**2, random.Random(0))
        # Equivalent expressions: no disagreement, and the count is real
        # evidence that points were actually checked.
        assert not screen.disagrees
        assert screen.checked > 0


class TestUnverifiableIsNotAVerdict:
    """Failing to measure must not be reported as a fact about the learner."""

    def test_timeout_without_numeric_evidence_is_unparseable(
        self, calculus, monkeypatch
    ) -> None:
        import sympy

        from agent_newton.domains.calculus import verifier as module

        # No sample point evaluable, and simplify never returns: the answer has
        # been checked by nothing. Scoring it CORRECT would pass an unverified
        # response; scoring it INCORRECT would blame the learner for our failure.
        monkeypatch.setattr(
            module, "_numeric_screen", lambda a, b, rng: module._Screen(False, 0)
        )

        def _hang(*args, **kwargs):
            raise TimeoutError("simulated")

        monkeypatch.setattr(sympy, "simplify", _hang)

        result = calculus.verifier.verify(item("2*sin(x)*cos(x)"), "sin(2*x)")
        assert result.verdict is Verdict.UNPARSEABLE
        assert not result.is_evidence
        assert "could not verify" in result.detail

    def test_timeout_with_enough_evidence_is_accepted(self, calculus, monkeypatch) -> None:
        import sympy

        from agent_newton.domains.calculus import verifier as module

        monkeypatch.setattr(
            module,
            "_numeric_screen",
            lambda a, b, rng: module._Screen(False, module.MIN_SAMPLES_TO_ACCEPT),
        )

        def _hang(*args, **kwargs):
            raise TimeoutError("simulated")

        monkeypatch.setattr(sympy, "simplify", _hang)

        result = calculus.verifier.verify(item("2*sin(x)*cos(x)"), "sin(2*x)")
        assert result.verdict is Verdict.CORRECT
        # The compromise is recorded rather than hidden, so it can be audited.
        assert "simplify timed out" in result.detail

    def test_internal_failure_is_not_charged_to_the_learner(
        self, calculus, monkeypatch
    ) -> None:
        import sympy

        from agent_newton.domains.calculus import verifier as module

        monkeypatch.setattr(
            module, "_numeric_screen", lambda a, b, rng: module._Screen(False, 8)
        )
        monkeypatch.setattr(
            sympy, "simplify", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        result = calculus.verifier.verify(item("2*sin(x)*cos(x)"), "sin(2*x)")
        assert result.verdict is Verdict.UNPARSEABLE


class TestDeterminism:
    def test_same_input_gives_same_verdict(self, calculus) -> None:
        # The numeric screen samples random points; a verdict that varied
        # between runs would make a seeded cohort irreproducible.
        verdicts = {
            calculus.verifier.verify(item("2*sin(x)*cos(x)"), "sin(2*x)").verdict
            for _ in range(20)
        }
        assert verdicts == {Verdict.CORRECT}

    def test_independent_verifiers_agree(self, calculus) -> None:
        a = SymbolicVerifier().verify(item("x**2 - 1"), "(x - 1)*(x + 1)")
        b = SymbolicVerifier().verify(item("x**2 - 1"), "(x - 1)*(x + 1)")
        assert a.verdict is b.verdict is Verdict.CORRECT


class TestBuggyRules:
    def test_every_rule_produces_a_wrong_answer_where_declared(self, calculus) -> None:
        # The same check the validator runs, asserted here so a broken rule
        # fails the suite rather than only the CLI.
        for entry in calculus.items.all():
            for probed in entry.probes:
                wrong = calculus.buggy_rule(probed).apply(entry)
                assert wrong is not None, f"{entry.id} probes {probed} but produced nothing"
                verdict = calculus.verifier.verify(entry, wrong).verdict
                assert verdict is Verdict.INCORRECT, (
                    f"{entry.id}: rule {probed} produced {wrong!r}, which verified "
                    f"as {verdict.value} — the misconception would be invisible"
                )

    def test_rules_are_deterministic(self, calculus) -> None:
        entry = calculus.items.get("ca_chain_p1")
        rule = calculus.buggy_rule("chain_rule_omits_inner")
        assert len({rule.apply(entry) for _ in range(10)}) == 1

    def test_chain_rule_error_is_the_documented_one(self, calculus) -> None:
        # d/dx sin(x^2) given as cos(x^2): the outer derivative alone.
        entry = calculus.items.get("ca_chain_p1")
        assert calculus.buggy_rule("chain_rule_omits_inner").apply(entry) == "cos(x**2)"


class TestDomainContent:
    def test_validates(self, calculus) -> None:
        report = validate(calculus)
        assert report.ok, "\n".join(str(p) for p in report.problems)

    def test_unconfirmed_sources_are_warnings_not_failures(self, calculus) -> None:
        # These entries are usable but not yet backed by a study documenting the
        # error. They must stay visible without blocking work.
        report = validate(calculus)
        assert report.warnings
        assert all(w.check == UNCONFIRMED_SOURCE for w in report.warnings)
        assert report.ok

    def test_every_source_is_either_a_citation_or_flagged(self, calculus) -> None:
        for misconception in calculus.misconceptions.all():
            source = misconception.source.strip()
            assert source, misconception.id
            if not source.upper().startswith(NEEDS_SOURCE):
                # A real citation carries a year.
                assert any(ch.isdigit() for ch in source), misconception.id

    def test_graph_reaches_the_advanced_topics(self, calculus) -> None:
        prerequisites = calculus.concepts.all_prerequisites("integration_by_substitution")
        assert {"chain_rule", "power_rule", "limits_of_sequences"} <= prerequisites
