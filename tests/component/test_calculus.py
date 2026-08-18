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
    _tighter_denominator,
    parse,
)
from agent_newton.domains.validate import (
    ANSWERS_ARE_UNAMBIGUOUS,
    NEEDS_SOURCE,
    RESOURCE_DRAWS,
    UNCONFIRMED_SOURCE,
    VARIANT_DRAWS,
    validate,
)


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
        # d/dx (x^2 + 1)^3 given as 3(x^2 + 1)^2: the outer derivative alone,
        # with the inner derivative 2x dropped.
        entry = calculus.items.get("ca_chain_p1")
        assert (
            calculus.buggy_rule("chain_rule_omits_inner").apply(entry)
            == "3*(x**2 + 1)**2"
        )


class TestDomainContent:
    def test_validates(self, calculus) -> None:
        report = validate(calculus)
        assert report.ok, "\n".join(str(p) for p in report.problems)

    def test_unconfirmed_sources_are_warnings_not_failures(self, calculus) -> None:
        # These entries are usable but not yet backed by a study documenting the
        # error. They must stay visible without blocking work.
        report = validate(calculus)
        assert report.warnings
        # The unconfirmed sources are reported, and reported as warnings.
        #
        # This asserted `all(...)` until a second warning kind existed, which
        # made it a test of how many kinds there are rather than of how this one
        # is classified. `guessable_family` then arrived and broke it while
        # nothing about sources had changed.
        assert any(w.check == UNCONFIRMED_SOURCE for w in report.warnings)
        assert not any(p.check == UNCONFIRMED_SOURCE for p in report.problems)
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


#: Functions no concept in this graph teaches. An item using one asks the
#: learner for knowledge the syllabus never supplies, and the planner cannot
#: route to a gap that is not a concept.
UNTAUGHT = ("sin", "cos", "tan", "log", "ln", "exp", "sqrt")


class TestItemsStayInsideTheGraph:
    """No item may assume knowledge the concept graph does not contain.

    A person met ``y = x^2 sin(x)`` under the product rule and asked three
    times, in three channels, what sin(x) was. Nothing in the graph teaches the
    derivatives of trig functions, so the planner had no concept to route to and
    the tutor could only keep explaining the product rule. The session could not
    help them, and no amount of tutoring quality would have changed that.

    ``domain validate`` checks that every concept on a goal's route has practice
    items. It cannot check this: it has no way to know which functions the graph
    covers. So the list is stated here, next to the domain it describes.
    """

    def test_no_item_uses_an_untaught_function(self, calculus) -> None:
        offenders = [
            (item.id, name)
            for item in calculus.items.all()
            for name in UNTAUGHT
            if f"{name}(" in item.prompt or f"{name}(" in item.answer
        ]
        assert not offenders, f"items assuming untaught functions: {offenders}"

    def test_no_generated_variant_reintroduces_one(self, calculus) -> None:
        # The templates regenerate the prompt and answer, so an item cleaned up
        # in the YAML could still produce a trig variant on the second asking.
        offenders = [
            (item.id, draw, name)
            for item in calculus.items.bank("practice")
            for draw in range(8)
            for name in UNTAUGHT
            if f"{name}(" in calculus.variant(item, draw).prompt
            or f"{name}(" in calculus.variant(item, draw).answer
        ]
        assert not offenders, f"variants assuming untaught functions: {offenders}"

    def test_the_check_can_fail(self, calculus) -> None:
        # A guard that cannot fail proves nothing. This is the shape it looks
        # for, and it must be caught.
        from dataclasses import replace

        stray = replace(calculus.items.bank("practice")[0], answer="cos(x)")
        assert any(f"{name}(" in stray.answer for name in UNTAUGHT)


class TestEveryVariantAcceptsEquivalentAnswers:
    """The gold set covers the 17 *written* items. Variants had no such cover.

    `domain validate` checks that each draw's stated answer verifies correct.
    Nothing checked whether a learner who writes that same answer *differently*
    is accepted — and a false negative on a variant is the worst kind, because
    unlike `UNPARSEABLE` it reaches the learner model as evidence of an error the
    learner did not make.

    Hand-labelling 18 templates × 8 draws is not the way. Equivalent forms are
    generated mechanically instead: sympy's own `expand`, `factor`, `simplify`,
    `together` and `cancel` all preserve equivalence by construction, so anything
    they produce *must* be accepted. That makes this a property test rather than a
    fixture, and it grows with the templates.

    ⚠️ `UNPARSEABLE` is tolerated and counted separately, on the same grounds the
    gold set uses: the verifier failing to read an answer updates no estimate and
    enters no error trace, so it is a measurement failure rather than a false
    accusation.
    """

    _TRANSFORMS = ("expand", "factor", "simplify", "together", "cancel")

    def _equivalent_forms(self, answer: str) -> list[tuple[str, str]]:
        import sympy
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        rules = standard_transformations + (
            convert_xor,
            implicit_multiplication_application,
        )
        try:
            expression = parse_expr(answer, transformations=rules)
        except Exception:  # noqa: BLE001 - an unparseable answer has no forms to try
            return []

        forms = []
        for name in self._TRANSFORMS:
            try:
                rewritten = str(getattr(sympy, name)(expression))
            except Exception:  # noqa: BLE001 - sympy declines on some shapes
                continue
            if rewritten != answer:
                forms.append((name, rewritten))
        return forms

    def test_no_variant_scores_an_equivalent_answer_as_wrong(self, calculus) -> None:
        false_negatives = []
        checked = 0
        for item_id, template in sorted(calculus.templates.items()):
            base = calculus.items.get(item_id)
            for draw in range(VARIANT_DRAWS):
                variant = template.variant(base, draw)
                for name, form in self._equivalent_forms(variant.answer):
                    checked += 1
                    if calculus.verifier.verify(variant, form).verdict is Verdict.INCORRECT:
                        false_negatives.append(
                            f"{item_id} draw {draw} via {name}: "
                            f"{variant.answer!r} rewritten as {form!r} scored INCORRECT"
                        )

        assert checked > 100, "the generator produced too few forms to be a real check"
        assert not false_negatives, "\n".join(false_negatives)

    def test_the_unreadable_ones_are_only_the_known_root_list(self, calculus) -> None:
        # `ca_stat_p1`'s answer is a list of roots ("0, 2"), which sympy reads as a
        # tuple rather than an expression — the same limitation the gold set
        # already records for `x = 0 or x = 2`. Pinned to that one item so a new
        # unreadable family shows up as a change here rather than as silence.
        unreadable = set()
        for item_id, template in sorted(calculus.templates.items()):
            base = calculus.items.get(item_id)
            for draw in range(VARIANT_DRAWS):
                variant = template.variant(base, draw)
                for _, form in self._equivalent_forms(variant.answer):
                    if calculus.verifier.verify(variant, form).verdict is Verdict.UNPARSEABLE:
                        unreadable.add(item_id)
        assert unreadable == {"ca_stat_p1"}, (
            f"unreadable equivalent forms outside the known root-list case: "
            f"{sorted(unreadable - {'ca_stat_p1'})}"
        )


class TestNotationWithTwoReadings:
    """``a/bc`` means ``(a/b)*c`` formally and ``a/(bc)`` in ordinary writing.

    Found in a sitting. The learner wrote ``-x/3y`` for an item whose answer is
    ``-x/(3*y)``, was told they were wrong, wrote ``-2x/6y``, and was told again
    — and each verdict lowered the estimate, entered the error trace and
    produced ``implicit_diff_omits_dydx``, naming an error they had not made.
    The tutor then aimed a hint at it.

    Neither reading is wrong, which is exactly why the verifier must not choose
    one. It reports that it could not measure, for the reason prose does.
    """

    def _impl(self, calculus):
        # The variant the sitting met: answer -x/(3*y).
        base = calculus.items.get("ca_impl_p1")
        return calculus.templates["ca_impl_p1"].variant(base, 5)

    def test_the_response_from_the_sitting(self, calculus) -> None:
        found = SymbolicVerifier().verify(self._impl(calculus), "-x/3y")
        assert found.verdict is Verdict.UNPARSEABLE
        assert "brackets" in found.detail

    def test_and_the_unsimplified_one_that_followed(self, calculus) -> None:
        found = SymbolicVerifier().verify(self._impl(calculus), "-2x/6y")
        assert found.verdict is Verdict.UNPARSEABLE

    def test_it_costs_nothing_and_says_nothing_about_the_learner(
        self, calculus
    ) -> None:
        # The property that makes this the right verdict rather than a lenient
        # one: UNPARSEABLE is not evidence, so no estimate moves, no error event
        # is written and no diagnosis is asked for.
        found = SymbolicVerifier().verify(self._impl(calculus), "-x/3y")
        assert not found.is_evidence

    def test_brackets_settle_it_either_way(self, calculus) -> None:
        item = self._impl(calculus)
        assert SymbolicVerifier().verify(item, "-x/(3y)").verdict is Verdict.CORRECT
        assert SymbolicVerifier().verify(item, "-x/(3*y)").verdict is Verdict.CORRECT

    def test_an_explicit_product_has_one_reading_and_keeps_its_verdict(
        self, calculus
    ) -> None:
        # `-x/3*y` is unambiguous and means (-x/3)*y. A learner who wrote that
        # wrote something else, and nothing here rescues them.
        found = SymbolicVerifier().verify(self._impl(calculus), "-x/3*y")
        assert found.verdict is Verdict.INCORRECT

    def test_wrong_under_both_readings_stays_wrong(self, calculus) -> None:
        # The boundary. Ambiguity is reported only when it decides the verdict;
        # where both readings are wrong the notation is not the reason, and
        # converting this to unreadable would lose a real error.
        found = SymbolicVerifier().verify(self._impl(calculus), "-2x/3y^2")
        assert found.verdict is Verdict.INCORRECT

    def test_the_message_does_not_say_which_reading_is_right(
        self, calculus
    ) -> None:
        """This verdict costs no attempt, so saying would be a free answer.

        The same failure a worked step once had: the reply assembled the answer
        and all that was left was to type it back.
        """
        found = SymbolicVerifier().verify(self._impl(calculus), "-x/3y")
        assert "-x/(3y)" in found.detail  # the bracketing is shown
        for tell in ("right", "correct", "is the answer"):
            assert tell not in found.detail.lower()

    @pytest.mark.parametrize(
        "text",
        [
            "-x/(3*y)",  # already bracketed
            "-x/3*y",  # explicit product
            "x/y**2",  # one atom, exponent included
            "(2*x*(x + 1) - x**2)/(x + 1)**2",  # a real item answer
            "x**3 + C",
            "5*x**4",
        ],
    )
    def test_unambiguous_forms_are_left_alone(self, text: str) -> None:
        assert _tighter_denominator(text) is None

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("-x/3y", "-x/(3y)"),
            ("-2x/6y", "-2x/(6y)"),
            ("x/2y + 3", "x/(2y) + 3"),
            ("sin(x)/2x", "sin(x)/(2x)"),
        ],
    )
    def test_the_other_reading_is_what_a_reader_would_take(
        self, text: str, expected: str
    ) -> None:
        assert _tighter_denominator(text) == expected

    def test_dy_dx_is_notation_and_not_a_division(self) -> None:
        # It would otherwise read as d*y divided by d*x and be "ambiguous" on
        # every implicit-differentiation answer in the bank.
        assert _tighter_denominator("dy/dx") is None


class TestNoSimulatedLearnerCanWriteAnAmbiguousAnswer:
    """Which is what makes the rule above inert for every measured result.

    An INCORRECT becoming UNPARSEABLE would stop being evidence: BKT would not
    update and no error event would be written. If a buggy rule could produce
    one, every cohort number would move. None can — the rules all emit explicit
    products — and this asserts it rather than trusting it.
    """

    def test_no_item_answer_or_rule_output_has_two_readings(self, calculus) -> None:
        checked = 0
        for item in calculus.items.all():
            forms = [item]
            template = calculus.templates.get(item.id)
            if template is not None:
                # Deeper than `domain validate` goes, because this costs nothing
                # — no verifier calls, only the scanner — and a family is
                # unbounded, so more draws is strictly more evidence.
                forms += [
                    template.variant(item, draw) for draw in range(1, RESOURCE_DRAWS)
                ]
            for form in forms:
                texts = [form.answer]
                for label in form.probes:
                    rule = calculus.buggy_rule(label)
                    if rule is not None:
                        texts.append(rule.apply(form))
                for text in texts:
                    if not text:
                        continue
                    checked += 1
                    assert _tighter_denominator(text) is None, (
                        f"{form.id} can produce {text!r}, which has two readings "
                        f"— a simulated learner would now go unmeasured on it"
                    )
        assert checked > 2000, "the sweep covered too little to mean anything"


class TestAnItemMayNotShipAnAmbiguousAnswer:
    """A learner is asked to bracket. An item has no such excuse.

    It is the thing being compared against, so an ambiguous one would be
    compared under whichever reading the parser happened to take.
    """

    def test_the_bank_is_clean(self, calculus) -> None:
        assert not [
            p for p in validate(calculus).problems if p.check == ANSWERS_ARE_UNAMBIGUOUS
        ]

    def test_the_check_can_fail(self, calculus) -> None:
        from dataclasses import replace

        from agent_newton.domains.content import YamlItemBank

        stray = replace(calculus.items.get("ca_impl_p1"), answer="-x/3y")
        broken = replace(
            calculus,
            items=YamlItemBank(
                [stray]
                + [i for i in calculus.items.all() if i.id != "ca_impl_p1"]
            ),
        )
        report = validate(broken)
        assert not report.ok
        assert [p for p in report.problems if p.check == ANSWERS_ARE_UNAMBIGUOUS]
