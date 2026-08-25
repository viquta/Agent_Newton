"""Domain content loading, the registry, and the validator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from agent_newton.domains import registry
from agent_newton.domains.base import (
    BuggyRule,
    Concept,
    DomainError,
    Item,
    Verdict,
    Verifier,
)
from agent_newton.domains.content import YamlConceptGraph
from agent_newton.domains.validate import (
    ANSWERS_VERIFY,
    CONCEPT_HAS_A_LABEL,
    CONCEPT_HAS_A_LESSON,
    CONCEPT_HAS_A_RESOURCE,
    GOALS_ARE_REACHABLE,
    RESOURCE_IS_PLAIN_TEXT,
    RESOURCE_KEEPS_ITS_DISTANCE,
    RULES_PRODUCE_ERRORS,
    GUESSABLE_FAMILY,
    TEMPLATES_ARE_SOUND,
    validate,
)


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


@pytest.fixture(scope="module")
def calculus():
    return registry.load_domain("calculus")


class TestRegistry:
    def test_lists_available_domains(self) -> None:
        assert "toy_algebra" in registry.available()

    def test_rejects_unknown_domain_with_options(self) -> None:
        with pytest.raises(DomainError, match="available: "):
            registry.load_domain("phrenology")

    def test_loading_toy_algebra_does_not_import_sympy(self) -> None:
        # Lazy builders keep the CI path and the threshold sweep off sympy's
        # import cost. If this fails, the registry started importing eagerly.
        import subprocess
        import sys

        code = (
            "import sys;"
            "from agent_newton.domains import registry;"
            "registry.load_domain('toy_algebra');"
            "print('sympy' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"


class TestConceptGraph:
    def test_topological_order_respects_prerequisites(self, toy) -> None:
        order = list(toy.concepts.topological_order())
        for concept in toy.concepts.concepts():
            for prereq in concept.prerequisites:
                assert order.index(prereq) < order.index(concept.id)

    def test_transitive_prerequisites(self, toy) -> None:
        # solve_linear <- distribute <- combine_like_terms <- integer_arithmetic
        assert "integer_arithmetic" in toy.concepts.all_prerequisites("solve_linear")
        assert "integer_arithmetic" not in toy.concepts.prerequisites("solve_linear")

    def test_roots_have_depth_zero(self, toy) -> None:
        assert toy.concepts.depth("integer_arithmetic") == 0

    def test_depth_is_longest_path_not_shortest(self) -> None:
        # d has a short route (via b) and a long one (via c). Depth must report
        # the long one, or the frontier fallback would prefer a concept whose
        # prerequisites run deeper than it appears.
        graph = YamlConceptGraph(
            [
                Concept("a", "a"),
                Concept("b", "b", ("a",)),
                Concept("c", "c", ("b",)),
                Concept("d", "d", ("a", "c")),
            ]
        )
        assert graph.depth("d") == 3

    def test_rejects_cycles(self) -> None:
        with pytest.raises(DomainError, match="cycle"):
            YamlConceptGraph([Concept("a", "a", ("b",)), Concept("b", "b", ("a",))])

    def test_rejects_dangling_prerequisite(self) -> None:
        with pytest.raises(DomainError, match="unknown prerequisite"):
            YamlConceptGraph([Concept("a", "a", ("ghost",))])

    def test_rejects_duplicate_ids(self) -> None:
        with pytest.raises(DomainError, match="duplicate concept id"):
            YamlConceptGraph([Concept("a", "one"), Concept("a", "two")])


class TestGoals:
    """What the planner works toward.

    A goal narrows planning to its prerequisite closure, so a wrong one does not
    fail loudly — it quietly changes what the learner is ever offered.
    """

    CHAIN = (Concept("a", "a"), Concept("b", "b", ("a",)), Concept("c", "c", ("b",)))

    def test_declared_goals_keep_their_order(self) -> None:
        # Order is the curriculum decision: the planner takes the first goal not
        # yet mastered, so sorting these would silently retarget the run.
        graph = YamlConceptGraph(self.CHAIN, goals=("c", "b"))
        assert list(graph.goals()) == ["c", "b"]

    def test_sinks_are_the_default(self) -> None:
        # A concept nothing depends on is where a path through the graph ends.
        graph = YamlConceptGraph(self.CHAIN)
        assert list(graph.goals()) == ["c"]

    def test_every_sink_is_a_default_goal(self) -> None:
        graph = YamlConceptGraph(
            (Concept("a", "a"), Concept("b", "b", ("a",)), Concept("c", "c", ("a",)))
        )
        assert set(graph.goals()) == {"b", "c"}

    def test_an_unknown_goal_is_refused(self) -> None:
        with pytest.raises(DomainError, match="unknown goal concept"):
            YamlConceptGraph(self.CHAIN, goals=("ghost",))

    def test_the_shipped_domains_declare_theirs(self) -> None:
        assert list(registry.load_domain("toy_algebra").concepts.goals()) == ["solve_linear"]
        calculus = list(registry.load_domain("calculus").concepts.goals())
        assert calculus[0] == "negative_fractional_exponents"
        assert calculus[-1] == "integration_by_substitution"

    def test_a_nearer_goal_is_relevant_to_fewer_concepts(self) -> None:
        # The point of ordering: planning starts narrow and widens.
        graph = registry.load_domain("calculus").concepts
        near = len(graph.all_prerequisites("negative_fractional_exponents"))
        far = len(graph.all_prerequisites("integration_by_substitution"))
        assert near < far


class TestContentHashes:
    def test_are_stable_across_loads(self) -> None:
        assert (
            registry.load_domain("toy_algebra").content_hashes()
            == registry.load_domain("toy_algebra").content_hashes()
        )

    def test_are_insensitive_to_declaration_order(self) -> None:
        # Reordering YAML must not invalidate comparability between runs.
        forward = YamlConceptGraph([Concept("a", "a"), Concept("b", "b", ("a",))])
        reverse = YamlConceptGraph([Concept("b", "b", ("a",)), Concept("a", "a")])
        assert forward.content_hash() == reverse.content_hash()

    def test_change_when_content_changes(self) -> None:
        original = YamlConceptGraph([Concept("a", "a"), Concept("b", "b", ("a",))])
        edited = YamlConceptGraph([Concept("a", "a"), Concept("b", "b")])
        assert original.content_hash() != edited.content_hash()

    def test_change_when_the_goal_changes(self) -> None:
        # Retargeting a run changes what it measured, so analysis must refuse to
        # pool across the change rather than average two different studies.
        concepts = [Concept("a", "a"), Concept("b", "b", ("a",))]
        assert (
            YamlConceptGraph(concepts, goals=("a",)).content_hash()
            != YamlConceptGraph(concepts, goals=("b",)).content_hash()
        )

    def test_change_when_the_goal_order_changes(self) -> None:
        concepts = [Concept("a", "a"), Concept("b", "b")]
        assert (
            YamlConceptGraph(concepts, goals=("a", "b")).content_hash()
            != YamlConceptGraph(concepts, goals=("b", "a")).content_hash()
        )


class TestToyAlgebraProtocolConformance:
    def test_verifier_satisfies_the_protocol(self, toy) -> None:
        assert isinstance(toy.verifier, Verifier)

    def test_rules_satisfy_the_protocol(self, toy) -> None:
        assert toy.buggy_rules
        for rule in toy.buggy_rules.values():
            assert isinstance(rule, BuggyRule)

    def test_verifier_is_not_a_cas(self) -> None:
        # toy_algebra deliberately avoids sympy so the Verifier Protocol is
        # shown to work for non-CAS subjects too. Checked against the module's
        # actual imports, not its text — prose about sympy is fine.
        import ast

        import agent_newton.domains.toy_algebra.verifier as module

        tree = ast.parse(Path(module.__file__).read_text())
        imported = {
            name.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for name in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "sympy" not in imported


class TestNormalizingVerifier:
    @pytest.mark.parametrize(
        ("answer", "response"),
        [
            ("3x + 12", "3x + 12"),
            ("3x + 12", "12 + 3x"),  # order-insensitive
            ("3x + 12", "3*x+12"),  # explicit multiplication
            ("3x + 12", " 3X  +  12 "),  # whitespace and case
            ("4x + 5", "5+4x"),
        ],
    )
    def test_accepts_equivalent_forms(self, toy, answer: str, response: str) -> None:
        item = Item("t", "distribute", "p", answer)
        assert toy.verifier.verify(item, response).verdict is Verdict.CORRECT

    @pytest.mark.parametrize("response", ["3x + 4", "9x", "7", "3x"])
    def test_rejects_wrong_answers(self, toy, response: str) -> None:
        item = Item("t", "distribute", "p", "3x + 12")
        assert toy.verifier.verify(item, response).verdict is Verdict.INCORRECT

    @pytest.mark.parametrize("response", ["", "   ", "I don't know", "3x +"])
    def test_reports_unreadable_responses_separately(self, toy, response: str) -> None:
        # Not INCORRECT: an unreadable response is a measurement failure and
        # must not update the learner model.
        item = Item("t", "distribute", "p", "3x + 12")
        result = toy.verifier.verify(item, response)
        assert result.verdict is Verdict.UNPARSEABLE
        assert not result.is_evidence


class TestBuggyRules:
    @pytest.mark.parametrize(
        ("misconception", "item", "expected"),
        [
            (
                "distribute_first_term_only",
                Item("i", "distribute", "Expand: 3(x + 4)", "3x + 12", params={"a": 3, "c": 4}),
                "3x + 4",
            ),
            (
                "combine_unlike_terms",
                Item("i", "combine_like_terms", "4x + 5", "4x + 5", params={"a": 4, "c": 5}),
                "9x",
            ),
            (
                "sign_error_moving_term",
                Item("i", "solve_linear", "x + 5 = 12", "7", params={"form": "add", "c": 5, "rhs": 12}),
                "17",
            ),
            (
                "drop_coefficient_when_solving",
                Item("i", "solve_linear", "3x = 12", "4", params={"form": "mul", "a": 3, "rhs": 12}),
                "12",
            ),
        ],
    )
    def test_produce_the_documented_error(
        self, toy, misconception: str, item: Item, expected: str
    ) -> None:
        assert toy.buggy_rule(misconception).apply(item) == expected

    def test_are_deterministic(self, toy) -> None:
        # A seeded cohort is only reproducible if rules are.
        item = Item("i", "distribute", "p", "3x + 12", params={"a": 3, "c": 4})
        rule = toy.buggy_rule("distribute_first_term_only")
        assert len({rule.apply(item) for _ in range(10)}) == 1

    def test_return_none_when_item_lacks_params(self, toy) -> None:
        bare = Item("i", "distribute", "p", "3x + 12")
        assert toy.buggy_rule("distribute_first_term_only").apply(bare) is None

    def test_return_none_for_wrong_equation_form(self, toy) -> None:
        multiplicative = Item("i", "solve_linear", "p", "4", params={"form": "mul", "rhs": 12})
        assert toy.buggy_rule("sign_error_moving_term").apply(multiplicative) is None


class TestValidator:
    def test_toy_algebra_is_clean(self, toy) -> None:
        report = validate(toy)
        assert report.ok, "\n".join(str(p) for p in report.problems)

    def test_reports_useful_stats(self, toy) -> None:
        stats = validate(toy).stats
        assert stats["concepts"] == 5
        assert stats["misconceptions"] == 4
        assert stats["pretest"] and stats["posttest"]

    def test_catches_an_answer_that_does_not_verify(self, toy) -> None:
        broken = replace(toy, items=_ItemBankStub([Item("bad", "distribute", "p", "not an answer")]))
        problems = validate(broken).problems
        assert any(p.check == ANSWERS_VERIFY for p in problems)

    def test_catches_a_rule_that_produces_the_correct_answer(self, toy) -> None:
        # The dangerous case: a mis-written buggy transform yields the right
        # answer, so the misconception is invisible to the diagnostic agent and
        # accuracy is silently measured against a catalogue with a dead label.
        item = Item(
            "sneaky",
            "distribute",
            "Expand: 3(x + 4)",
            "3x + 4",  # states the *buggy* result as correct
            probes=("distribute_first_term_only",),
            params={"a": 3, "c": 4},
        )
        broken = replace(toy, items=_ItemBankStub([item]))
        problems = validate(broken).problems
        assert any(p.check == RULES_PRODUCE_ERRORS for p in problems)

    def test_catches_probe_with_no_usable_params(self, toy) -> None:
        item = Item("np", "distribute", "p", "3x + 12", probes=("distribute_first_term_only",))
        broken = replace(toy, items=_ItemBankStub([item]))
        assert any(p.check == RULES_PRODUCE_ERRORS for p in validate(broken).problems)

    def test_catches_misconception_missing_from_the_posttest(self, toy) -> None:
        practice_only = [i for i in toy.items.all() if i.bank != "posttest"]
        broken = replace(toy, items=_ItemBankStub(practice_only))
        problems = validate(broken).problems
        assert any(p.check == "misconception_probed_in_tests" for p in problems)

    def test_warns_about_a_concept_with_no_misconception(self, toy) -> None:
        # Silent until a human session produced a diagnosis naming a
        # misconception from a different concept entirely. Every label offered
        # for a wrong answer on such a concept belongs to another one, so the
        # agent picks the nearest plausible fit rather than abstaining.
        warnings = validate(toy).warnings
        assert any(p.check == CONCEPT_HAS_A_LABEL for p in warnings)

    def test_the_warning_is_not_raised_when_every_concept_has_one(self) -> None:
        # The guard must be able to stay quiet, or it says nothing about the
        # domain it is checking. Calculus is the case where it does.
        calculus = registry.load_domain("calculus")
        assert not [
            p for p in validate(calculus).warnings if p.check == CONCEPT_HAS_A_LABEL
        ]

    def test_a_concept_with_no_practice_items_is_not_warned_about(self, toy) -> None:
        # Nothing can be asked about it, so nothing can be misdiagnosed on it.
        labelled = {m.concept_id for m in toy.misconceptions.all()}
        unlabelled = [c for c in toy.concepts.ids() if c not in labelled]
        assert unlabelled, "the fixture no longer has an unlabelled concept"
        broken = replace(
            toy,
            items=_ItemBankStub(
                [i for i in toy.items.all() if i.concept_id not in unlabelled]
            ),
        )
        assert not [
            p for p in validate(broken).warnings if p.check == CONCEPT_HAS_A_LABEL
        ]

    def test_counts_the_goals(self, toy) -> None:
        assert validate(toy).stats["goals"] == 1

    def test_catches_a_template_that_does_not_reproduce_its_item(self, toy) -> None:
        # The property everything else rests on. If draw 0 differs from the
        # YAML, the file stops being the definition and anything naming the item
        # by id — the verifier gold set, a stored transcript — silently changes
        # meaning.
        item = toy.items.bank("practice")[0]

        class Drifting:
            item_id = item.id

            def variant(self, base, draw):  # noqa: ANN001
                return replace(base, prompt=f"{base.prompt} (draw {draw})")

        broken = replace(toy, templates={item.id: Drifting()})
        # `Domain.variant` short-circuits draw 0, so ask the template directly:
        # what is being checked is the template's own contract.
        assert Drifting().variant(item, 0) != item
        assert any(p.check == TEMPLATES_ARE_SOUND for p in validate(broken).problems)

    def test_catches_a_variant_whose_answer_does_not_verify(self, toy) -> None:
        item = toy.items.bank("practice")[0]

        class Desynchronised:
            item_id = item.id

            def variant(self, base, draw):  # noqa: ANN001
                if draw == 0:
                    return base
                return replace(base, prompt=f"take {draw}", answer="not an answer")

        broken = replace(toy, templates={item.id: Desynchronised()})
        assert any(p.check == TEMPLATES_ARE_SOUND for p in validate(broken).problems)

    def test_catches_params_that_drift_from_the_question(self, toy) -> None:
        # The dangerous one, and the reason prompt, answer and params are
        # regenerated together: params left behind from a different question
        # make the buggy rule produce an answer that is correct for *this* one,
        # so the misconception becomes invisible to the diagnostic agent.
        #
        # Note what is *not* checkable here: whether the prompt and the answer
        # describe the same question. Nothing short of solving the prompt could
        # tell, which is why draw 0 pins the pair against the reviewed YAML and
        # the arithmetic is kept in one place per item.
        item = next(i for i in toy.items.bank("practice") if i.probes)

        class StaleParams:
            item_id = item.id

            def variant(self, base, draw):  # noqa: ANN001
                if draw == 0:
                    return base
                return replace(base, prompt=f"take {draw}", params={})

        broken = replace(toy, templates={item.id: StaleParams()})
        assert any(
            p.check == TEMPLATES_ARE_SOUND and "drifted apart" in p.message
            for p in validate(broken).problems
        )

    def test_catches_a_template_that_repeats_a_prompt(self, toy) -> None:
        item = toy.items.bank("practice")[0]

        class Static:
            item_id = item.id

            def variant(self, base, draw):  # noqa: ANN001
                return base

        broken = replace(toy, templates={item.id: Static()})
        assert any(
            p.check == TEMPLATES_ARE_SOUND and "repeats a prompt" in p.message
            for p in validate(broken).problems
        )


class TestCalculusVariants:
    """The generated questions, checked as behaviour rather than as content.

    `domain validate` already checks every draw against the verifier. What is
    left is the property the feature exists for: asking again asks something
    different.
    """

    def test_every_practice_item_has_a_template(self, calculus) -> None:
        # A concept is worked until mastery, and any item without one is asked
        # verbatim until then.
        missing = [
            i.id for i in calculus.items.bank("practice") if i.id not in calculus.templates
        ]
        assert not missing, f"practice items with no variants: {missing}"

    def test_asking_again_asks_something_different(self, calculus) -> None:
        for item in calculus.items.bank("practice"):
            prompts = {calculus.variant(item, draw).prompt for draw in range(5)}
            assert len(prompts) == 5, f"{item.id} repeats within five repetitions"

    def test_the_first_asking_is_the_item_as_written(self, calculus) -> None:
        for item in calculus.items.bank("practice"):
            assert calculus.variant(item, 0) == item

    def test_a_variant_is_the_same_item(self, calculus) -> None:
        # The session counts repetitions by id and the planner selects by id, so
        # a variant that changed either would be a different item wearing the
        # same name.
        for item in calculus.items.bank("practice"):
            variant = calculus.variant(item, 3)
            assert variant.id == item.id
            assert variant.concept_id == item.concept_id
            assert variant.bank == item.bank
            assert variant.probes == item.probes

    def test_the_draw_alone_decides_the_variant(self, calculus) -> None:
        # No generator state: reproducibility must not depend on call ordering.
        item = calculus.items.bank("practice")[0]
        forwards = [calculus.variant(item, d) for d in range(6)]
        backwards = [calculus.variant(item, d) for d in reversed(range(6))]
        assert forwards == list(reversed(backwards))

    def test_test_bank_items_are_not_varied(self, calculus) -> None:
        # The held-out banks are the measuring instrument. Varying them would
        # make a pre-test and a post-test score incomparable between learners.
        for bank in ("pretest", "posttest"):
            for item in calculus.items.bank(bank):
                assert calculus.variant(item, 4) == item

    def test_catches_a_goal_that_is_not_a_concept(self, toy) -> None:
        broken = replace(toy, concepts=_GraphWithGoals(toy.concepts, ("ghost",)))
        problems = validate(broken).problems
        assert any(p.check == GOALS_ARE_REACHABLE for p in problems)

    def test_catches_a_goal_requiring_an_unteachable_concept(self, toy) -> None:
        # The silent case. The graph says the goal is reachable, but a concept
        # on the way has no practice items, so the planner arrives there and has
        # nothing to give. The run ends early and looks like a short session.
        without_distribute = [i for i in toy.items.all() if i.concept_id != "distribute"]
        broken = replace(toy, items=_ItemBankStub(without_distribute))
        problems = validate(broken).problems
        assert any(
            p.check == GOALS_ARE_REACHABLE and "distribute" in p.message for p in problems
        )

    def test_catches_a_graph_with_nothing_to_plan_toward(self, toy) -> None:
        broken = replace(toy, concepts=_GraphWithGoals(toy.concepts, ()))
        assert any(p.check == GOALS_ARE_REACHABLE for p in validate(broken).problems)

    def test_catches_a_test_item_copied_from_practice(self, toy) -> None:
        practice = toy.items.bank("practice")[0]
        leaked = Item(
            "leak", practice.concept_id, practice.prompt, practice.answer, bank="posttest"
        )
        broken = replace(toy, items=_ItemBankStub([*toy.items.all(), leaked]))
        problems = validate(broken).problems
        assert any(p.check == "test_items_held_out" for p in problems)


class _ItemBankStub:
    """Minimal ItemBank over a fixed list, for validator tests."""

    def __init__(self, items) -> None:
        self._items = list(items)

    def all(self):
        return tuple(self._items)

    def get(self, item_id):
        return next(i for i in self._items if i.id == item_id)

    def bank(self, bank):
        return tuple(i for i in self._items if i.bank == bank)

    def for_concept(self, concept_id, bank="practice"):
        return tuple(
            i for i in self._items if i.concept_id == concept_id and i.bank == bank
        )

    def content_hash(self):
        return "stub"


class _GraphWithGoals:
    """A real graph with its goals overridden.

    Constructing the graph with a bad goal raises at load, which is the right
    behaviour and also means the validator's own goal checks could never fire.
    This lets them be exercised.
    """

    def __init__(self, inner, goals) -> None:
        self._inner = inner
        self._goals = tuple(goals)

    def goals(self):
        return self._goals

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TestYamlContentOnDisk:
    """The shipped YAML must stay loadable and complete."""

    def test_every_misconception_cites_a_source(self, toy) -> None:
        for misconception in toy.misconceptions.all():
            assert misconception.source.strip(), misconception.id

    def test_misconceptions_yaml_requires_a_source_field(self, tmp_path: Path) -> None:
        from agent_newton.domains.content import YamlMisconceptionCatalogue

        path = tmp_path / "misconceptions.yaml"
        path.write_text(
            yaml.safe_dump(
                {"misconceptions": [{"id": "m", "concept_id": "c", "description": "d"}]}
            )
        )
        with pytest.raises(DomainError, match="source"):
            YamlMisconceptionCatalogue.from_yaml(path)


class TestACatalogueEntryNeverStatesAnItemsAnswer:
    """⚠️ Found by a cohort, not by reading the content.

    The tutor quotes a misconception's description verbatim, so a number in one
    can happen to be the answer to the item being worked. *"Moves a term across
    the equals sign without negating it: x + 5 = 12 gives x = 12 + 5"* was
    quoted at a learner solving `x + 8 = 20`, whose answer is 12.

    Neither piece of content is wrong on its own, which is why nothing caught
    it: the collision exists only where the two meet. Checked over every
    (description, item) pair on a concept, and over the variants too — a
    template regenerates the numbers, so a draw nobody has seen yet can collide
    where draw 0 does not.
    """

    @pytest.mark.parametrize("domain_name", ["toy_algebra", "calculus"])
    def test_no_description_gives_an_answer_away(self, domain_name: str) -> None:
        from agent_newton.core.evaluation.tutor import leaks_answer

        domain = registry.load_domain(domain_name)
        collisions = [
            (misconception.id, variant.id, draw)
            for misconception in domain.misconceptions.all()
            for item in domain.items.all()
            if item.concept_id == misconception.concept_id
            for draw in range(4)
            for variant in [domain.variant(item, draw)]
            if leaks_answer(misconception.description, variant, domain)
        ]
        assert not collisions, f"a description states an item's answer: {collisions}"

    def test_the_check_would_notice_one(self) -> None:
        # A guard that cannot fail proves nothing. The description that started
        # this, against the item it was quoted at.
        from dataclasses import replace

        from agent_newton.core.evaluation.tutor import leaks_answer

        domain = registry.load_domain("toy_algebra")
        item = domain.items.get("ta_solve_p3")
        assert item.answer == "12"
        offending = (
            "Moves a term across the equals sign without negating it: "
            "x + 5 = 12 gives x = 12 + 5."
        )
        assert leaks_answer(offending, item, domain)
        assert not leaks_answer(
            domain.misconceptions.get("sign_error_moving_term").description, item, domain
        )
        assert replace(item, answer="12").answer == "12"


class TestAGuessableFamilyIsFlagged:
    """A draw can be individually correct and still teach nothing.

    Every template check before this one asked whether a draw was *right*. None
    asked whether it was **earnable**. The antiderivative family generated only
    `(n+1)x^n`, so `3x^2` gave `x^3 + C` and `4x^3` gave `x^4 + C` — read the
    coefficient, write it as the exponent. Correct at every draw, answerable
    without the concept at every draw, and a learner said so at the keyboard
    while the mastery estimate believed them.

    A warning rather than a problem: some concepts genuinely are one step, so a
    constant map is evidence to look rather than proof of a defect.
    """

    def _domain_with(self, toy, template):  # noqa: ANN001
        from dataclasses import replace

        return replace(toy, templates={template.item_id: template})

    def test_a_constant_map_from_prompt_to_answer_is_caught(self, toy) -> None:
        from dataclasses import replace

        item = toy.items.bank("practice")[0]

        class Guessable:
            """The answer is always the prompt's number, one lower."""

            item_id = item.id

            def variant(self, base, draw):  # noqa: ANN001
                if draw == 0:
                    return base
                return replace(base, prompt=f"take {draw + 9}", answer=str(draw + 8))

        warnings = validate(self._domain_with(toy, Guessable())).warnings
        assert any(w.check == GUESSABLE_FAMILY for w in warnings)

    def test_rotating_the_shape_is_not_caught(self, toy) -> None:
        # The fix the check exists to reward: when the *number* of values in the
        # answer changes across draws, no positional map can hold, and the family
        # cannot be answered by one surface rule.
        from dataclasses import replace

        item = toy.items.bank("practice")[0]

        class Rotating:
            item_id = item.id

            def variant(self, base, draw):  # noqa: ANN001
                if draw == 0:
                    return base
                if draw % 2:
                    return replace(base, prompt=f"take {draw}", answer=f"{draw}")
                return replace(base, prompt=f"take {draw}", answer=f"{draw} {draw} {draw}")

        warnings = validate(self._domain_with(toy, Rotating())).warnings
        assert not any(w.check == GUESSABLE_FAMILY for w in warnings)

    def test_the_real_antiderivative_family_is_no_longer_guessable(self) -> None:
        # The family that prompted the check. Its draws now rotate over four
        # shapes; before the rewrite this assertion failed.
        calculus = registry.load_domain("calculus")
        warnings = validate(calculus).warnings
        flagged = {
            w.message.split("'")[1]
            for w in warnings
            if w.check == GUESSABLE_FAMILY
        }
        assert "ca_anti_p1" not in flagged

    def test_no_calculus_family_is_guessable_any_more(self) -> None:
        # ⚠️ This asserted the opposite until 2026-08-17, and the comment said so:
        # four families were flagged, and a non-empty set kept the check honest
        # "if the others are ever rewritten too". They have been — antiderivative,
        # quotient, product and implicit all rotate their shape now, and power and
        # binomial reach the cases where the pattern breaks.
        #
        # So the assertion inverts. The duty it was carrying — a guard that cannot
        # fire proves nothing — is discharged by
        # `test_a_constant_map_from_prompt_to_answer_is_caught`, which fires the
        # check on a synthetic family rather than relying on a real defect
        # surviving.
        calculus = registry.load_domain("calculus")
        flagged = sorted(
            w.message.split("'")[1]
            for w in validate(calculus).warnings
            if w.check == GUESSABLE_FAMILY
        )
        assert flagged == [], f"guessable families are back: {flagged}"


class TestResources:
    """What may be shown beside a question, and the two ways it can be wrong.

    Both problems are content defects that would reach a learner looking
    perfectly reasonable — one hands over an answer, the other arrives as
    mangled notation — so both are caught here rather than at the keyboard.
    """

    def test_calculus_covers_every_concept_it_teaches(self, calculus) -> None:
        taught = {
            concept_id
            for concept_id in calculus.concepts.ids()
            if calculus.items.for_concept(concept_id, "practice")
        }
        assert {r.concept_id for r in calculus.resources.all()} >= taught

    def test_a_domain_may_offer_none(self, toy) -> None:
        # Optional, like templates. toy_algebra has none and must still validate.
        assert toy.resources is None
        assert validate(toy).ok

    def test_the_hash_appears_only_when_there_are_resources(self, toy, calculus) -> None:
        # Absent for a domain with none, so every manifest written before
        # resources existed stays byte-identical and comparable.
        assert "resources_hash" not in toy.content_hashes()
        assert "resources_hash" in calculus.content_hashes()

    def test_the_hash_moves_when_the_content_does(self, calculus) -> None:
        from agent_newton.domains.content import YamlConceptResources

        before = calculus.resources.content_hash()
        changed = YamlConceptResources(
            [replace(r, formula=r.formula + " and one more thing")
             for r in calculus.resources.all()]
        )
        assert changed.content_hash() != before

    def test_reordering_does_not_move_the_hash(self, calculus) -> None:
        from agent_newton.domains.content import YamlConceptResources

        shuffled = YamlConceptResources(list(reversed(calculus.resources.all())))
        assert shuffled.content_hash() == calculus.resources.content_hash()

    def test_no_worked_example_answers_an_item(self, calculus) -> None:
        assert not [p for p in validate(calculus).problems
                    if p.check == RESOURCE_KEEPS_ITS_DISTANCE]

    def test_that_check_can_fail(self, calculus) -> None:
        """A guard that cannot fail proves nothing.

        The shape it looks for: an example whose answer is an item's answer.
        Taken from the item itself rather than written out, so this stays true
        if the bank is re-authored.
        """
        from agent_newton.domains.content import YamlConceptResources

        item = calculus.items.for_concept("power_rule", "practice")[0]
        stray = replace(
            calculus.resources.get("power_rule"), example_answer=item.answer
        )
        broken = replace(
            calculus,
            resources=YamlConceptResources(
                [stray]
                + [r for r in calculus.resources.all() if r.concept_id != "power_rule"]
            ),
        )
        report = validate(broken)
        assert not report.ok
        assert [p for p in report.problems if p.check == RESOURCE_KEEPS_ITS_DISTANCE]

    def test_it_catches_an_answer_written_differently(self, calculus) -> None:
        """Asked of the verifier, not of the string.

        `5x^4` and `5*x**4` are the same disclosure, and a string comparison
        sees two different texts — which is the whole reason `answer_leaked`
        works the way it does.
        """
        from agent_newton.domains.content import YamlConceptResources

        stray = replace(calculus.resources.get("power_rule"), example_answer="5x^4")
        broken = replace(
            calculus,
            resources=YamlConceptResources(
                [stray]
                + [r for r in calculus.resources.all() if r.concept_id != "power_rule"]
            ),
        )
        assert [p for p in validate(broken).problems
                if p.check == RESOURCE_KEEPS_ITS_DISTANCE]

    def test_it_looks_past_the_item_as_written(self, calculus) -> None:
        """A variant is what a learner sees on a repetition.

        This is not hypothetical: an example written for this branch cleared
        every item as written and every one of the first eight draws, and
        answered the ninth. Checking the item alone would have shipped it.
        """
        from agent_newton.domains.content import YamlConceptResources

        base = calculus.items.get("ca_impl_p1")
        ninth = calculus.templates["ca_impl_p1"].variant(base, 9)
        assert ninth.answer != base.answer

        stray = replace(
            calculus.resources.get("implicit_differentiation"),
            example_answer=ninth.answer,
        )
        broken = replace(
            calculus,
            resources=YamlConceptResources(
                [stray]
                + [r for r in calculus.resources.all()
                   if r.concept_id != "implicit_differentiation"]
            ),
        )
        assert [p for p in validate(broken).problems
                if p.check == RESOURCE_KEEPS_ITS_DISTANCE]

    def test_no_resource_carries_a_backslash_command(self, calculus) -> None:
        assert not [p for p in validate(calculus).problems
                    if p.check == RESOURCE_IS_PLAIN_TEXT]

    @pytest.mark.parametrize("field", ["formula", "worked_example"])
    def test_that_check_can_fail_too(self, calculus, field: str) -> None:
        from agent_newton.domains.content import YamlConceptResources

        stray = replace(
            calculus.resources.get("power_rule"),
            **{field: r"the rate is \frac{f(b) - f(a)}{b - a}"},
        )
        broken = replace(
            calculus,
            resources=YamlConceptResources(
                [stray]
                + [r for r in calculus.resources.all() if r.concept_id != "power_rule"]
            ),
        )
        report = validate(broken)
        assert not report.ok
        assert [p for p in report.problems if p.check == RESOURCE_IS_PLAIN_TEXT]

    def test_a_gap_is_a_warning_not_a_failure(self, calculus) -> None:
        """A concept may legitimately need no statement beyond its questions.

        What must not happen is that nobody noticed — the same reasoning as the
        catalogue-entry warning, which exists because three concepts went
        undiagnosable and nothing said so.
        """
        from agent_newton.domains.content import YamlConceptResources

        thinned = replace(
            calculus,
            resources=YamlConceptResources(
                [r for r in calculus.resources.all() if r.concept_id != "power_rule"]
            ),
        )
        report = validate(thinned)
        assert report.ok
        assert [w for w in report.warnings
                if w.check == CONCEPT_HAS_A_RESOURCE and "power_rule" in w.message]

    def test_two_resources_for_one_concept_are_refused(self, calculus) -> None:
        from agent_newton.domains.content import YamlConceptResources

        one = calculus.resources.get("power_rule")
        with pytest.raises(DomainError, match="duplicate resource"):
            YamlConceptResources([one, one])

    def test_the_example_is_labelled_as_not_the_question(self, calculus) -> None:
        """An unlabelled solved problem above a question invites a copy.

        The label is the only thing telling the reader the numbers are not
        theirs, so it is composed in one place and asserted here rather than
        left to whichever front end happens to render it.
        """
        resource = calculus.resources.get("power_rule")
        assert resource.shown(False) == resource.formula
        shown = resource.shown(True)
        assert resource.formula in shown
        assert resource.worked_example in shown
        assert "not your question" in shown


class TestLessons:
    """What a concept *is*, as opposed to what to do with it.

    Two optional fields on ``ConceptResource`` rather than a structure of their
    own, because everything a lesson needs already surrounds it: one entry per
    concept, the plain-text rule, an example validated against every item and
    every template draw, a content hash and a column in the store.

    They exist because every instructional move before them was a reply to a
    failed step. A learner who had never met a concept and one who held a
    misconception about it were answered identically, and a sitting recorded
    what that costs — someone asked what sin(x) was three times, in three
    channels, and was told about the product rule each time.
    """

    @pytest.fixture
    def calculus(self):
        return registry.load_domain("calculus")

    @pytest.fixture
    def toy(self):
        return registry.load_domain("toy_algebra")

    def test_every_calculus_concept_carries_one(self, calculus) -> None:
        missing = [
            c for c in calculus.concepts.ids()
            if (r := calculus.resources.for_concept(c)) is not None and not r.teaches
        ]
        assert not missing, f"concepts with a rule but no lesson: {missing}"

    def test_a_resource_without_one_is_still_valid(self, calculus) -> None:
        # The fields are optional, and a domain that fills in neither behaves
        # exactly as every domain did before they existed.
        bare = replace(
            calculus.resources.get("power_rule"), what_it_means="", why_it_works=""
        )
        assert not bare.teaches
        assert bare.shown(with_example=False) == bare.formula

    def test_a_reason_alone_is_not_a_lesson(self, calculus) -> None:
        # `teaches` keys on `what_it_means` alone. A concept can be explained
        # without a justification being offered, and some genuinely cannot be
        # given one at this level; a justification with nothing to justify is
        # the combination that makes no sense.
        assert not replace(
            calculus.resources.get("power_rule"),
            what_it_means="",
            why_it_works="because it does",
        ).teaches

    # -- the hash ---------------------------------------------------------

    def test_adding_the_fields_did_not_move_the_hash_for_content_without_them(
        self, calculus
    ) -> None:
        """⚠️ The guard that kept every existing learner out of a false warning.

        ``content_drift`` refuses to resume a learner across a change in the
        subject matter by comparing this hash. Folding two empty strings into
        every entry would have told every learner already in the store that
        their content had changed — over an addition none of them could have
        been shown.

        So the lesson fields enter the hash only where a lesson exists. An entry
        without one hashes exactly as it did before they were added, and the
        expected value here is the one recorded in sittings already stored.
        """
        from agent_newton.domains.content import YamlConceptResources

        bare = YamlConceptResources(
            [
                replace(r, what_it_means="", why_it_works="")
                for r in calculus.resources.all()
            ]
        )
        assert bare.content_hash() == "6a76d53d16344d71"

    def test_writing_a_lesson_does_move_the_hash(self, calculus) -> None:
        # The other half, and it is what makes the hash mean anything: a learner
        # now reads something they did not read before, and resuming across that
        # is a change worth reporting.
        from agent_newton.domains.content import YamlConceptResources

        bare = YamlConceptResources(
            [
                replace(r, what_it_means="", why_it_works="")
                for r in calculus.resources.all()
            ]
        )
        assert calculus.resources.content_hash() != bare.content_hash()

    # -- what a learner reads ---------------------------------------------

    def test_the_lesson_is_composed_in_one_place(self, calculus) -> None:
        # The session records what was taught and the front end displays it. A
        # learner reading one thing while the audit log records another would be
        # unnoticeable, and would make the record worth nothing.
        resource = calculus.resources.get("antiderivative")
        text = resource.lesson()
        assert resource.what_it_means.strip() in text
        assert resource.why_it_works.strip() in text
        assert resource.worked_example in text

    def test_the_example_is_labelled_as_not_being_the_question(self, calculus) -> None:
        # An unlabelled solved problem is an invitation to copy its answer into
        # the box. The label is load-bearing here for the same reason it is in
        # `shown`.
        text = calculus.resources.get("power_rule").lesson()
        assert "not your question" in text

    def test_a_lesson_never_states_its_own_example_answer(self, calculus) -> None:
        # The example is worked, and stopping short of the result is the same
        # rule the worked-step level follows. The answer is checked, never shown.
        for resource in calculus.resources.all():
            assert resource.example_answer not in resource.what_it_means
            assert resource.example_answer not in resource.why_it_works

    # -- validation -------------------------------------------------------

    def test_the_shipped_lessons_are_plain_text(self, calculus) -> None:
        assert not [
            p for p in validate(calculus).problems if p.check == RESOURCE_IS_PLAIN_TEXT
        ]

    def test_the_plain_text_check_covers_the_lesson_fields(self, calculus) -> None:
        """A guard that cannot fail proves nothing.

        The lesson fields are the longest prose a learner reads anywhere here,
        so they are the likeliest place for a backslash to arrive — and it is
        invisible to whoever wrote it, appearing only once the text has been
        through JSON. A learner read one as ``rac{f(b) - f(a)}{b - a}`` and
        could not tell it meant a division.
        """
        from agent_newton.domains.content import YamlConceptResources

        for field in ("what_it_means", "why_it_works"):
            broken = replace(
                calculus,
                resources=YamlConceptResources(
                    [
                        replace(
                            calculus.resources.get("power_rule"),
                            **{field: "the rule is \\frac{n}{x}"},
                        )
                    ]
                ),
            )
            assert [
                p
                for p in validate(broken).problems
                if p.check == RESOURCE_IS_PLAIN_TEXT and field in p.message
            ], f"a backslash in {field} was not caught"

    def test_a_concept_with_no_lesson_is_warned_about(self, calculus) -> None:
        """And the check can fail, which is the point of this test.

        A warning rather than a problem: a concept may genuinely not want one.
        What must not happen is that nobody noticed.
        """
        from agent_newton.domains.content import YamlConceptResources

        thinned = replace(
            calculus,
            resources=YamlConceptResources(
                [
                    replace(r, what_it_means="", why_it_works="")
                    if r.concept_id == "power_rule"
                    else r
                    for r in calculus.resources.all()
                ]
            ),
        )
        report = validate(thinned)
        assert report.ok, "a missing lesson must not fail validation"
        assert [
            w
            for w in report.warnings
            if w.check == CONCEPT_HAS_A_LESSON and "power_rule" in w.message
        ]

    def test_nothing_is_warned_about_for_a_domain_with_no_resources(self, toy) -> None:
        # A domain offering nothing is not a domain missing something.
        assert toy.resources is None
        assert not [
            w for w in validate(toy).warnings if w.check == CONCEPT_HAS_A_LESSON
        ]
