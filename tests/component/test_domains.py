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
from agent_newton.domains.validate import ANSWERS_VERIFY, RULES_PRODUCE_ERRORS, validate


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


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
