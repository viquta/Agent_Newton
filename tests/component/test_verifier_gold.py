"""The verifier against its hand-labelled gold set.

``tests/fixtures/gold/calculus_verifier_cases.yaml`` holds ``(item, response)``
pairs labelled by hand from the mathematics, not from what the verifier
returned. It runs here as a gate rather than only from the CLI, because the
verifier decides correctness for every student step and a regression in it is
silent: no exception, no crash, just a learner charged with an error they did
not make.

The set is deliberately weighted toward correct answers written differently.
Those are the cases the verifier exists to get right, and the ones a naive
implementation fails.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from agent_newton.core.evaluation.verifier import EXPECTED, load, score
from agent_newton.domains import registry
from agent_newton.domains.base import DomainError, Item, VerificationResult, Verdict

GOLD = Path(__file__).resolve().parents[1] / "fixtures" / "gold" / "calculus_verifier_cases.yaml"


@pytest.fixture(scope="module")
def calculus():
    return registry.load_domain("calculus")


@pytest.fixture(scope="module")
def cases(calculus):
    return load(GOLD, calculus)


@pytest.fixture(scope="module")
def report(calculus, cases):
    return score(calculus, cases)


class TestTheSetItself:
    """A gold set that probes nothing measures nothing."""

    def test_every_case_class_is_represented(self, report) -> None:
        for kind in EXPECTED:
            assert report.of_kind(kind), f"no {kind!r} cases"

    def test_equivalent_forms_dominate(self, report) -> None:
        # The point of the set. If these ever stopped being the largest class,
        # it would have drifted into testing something else.
        equivalent = len(report.of_kind("equivalent"))
        assert equivalent >= 30
        assert equivalent > len(report.of_kind("canonical"))

    def test_cases_span_the_bank(self, cases) -> None:
        # Concentrated on a few items, the set would measure those items rather
        # than the verifier.
        assert len({c.item_id for c in cases}) >= 12

    def test_a_stated_limitation_carries_its_reason(self, cases) -> None:
        for case in cases:
            if case.known_limitation:
                assert len(case.known_limitation) > 40, case.id


class TestAgreementWithTheHandLabels:
    def test_no_unpredicted_disagreement(self, report) -> None:
        assert not report.surprises(), "\n".join(
            f"  {s.case.id}: {s.case.response!r} on {s.case.item_id} — labelled "
            f"{s.case.kind} ({s.case.expected.value}), verified {s.actual.value}"
            f"{' — ' + s.detail if s.detail else ''}"
            for s in report.surprises()
        )

    def test_no_stated_limitation_has_quietly_been_fixed(self, report) -> None:
        # The other direction, and the one that rots unnoticed: a case marked as
        # a known limitation that now agrees is a marker describing the past.
        assert not report.resolved(), (
            "these cases now agree with their label; drop the known_limitation:\n"
            + "\n".join(f"  {s.case.id}" for s in report.resolved())
        )

    def test_no_correct_answer_is_scored_as_an_error(self, report) -> None:
        # The dangerous half of a false negative. UNPARSEABLE wastes a step;
        # INCORRECT writes an error event about a learner who made none, and the
        # tutor then aims a hint at a misconception nobody holds.
        assert not report.scored_incorrect(), "\n".join(
            f"  {s.case.id}: {s.case.response!r} is correct, scored incorrect"
            for s in report.scored_incorrect()
        )

    def test_the_measured_rates_are_reported(self, report) -> None:
        # Not a threshold — a floor low enough that a broken verifier trips it,
        # so the suite fails rather than silently reporting a poor rate.
        assert report.accuracy >= 0.90
        assert report.false_negative_rate <= 0.10
        assert report.false_accept_rate <= 0.10


class TestTheGoldSetCatchesABrokenVerifier:
    """A check that cannot fail proves nothing."""

    def _against(self, calculus, cases, verifier):
        return score(dataclasses.replace(calculus, verifier=verifier), cases)

    def test_a_verifier_that_rejects_everything_is_caught(self, calculus, cases) -> None:
        class RejectsEverything:
            def verify(self, item: Item, response: str) -> VerificationResult:
                return VerificationResult(Verdict.INCORRECT, item.answer)

        broken = self._against(calculus, cases, RejectsEverything())
        assert broken.surprises()
        assert broken.scored_incorrect()
        assert broken.false_negative_rate > 0.9

    def test_a_verifier_that_accepts_everything_is_caught(self, calculus, cases) -> None:
        class AcceptsEverything:
            def verify(self, item: Item, response: str) -> VerificationResult:
                return VerificationResult(Verdict.CORRECT, item.answer)

        broken = self._against(calculus, cases, AcceptsEverything())
        assert broken.surprises()
        assert broken.false_accept_rate > 0.9

    def test_a_string_matching_verifier_is_caught(self, calculus, cases) -> None:
        # The implementation the set exists to rule out: right answer, wrong
        # spelling, marked wrong.
        class StringMatch:
            def verify(self, item: Item, response: str) -> VerificationResult:
                same = response.strip() == item.answer.strip()
                verdict = Verdict.CORRECT if same else Verdict.INCORRECT
                return VerificationResult(verdict, item.answer)

        broken = self._against(calculus, cases, StringMatch())
        assert broken.false_negative_rate > 0.9
        assert broken.scored_incorrect()


class TestLoading:
    """The file is checked against the domain it labels."""

    def _write(self, tmp_path, body: str):
        path = tmp_path / "gold.yaml"
        path.write_text(body)
        return path

    def test_an_unknown_item_id_fails_loudly(self, calculus, tmp_path) -> None:
        path = self._write(
            tmp_path,
            "cases:\n"
            "  - id: a\n    item_id: ca_no_such_item\n    response: '4'\n    kind: canonical\n",
        )
        with pytest.raises(DomainError):
            load(path, calculus)

    def test_an_unknown_kind_fails_loudly(self, calculus, tmp_path) -> None:
        path = self._write(
            tmp_path,
            "cases:\n  - id: a\n    item_id: ca_lim_p1\n    response: '4'\n    kind: maybe\n",
        )
        with pytest.raises(DomainError, match="unknown kind"):
            load(path, calculus)

    def test_duplicate_ids_fail_loudly(self, calculus, tmp_path) -> None:
        path = self._write(
            tmp_path,
            "cases:\n"
            "  - id: a\n    item_id: ca_lim_p1\n    response: '4'\n    kind: canonical\n"
            "  - id: a\n    item_id: ca_lim_p1\n    response: '0'\n    kind: wrong\n",
        )
        with pytest.raises(DomainError, match="duplicate"):
            load(path, calculus)

    def test_an_empty_file_fails_loudly(self, calculus, tmp_path) -> None:
        # Silently scoring zero cases would report perfect accuracy.
        with pytest.raises(DomainError, match="no cases"):
            load(self._write(tmp_path, "cases: []\n"), calculus)


class TestCanonicalAnswersAgreeWithTheValidator:
    def test_every_canonical_case_matches_its_item(self, calculus, report) -> None:
        # A canonical case must quote the bank, not paraphrase it; otherwise it
        # is an equivalent-form case wearing the wrong label.
        for scored in report.of_kind("canonical"):
            item = calculus.items.get(scored.case.item_id)
            assert scored.case.response == item.answer, scored.case.id
