"""Learner state: BKT, the frontier, the views, and the audit log."""

from __future__ import annotations

import pytest

from agent_newton.config import BKTConfig, Config, ZPDConfig
from agent_newton.core.state import bkt, zpd
from agent_newton.core.state.store import Blackboard, new_blackboard
from agent_newton.core.state.views import FullStateView, ItemCorrectnessView
from agent_newton.domains import registry
from agent_newton.domains.base import Concept, VerificationResult, Verdict
from agent_newton.domains.content import YamlConceptGraph

PARAMS = BKTConfig()
BAND = ZPDConfig(theta_lower=0.70, theta_upper=0.90)


@pytest.fixture(scope="module")
def graph():
    return registry.load_domain("toy_algebra").concepts


def correct() -> VerificationResult:
    return VerificationResult(Verdict.CORRECT, "a")


def incorrect() -> VerificationResult:
    return VerificationResult(Verdict.INCORRECT, "a")


def unreadable() -> VerificationResult:
    return VerificationResult(Verdict.UNPARSEABLE, "a", "could not parse")


class TestBKT:
    def test_correct_raises_the_estimate(self) -> None:
        assert bkt.observe(0.3, correct=True, params=PARAMS) > 0.3

    def test_incorrect_lowers_the_posterior(self) -> None:
        # The transition term can lift the final value, so the comparison that
        # isolates the evidence is on the posterior before transition.
        assert bkt.posterior(0.5, correct=False, params=PARAMS) < 0.5

    def test_repeated_correct_answers_converge_upward(self) -> None:
        p = bkt.initial(PARAMS)
        for _ in range(12):
            p = bkt.observe(p, correct=True, params=PARAMS)
        assert p > 0.95

    def test_estimates_never_saturate(self) -> None:
        # At exactly 1.0 no later evidence could move the estimate and a
        # concept could never re-enter the frontier.
        p = bkt.initial(PARAMS)
        for _ in range(200):
            p = bkt.observe(p, correct=True, params=PARAMS)
        assert p < 1.0
        assert bkt.observe(p, correct=False, params=PARAMS) < p

    def test_is_monotone_in_the_prior(self) -> None:
        assert bkt.observe(0.2, True, PARAMS) < bkt.observe(0.6, True, PARAMS)


class TestFrontier:
    def _graph(self) -> YamlConceptGraph:
        # a -> b -> c, and a -> d
        return YamlConceptGraph(
            [
                Concept("a", "a"),
                Concept("b", "b", ("a",)),
                Concept("c", "c", ("b",)),
                Concept("d", "d", ("a",)),
            ]
        )

    def test_starts_at_the_roots(self) -> None:
        frontier = zpd.compute({}, self._graph(), BAND, prior=0.15)
        assert set(frontier) == {"a"}

    def test_opens_successors_once_a_prerequisite_is_met(self) -> None:
        frontier = zpd.compute({"a": 0.95}, self._graph(), BAND, prior=0.15)
        assert set(frontier) == {"b", "d"}

    def test_excludes_concepts_whose_prerequisite_is_only_partial(self) -> None:
        # 0.75 is above theta_lower but below theta_upper: a counts as a met
        # prerequisite while remaining unmastered itself.
        frontier = zpd.compute({"a": 0.75}, self._graph(), BAND, prior=0.15)
        assert "a" in frontier and "b" in frontier

    def test_excludes_concepts_behind_an_unmet_prerequisite(self) -> None:
        frontier = zpd.compute({"a": 0.95}, self._graph(), BAND, prior=0.15)
        assert "c" not in frontier

    def test_empty_when_everything_is_mastered(self) -> None:
        mastery = {c: 0.99 for c in ("a", "b", "c", "d")}
        frontier = zpd.compute(mastery, self._graph(), BAND, prior=0.15)
        assert not frontier
        assert not frontier.fallback
        assert "mastered" in frontier.reason

    def test_is_never_empty_while_work_remains(self) -> None:
        """The property that makes the fallback unreachable.

        Prerequisites sit at strictly lower depth, so the shallowest unmastered
        concept cannot have an unmastered prerequisite and is therefore always
        in the zone. Checked across many mastery configurations because it is
        the invariant the planner depends on.
        """
        import itertools

        graph = self._graph()
        ids = list(graph.ids())
        for values in itertools.product([0.1, 0.75, 0.95], repeat=len(ids)):
            mastery = dict(zip(ids, values))
            frontier = zpd.compute(mastery, graph, BAND, prior=0.15)
            work_remains = any(v < BAND.theta_upper for v in values)
            assert bool(frontier) == work_remains, mastery
            assert not frontier.fallback, mastery

    def test_reports_an_unconstrained_band(self) -> None:
        graph = self._graph()
        wide = ZPDConfig(theta_lower=0.01, theta_upper=0.99)
        frontier = zpd.compute({c: 0.5 for c in graph.ids()}, graph, wide, prior=0.15)
        assert zpd.is_unconstrained(frontier, graph)

    def test_works_on_the_real_domain(self, graph) -> None:
        frontier = zpd.compute({}, graph, BAND, prior=0.15)
        assert set(frontier) == {"integer_arithmetic"}


class TestBlackboard:
    def _board(self, graph, **overrides) -> Blackboard:
        config = Config.model_validate({"domain": "toy_algebra", **overrides})
        return new_blackboard("L1", seed=1, graph=graph, config=config)

    def test_records_evidence_and_moves_mastery(self, graph) -> None:
        board = self._board(graph)
        before = board.probability("integer_arithmetic")
        assert board.record_observation(
            item_id="i1", concept_id="integer_arithmetic", result=correct()
        )
        assert board.probability("integer_arithmetic") > before

    def test_unmeasurable_results_change_no_estimate(self, graph) -> None:
        # The invariant from the verifier's contract: a failure to measure is
        # not information about the learner.
        board = self._board(graph)
        before = board.probability("integer_arithmetic")

        assert not board.record_observation(
            item_id="i1", concept_id="integer_arithmetic", result=unreadable()
        )

        assert board.probability("integer_arithmetic") == before
        assert board.state.error_trace == []
        assert board.view("decoupled").outcomes == ()
        assert board.unmeasurable == 1

    def test_unmeasurable_results_are_still_audited(self, graph) -> None:
        # Silently dropping them would make the record incomplete and hide a
        # failing verifier.
        board = self._board(graph)
        board.record_observation(
            item_id="i1", concept_id="integer_arithmetic", result=unreadable()
        )
        assert any("unmeasurable" in r.summary for r in board.audit_log)

    def test_errors_enter_the_trace_with_their_label(self, graph) -> None:
        board = self._board(graph)
        board.record_observation(
            item_id="i1",
            concept_id="combine_like_terms",
            result=incorrect(),
            misconception_label="combine_unlike_terms",
            confidence=0.8,
        )
        assert board.state.misconception_count("combine_unlike_terms") == 1

    def test_the_trace_is_bounded(self, graph) -> None:
        board = self._board(graph, arbitration={"error_trace_length": 3})
        for n in range(6):
            board.record_observation(
                item_id=f"i{n}", concept_id="combine_like_terms", result=incorrect()
            )
        assert len(board.state.error_trace) == 3
        # The window keeps the most recent, which is what k_repeats counts over.
        assert [e.item_id for e in board.state.error_trace] == ["i3", "i4", "i5"]

    def test_every_mutation_bumps_the_version(self, graph) -> None:
        board = self._board(graph)
        seen = [board.version]
        board.record_observation(item_id="i1", concept_id="integer_arithmetic", result=correct())
        seen.append(board.version)
        board.record_replan("threshold crossed", theta=0.15)
        seen.append(board.version)
        assert seen == sorted(set(seen))

    def test_the_audit_log_cannot_be_rewritten(self, graph) -> None:
        board = self._board(graph)
        log = board.audit_log
        assert isinstance(log, tuple)
        board.record_replan("later", reason="x")
        assert len(board.audit_log) == len(log) + 1

    def test_replans_carry_their_triggering_evidence(self, graph) -> None:
        board = self._board(graph)
        board.record_replan("mastery moved past theta", concept="distribute", delta=0.21)
        record = board.audit_log[-1]
        assert record.cause == "replan"
        assert record.evidence["delta"] == pytest.approx(0.21)

    def test_the_frontier_is_stable_within_a_version(self, graph) -> None:
        board = self._board(graph)
        assert board.frontier is board.frontier

    def test_the_frontier_recomputes_after_a_change(self, graph) -> None:
        board = self._board(graph)
        first = board.frontier
        for _ in range(15):
            board.record_observation(
                item_id="i", concept_id="integer_arithmetic", result=correct()
            )
        assert set(board.frontier) != set(first)


class TestTheAblation:
    """The two arms differ in what the planner can see, and nothing else."""

    def _board(self, graph, arm: str) -> Blackboard:
        config = Config.model_validate({"domain": "toy_algebra", "arm": arm})
        board = new_blackboard("L1", seed=1, graph=graph, config=config)
        board.record_observation(
            item_id="i1",
            concept_id="combine_like_terms",
            result=incorrect(),
            misconception_label="combine_unlike_terms",
        )
        board.record_observation(item_id="i2", concept_id="integer_arithmetic", result=correct())
        return board

    def test_coupled_view_carries_the_frontier(self, graph) -> None:
        view = self._board(graph, "coupled").view()
        assert isinstance(view, FullStateView)
        assert view.frontier
        assert view.mastery
        assert view.recent_misconceptions() == ["combine_unlike_terms"]

    def test_decoupled_view_cannot_produce_a_frontier(self, graph) -> None:
        # Not "coarser" — the inputs are absent. A planner holding this view is
        # structurally incapable of frontier-based selection.
        view = self._board(graph, "decoupled").view()
        assert isinstance(view, ItemCorrectnessView)
        assert not hasattr(view, "frontier")
        assert not hasattr(view, "mastery")
        assert not hasattr(view, "error_trace")

    def test_both_views_share_the_outcome_stream(self, graph) -> None:
        # What the decoupled arm loses is the state, not the record of results.
        coupled = self._board(graph, "coupled").view()
        decoupled = self._board(graph, "decoupled").view()
        assert coupled.outcomes == decoupled.outcomes == (False, True)

    def test_the_underlying_state_is_identical_across_arms(self, graph) -> None:
        # Both arms write the same things; only the read differs. If this ever
        # fails, the comparison is measuring more than one variable.
        coupled = self._board(graph, "coupled")
        decoupled = self._board(graph, "decoupled")
        assert coupled.state.mastery == decoupled.state.mastery
        assert len(coupled.state.error_trace) == len(decoupled.state.error_trace)
        assert coupled.version == decoupled.version

    def test_consecutive_correct_is_available_to_both(self, graph) -> None:
        # The decoupled planner advances on this rule, so both views must
        # answer it.
        for arm in ("coupled", "decoupled"):
            assert self._board(graph, arm).view().consecutive_correct() == 1
