"""Forgetting, as it applies to what the system believes.

The mechanism is small, so the tests are about its *properties* rather than
sample points: a mechanism added to make an architecture look good is the failure
mode §7c warns about, and the guard against it is that the identity case, the
limit and the monotonicity are all pinned.
"""

from __future__ import annotations

import math

import pytest

from agent_newton.config import BKTConfig, Config, DecayConfig, ZPDConfig
from agent_newton.core.state import bkt, decay, zpd
from agent_newton.core.state.store import new_blackboard
from agent_newton.domains import registry
from agent_newton.domains.base import VerificationResult, Verdict

PRIOR = 0.15
BAND = ZPDConfig(theta_lower=0.70, theta_upper=0.90)


@pytest.fixture(scope="module")
def graph():
    return registry.load_domain("toy_algebra").concepts


def board_with(graph, half_life: float | None, **mastery: float):
    config = Config.model_validate(
        {"domain": "toy_algebra", "decay": {"half_life_days": half_life}}
    )
    board = new_blackboard("L1", seed=1, graph=graph, config=config)
    board.state.mastery.update(mastery)
    return board


class TestRelax:
    def test_no_elapsed_time_is_the_identity(self) -> None:
        # A sequence with no gaps must reproduce sessions run back to back.
        # Without this the mechanism would manufacture an effect from nothing.
        assert decay.relax(0.93, PRIOR, 0.0, 30.0) == 0.93

    def test_one_half_life_closes_half_the_gap(self) -> None:
        assert decay.relax(0.95, PRIOR, 30.0, 30.0) == pytest.approx(
            PRIOR + (0.95 - PRIOR) / 2
        )

    def test_the_prior_is_the_limit(self) -> None:
        assert decay.relax(0.95, PRIOR, 10_000.0, 30.0) == pytest.approx(PRIOR, abs=1e-9)

    def test_it_never_crosses_the_prior(self) -> None:
        for elapsed in (1, 5, 30, 90, 365, 10_000):
            assert decay.relax(0.95, PRIOR, elapsed, 30.0) >= PRIOR

    def test_an_estimate_below_the_prior_relaxes_upward(self) -> None:
        # Time erodes bad news exactly as it erodes good news: both are evidence
        # going stale. A concept measured worse than the prior weeks ago is not
        # still known to be that bad.
        assert decay.relax(0.02, PRIOR, 30.0, 30.0) > 0.02
        assert decay.relax(0.02, PRIOR, 30.0, 30.0) <= PRIOR

    def test_it_is_monotone_in_elapsed_time(self) -> None:
        previous = 0.95
        for elapsed in (1, 2, 4, 8, 16, 32, 64):
            current = decay.relax(0.95, PRIOR, elapsed, 30.0)
            assert current < previous
            previous = current

    def test_a_non_positive_half_life_is_refused(self) -> None:
        # It would divide by zero or run the exponent backwards, and a silently
        # inverted decay would *sharpen* stale beliefs.
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="must be positive"):
                decay.relax(0.9, PRIOR, 1.0, bad)


class TestDecayOnTheBlackboard:
    def test_it_is_off_by_default(self, graph) -> None:
        # Every single-session run must behave exactly as it did before this
        # existed, or the first study's numbers stop being reproducible.
        assert DecayConfig().half_life_days is None
        assert not DecayConfig().enabled

        board = board_with(graph, None, distribute=0.95)
        assert board.apply_decay(365.0) == 0
        assert board.probability("distribute") == 0.95

    def test_it_lowers_an_observed_estimate(self, graph) -> None:
        board = board_with(graph, 30.0, distribute=0.95)
        assert board.apply_decay(30.0) == 1
        assert board.probability("distribute") < 0.95

    def test_an_unobserved_concept_is_left_alone(self, graph) -> None:
        # It already sits at the prior. Writing one would make an untouched
        # concept indistinguishable from one that was worked and forgotten.
        board = board_with(graph, 30.0, distribute=0.95)
        board.apply_decay(30.0)
        assert set(board.state.mastery) == {"distribute"}

    def test_zero_elapsed_days_changes_nothing(self, graph) -> None:
        board = board_with(graph, 30.0, distribute=0.95)
        assert board.apply_decay(0.0) == 0
        assert board.probability("distribute") == 0.95

    def test_a_mastered_concept_re_enters_the_frontier(self, graph) -> None:
        # The point of the whole mechanism: spaced review is a consequence of
        # the frontier reopening, not a scheduler deciding to revisit.
        board = board_with(graph, 30.0, integer_arithmetic=0.95)
        assert "integer_arithmetic" not in board.frontier

        board.apply_decay(120.0)
        assert board.probability("integer_arithmetic") < BAND.theta_upper
        assert "integer_arithmetic" in zpd.compute(
            dict(board.state.mastery), graph, BAND, PRIOR
        )

    def test_it_is_audited_under_its_own_cause(self, graph) -> None:
        # Distinguishable from evidence: it moves an estimate without the
        # learner having done anything at all.
        board = board_with(graph, 30.0, distribute=0.95)
        board.apply_decay(30.0)
        decayed = [r for r in board.audit_log if r.cause == "decay"]
        assert len(decayed) == 1
        assert decayed[0].evidence["elapsed_days"] == 30.0
        assert decayed[0].evidence["mastery_after"] < decayed[0].evidence["mastery_before"]

    def test_it_is_not_recorded_as_an_observation(self, graph) -> None:
        board = board_with(graph, 30.0, distribute=0.95)
        board.apply_decay(30.0)
        assert not [r for r in board.audit_log if r.cause == "observation"]

    def test_the_error_trace_and_outcomes_survive_it(self, graph) -> None:
        # What the learner did is a fact about the past. Only the inference
        # from it goes stale.
        board = board_with(graph, 30.0)
        board.record_observation(
            item_id="i1",
            concept_id="distribute",
            result=VerificationResult(Verdict.INCORRECT, "a"),
            misconception_label="distribute_first_term_only",
        )
        trace_before = list(board.state.error_trace)
        outcomes_before = board.view("decoupled").outcomes

        board.apply_decay(60.0)

        assert board.state.error_trace == trace_before
        assert board.view("decoupled").outcomes == outcomes_before


class TestDecayIsRecordedForComparability:
    def test_the_manifest_carries_it(self) -> None:
        from agent_newton.manifest import RunManifest

        config = Config.model_validate({"decay": {"half_life_days": 21.0}})
        assert RunManifest.create(config, "r1").decay_half_life_days == 21.0

    def test_runs_under_different_decay_cannot_be_pooled(self) -> None:
        # Two different learner models. Averaging them would report posteriors
        # that mean different things as though they were one scale.
        from agent_newton.manifest import IncomparableRunsError, RunManifest, assert_poolable

        stale = RunManifest.create(
            Config.model_validate({"decay": {"half_life_days": 21.0}}), "r1"
        )
        fresh = RunManifest.create(Config.model_validate({}), "r2")
        with pytest.raises(IncomparableRunsError, match="belief half-life"):
            assert_poolable([stale, fresh])

    def test_runs_under_the_same_decay_pool_fine(self) -> None:
        from agent_newton.manifest import RunManifest, assert_poolable

        config = Config.model_validate({"decay": {"half_life_days": 21.0}})
        assert_poolable([RunManifest.create(config, "r1"), RunManifest.create(config, "r2")])


class TestDecayIsNotTheLearnerForgetting:
    def test_it_touches_no_profile(self, graph) -> None:
        # Belief and ground truth are separate mechanisms with separate rates.
        # If decay moved the profile, the system's estimate would be right about
        # forgetting by construction and the comparison would be circular.
        from agent_newton.config import SimulatorConfig
        from agent_newton.core.simulator import sample_profile

        toy = registry.load_domain("toy_algebra")
        profile = sample_profile("L1", 1, toy.misconceptions, SimulatorConfig())
        before = profile.snapshot()

        board = board_with(graph, 30.0, distribute=0.95)
        board.apply_decay(90.0)

        assert profile.snapshot() == before

    def test_bkt_has_no_notion_of_elapsed_time(self) -> None:
        # Decay is deliberately not folded into `observe`: one is evidence
        # arriving, the other is evidence ageing, and a single function doing
        # both could not be swept independently.
        assert "elapsed" not in bkt.observe.__code__.co_varnames
        assert math.isclose(
            bkt.observe(0.5, True, BKTConfig()), bkt.observe(0.5, True, BKTConfig())
        )


class TestTheGapIsSpentOnce:
    """⚠️ Ageing is exponential in the elapsed time, so it must not run twice.

    The demo applies decay before asking the learner what they want to practise,
    because the estimates shown beside each concept have to be the ones the
    sitting will use rather than a picture of where they were before the gap. It
    then hands the session a zero gap. This is why.
    """

    def _board(self, graph):
        config = Config.model_validate(
            {"domain": "toy_algebra", "decay": {"half_life_days": 14.0}}
        )
        board = new_blackboard("L1", 1, graph, config)
        board.state.mastery["distribute"] = 0.96
        return board

    def test_applying_it_twice_ages_the_model_twice(self, graph) -> None:
        once = self._board(graph)
        once.apply_decay(14.0)

        twice = self._board(graph)
        twice.apply_decay(14.0)
        twice.apply_decay(14.0)

        assert twice.probability("distribute") < once.probability("distribute")

    def test_a_spent_gap_moves_nothing(self, graph) -> None:
        # What the demo relies on: after the gap is applied and zeroed, the
        # session's own call is a no-op rather than a second month.
        board = self._board(graph)
        board.apply_decay(14.0)
        after = board.probability("distribute")
        assert board.apply_decay(0.0) == 0
        assert board.probability("distribute") == after
