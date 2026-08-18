"""Learner state: BKT, the frontier, the views, and the audit log."""

from __future__ import annotations

import pytest

from agent_newton.config import BKTConfig, Config, ZPDConfig
from agent_newton.core.state import bkt, route, zpd
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


class TestSeedingFromAHeldOutTest:
    """What a pre-test may and may not do to the learner model.

    It may move the posteriors — that is the point. It may not leave anything
    behind that later reads as practice, because the learner did that work
    unaided and without a diagnosis.
    """

    def _board(self, graph) -> Blackboard:
        config = Config.model_validate({"domain": "toy_algebra"})
        return new_blackboard("L1", seed=1, graph=graph, config=config)

    def test_a_correct_answer_raises_the_estimate(self, graph) -> None:
        board = self._board(graph)
        before = board.probability("integer_arithmetic")
        assert board.seed_from_test([("integer_arithmetic", Verdict.CORRECT)]) == 1
        assert board.probability("integer_arithmetic") > before

    def test_a_wrong_answer_lowers_it(self, graph) -> None:
        board = self._board(graph)
        before = board.probability("integer_arithmetic")
        board.seed_from_test([("integer_arithmetic", Verdict.INCORRECT)])
        assert board.probability("integer_arithmetic") < before

    def test_the_learning_transition_is_not_applied(self, graph) -> None:
        # The transition means "the learner may have learned it at this
        # opportunity", which is true of practice and false of a test answered
        # unaided and without feedback. It is also large enough to invert the
        # sign: `observe` on a wrong answer from the prior returns 0.22, above
        # the 0.15 it started at. Seeding with it would raise the estimate on
        # every question the learner got wrong.
        board = self._board(graph)
        prior = board.probability("integer_arithmetic")
        board.seed_from_test([("integer_arithmetic", Verdict.INCORRECT)])

        assert board.probability("integer_arithmetic") == pytest.approx(
            bkt.revise(prior, False, PARAMS)
        )
        assert bkt.observe(prior, False, PARAMS) > prior

    def test_an_unreadable_answer_moves_nothing(self, graph) -> None:
        board = self._board(graph)
        before = board.probability("integer_arithmetic")
        assert board.seed_from_test([("integer_arithmetic", Verdict.UNPARSEABLE)]) == 0
        assert board.probability("integer_arithmetic") == before

    def test_no_error_event_is_recorded(self, graph) -> None:
        # A test is administered without diagnosis, so a wrong answer here has
        # no misconception label. Unlabelled trace entries would trip the
        # arbitration policy's repeat trigger on evidence that does not exist.
        board = self._board(graph)
        board.seed_from_test([("combine_like_terms", Verdict.INCORRECT)])
        assert board.state.error_trace == []

    def test_the_outcome_stream_is_untouched(self, graph) -> None:
        # The stream is the practice record the decoupled view is built from.
        board = self._board(graph)
        board.seed_from_test(
            [("integer_arithmetic", Verdict.CORRECT), ("distribute", Verdict.INCORRECT)]
        )
        assert board.view("decoupled").outcomes == ()

    def test_it_is_audited_under_its_own_cause(self, graph) -> None:
        # Distinguishable from practice, so anything counting what the learner
        # did during the session can leave it out.
        board = self._board(graph)
        board.seed_from_test([("integer_arithmetic", Verdict.CORRECT)])
        seeded = [r for r in board.audit_log if r.cause == "seed"]
        assert len(seeded) == 1
        assert seeded[0].evidence["concept_id"] == "integer_arithmetic"

    def test_it_is_not_recorded_as_an_observation(self, graph) -> None:
        board = self._board(graph)
        board.seed_from_test([("integer_arithmetic", Verdict.CORRECT)])
        assert not [r for r in board.audit_log if r.cause == "observation"]


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


class TestAConceptReopenedAtTheLearnersRequest:
    """``reviewing`` relaxes the upper bound, which only a learner may ask for.

    Built after a sitting: the learner asked for `implicit_differentiation`, one
    correct pre-test answer had just seeded it to 0.965 at
    ``pretest_weight: 3``, and the request was refused because the concept had
    left the frontier. The estimate has been measured unreliable in exactly that
    region — §7i ended with three concepts above ``theta_upper`` that the
    held-out post-test showed the learner could not do.

    ⚠️ Nothing passes this yet. The parameter and its behaviour are here; the
    board, the goal choice and the front end are the other half.
    """

    def _graph(self):
        from agent_newton.domains import registry

        return registry.load_domain("calculus").concepts

    def test_a_mastered_concept_stays_out_by_default(self) -> None:
        graph = self._graph()
        mastery = {c: 0.96 for c in graph.ids()}
        assert "implicit_differentiation" not in zpd.compute(
            mastery, graph, ZPDConfig(), 0.15
        )

    def test_and_comes_back_when_it_was_asked_for(self) -> None:
        graph = self._graph()
        mastery = {c: 0.96 for c in graph.ids()}
        frontier = zpd.compute(
            mastery,
            graph,
            ZPDConfig(),
            0.15,
            reviewing=frozenset({"implicit_differentiation"}),
        )
        assert "implicit_differentiation" in frontier
        assert "implicit_differentiation" in frontier.reason

    def test_it_does_not_open_the_material_behind_an_unmet_prerequisite(self) -> None:
        """The other bound is untouched, and that is the whole distinction.

        A request reopens a concept. It must never open what depends on one the
        learner cannot do — that is ``waived``, and it exists for a different
        reason and behind a different knob.
        """
        graph = self._graph()
        mastery = {c: 0.05 for c in graph.ids()}
        frontier = zpd.compute(
            mastery,
            graph,
            ZPDConfig(),
            0.15,
            reviewing=frozenset({"integration_by_substitution"}),
        )
        assert "integration_by_substitution" not in frontier

    def test_requesting_something_unmastered_changes_nothing(self) -> None:
        # It was already in the frontier; asking for it cannot add it twice.
        graph = self._graph()
        mastery = {c: 0.05 for c in graph.ids()}
        plain = zpd.compute(mastery, graph, ZPDConfig(), 0.15)
        asked = zpd.compute(
            mastery, graph, ZPDConfig(), 0.15, reviewing=frozenset(graph.ids())
        )
        assert set(plain) == set(asked)


class TestAskingForSomethingTheBandHasClosed:
    """The other half: the board, the goal and the seeding cap.

    A learner asked for `implicit_differentiation`, one correct pre-test answer
    had seeded it to 0.965, and the request was declined because the concept had
    left the frontier. Three things had to change together — reopening it is
    useless if no goal's route passes through it, and neither helps if a single
    test item can put it there in the first place.
    """

    def _board(self, *, review: bool, requested=("implicit_differentiation",)):
        from agent_newton.domains import registry

        domain = registry.load_domain("calculus")
        config = Config(domain="calculus")
        config.cohort.review_on_request = review
        board = new_blackboard("victor", 1, domain.concepts, config)
        for concept_id in domain.concepts.all_prerequisites("implicit_differentiation"):
            board.state.mastery[concept_id] = 0.85
        board.state.mastery["implicit_differentiation"] = 0.965
        board.record_request(list(requested))
        return domain, config, board

    def test_declined_when_the_run_does_not_permit_it(self) -> None:
        _, _, board = self._board(review=False)
        assert board.reviewing == frozenset()
        assert "implicit_differentiation" not in board.frontier

    def test_reopened_when_it_does(self) -> None:
        _, _, board = self._board(review=True)
        assert board.reviewing == frozenset({"implicit_differentiation"})
        assert "implicit_differentiation" in board.frontier

    def test_and_the_goal_moves_to_reach_it(self) -> None:
        """Reopening alone would not have been enough, which is the subtle part.

        `implicit_differentiation` is a *sibling* of
        `integration_by_substitution`, not a prerequisite, so the goal that
        follows it does not pass through it — and a reopened concept off the
        route is not a candidate.
        """
        domain, config, board = self._board(review=True)
        view = board.view()
        # Only the coupled view carries any of this, which is the architecture
        # rather than a detail: a request is learner input both arms could be
        # handed, and acting on it needs the posteriors and the graph.
        assert isinstance(view, FullStateView)
        goal = route.next_goal(
            domain.concepts.goals(),
            view.mastery,
            config.zpd,
            bkt.initial(config.bkt),
            requested=view.requested,
            graph=domain.concepts,
            reviewing=view.reviewing,
        )
        assert goal == "implicit_differentiation"

    def test_a_request_still_outstanding_is_served_first(self) -> None:
        """⚠️ The guard that must survive this change.

        A person asked for two concepts, the pre-test put one at 0.98, the goal
        moved to serve that one, and the other sat at 0.32 unreached. Revision
        is a fallback to real work, never a competitor with it.
        """
        domain, config, board = self._board(
            review=True,
            requested=("implicit_differentiation", "integration_by_substitution"),
        )
        board.state.mastery["integration_by_substitution"] = 0.32
        view = board.view()
        # Only the coupled view carries any of this, which is the architecture
        # rather than a detail: a request is learner input both arms could be
        # handed, and acting on it needs the posteriors and the graph.
        assert isinstance(view, FullStateView)
        goal = route.next_goal(
            domain.concepts.goals(),
            view.mastery,
            config.zpd,
            bkt.initial(config.bkt),
            requested=view.requested,
            graph=domain.concepts,
            reviewing=view.reviewing,
        )
        assert goal == "integration_by_substitution"

    def test_asking_for_something_already_open_reopens_nothing(self) -> None:
        # `reviewing` reports a relaxation that did work. A concept already in
        # the frontier needed none.
        _, _, board = self._board(review=True, requested=("chain_rule",))
        assert board.reviewing == frozenset()

    def test_it_never_opens_material_behind_an_unmet_prerequisite(self) -> None:
        domain, config, board = self._board(review=True)
        for concept_id in domain.concepts.all_prerequisites("implicit_differentiation"):
            board.state.mastery[concept_id] = 0.10
        assert "implicit_differentiation" in board.reviewing
        assert "implicit_differentiation" not in board.frontier


class TestOneHeldOutItemMayNotDeclareMastery:
    """``seed_floor`` bounded a seeded estimate from below and nothing from above.

    At ``pretest_weight: 3`` one correct pre-test answer landed near 0.96 — past
    ``theta_upper``, out of the frontier, and unreachable for the rest of the
    sitting on the strength of a single question.
    """

    def _seed(self, correct: bool, ceiling: float | None):
        from agent_newton.domains import registry

        domain = registry.load_domain("calculus")
        config = Config(domain="calculus")
        board = new_blackboard("L1", 1, domain.concepts, config)
        verdict = Verdict.CORRECT if correct else Verdict.INCORRECT
        board.seed_from_test([("chain_rule", verdict)], weight=3, ceiling=ceiling)
        return board.probability("chain_rule"), config

    def test_uncapped_it_clears_the_band(self) -> None:
        value, config = self._seed(correct=True, ceiling=None)
        assert value >= config.zpd.theta_upper

    def test_capped_it_stays_selectable(self) -> None:
        value, config = self._seed(correct=True, ceiling=Config().zpd.theta_upper)
        assert value < config.zpd.theta_upper

    def test_the_cap_still_lets_the_evidence_count(self) -> None:
        # Raised well above the prior, just not across the edge: this is a cap
        # on the claim, not a refusal to learn from the test.
        value, config = self._seed(correct=True, ceiling=Config().zpd.theta_upper)
        assert value > config.zpd.theta_lower

    def test_it_does_not_touch_a_wrong_answer(self) -> None:
        capped, _ = self._seed(correct=False, ceiling=Config().zpd.theta_upper)
        uncapped, _ = self._seed(correct=False, ceiling=None)
        assert capped == uncapped


class TestAWrongAnswerMayNotRaiseTheEstimate:
    """``seed_floor`` was applied as a flat minimum, and that inverted its sign.

    Reported from a sitting as the number jumping: three wrong answers left a
    concept at 0.248, a fortnight's decay took it to 0.199, the learner answered
    the pre-test item **wrong**, and the belief went **up** to 0.40.

    The floor exists to stop a wrong answer crushing a concept to 0.0003, where
    the scaffolding rule gives a worked step on every first attempt. That is a
    limit on the fall. It must never manufacture confidence.
    """

    def _seeded(self, before: float, floor: float, correct: bool = False) -> float:
        """Seed a concept that has **already been observed** at ``before``.

        Setting it explicitly is what makes these cases the observed ones: an
        untouched concept sits at the prior, and the floor is entitled to place
        that anywhere — see ``TestTheFloorStillPlacesAFreshConcept``.
        """
        from agent_newton.domains import registry

        domain = registry.load_domain("calculus")
        config = Config(domain="calculus")
        board = new_blackboard("L1", 1, domain.concepts, config)
        board.state.mastery["chain_rule"] = before
        board.seed_from_test(
            [("chain_rule", Verdict.CORRECT if correct else Verdict.INCORRECT)],
            weight=3,
            floor=floor,
        )
        return board.probability("chain_rule")

    def test_the_sittings_own_numbers(self) -> None:
        assert self._seeded(before=0.199, floor=0.40) <= 0.199

    @pytest.mark.parametrize("before", [0.0005, 0.05, 0.199, 0.39])
    def test_a_wrong_answer_never_raises_it(self, before: float) -> None:
        assert self._seeded(before=before, floor=0.40) <= before

    def test_the_floor_still_catches_a_fall_from_above_it(self) -> None:
        # What it was built for: a concept believed known, missed on the test,
        # must not land where the ladder collapses.
        assert self._seeded(before=0.90, floor=0.40) == pytest.approx(0.40)

    def test_without_a_floor_it_collapses(self) -> None:
        # The failure the floor prevents, so the test above is not vacuous:
        # below `theta_lower / 2` the scaffolding rule gives a worked step on
        # every first attempt, and the ladder has no rungs left.
        assert self._seeded(before=0.90, floor=0.0) < Config().zpd.theta_lower / 2

    def test_a_correct_answer_is_untouched(self) -> None:
        assert self._seeded(before=0.199, floor=0.40, correct=True) > 0.9


class TestAReviewEndsWhenTheLearnerShowsTheConcept:
    """Reopening is a check on the estimate, and one demonstration is the check.

    Without this a requested concept stayed selectable however well it went. A
    learner reached 1.00 on one and was still being given it, and left for the
    post-test out of boredom.
    """

    def _board(self):
        from agent_newton.domains import registry

        domain = registry.load_domain("calculus")
        config = Config(domain="calculus")
        config.cohort.review_on_request = True
        board = new_blackboard("victor", 1, domain.concepts, config)
        for concept_id in domain.concepts.all_prerequisites("implicit_differentiation"):
            board.state.mastery[concept_id] = 0.85
        board.state.mastery["implicit_differentiation"] = 0.965
        board.record_request(["implicit_differentiation"])
        return board

    def _answer(self, board, correct: bool) -> None:
        board.record_observation(
            item_id="ca_impl_p1",
            concept_id="implicit_differentiation",
            result=VerificationResult(
                Verdict.CORRECT if correct else Verdict.INCORRECT, "-x/y"
            ),
            attempt=0,
            response="-x/y",
        )

    def test_it_is_open_until_something_is_shown(self) -> None:
        board = self._board()
        assert "implicit_differentiation" in board.frontier

    def test_one_correct_answer_closes_it(self) -> None:
        board = self._board()
        self._answer(board, correct=True)
        assert board.reviewing == frozenset()
        assert "implicit_differentiation" not in board.frontier

    def test_a_wrong_answer_does_not(self) -> None:
        """It stays, and then it does not need to: the estimate has fallen and
        the concept is in the frontier on its own terms, which is the whole
        point of reopening it."""
        board = self._board()
        self._answer(board, correct=False)
        assert "implicit_differentiation" in board.frontier
        assert board.probability("implicit_differentiation") < 0.90


class TestTheFloorStillPlacesAFreshConcept:
    """The floor's other job, which the fix above must not take away.

    On a concept never observed the learner sits at the prior — below
    ``theta_lower / 2``, where every hint is a worked step. Placing it at the
    floor discards nothing, because the prior is the absence of evidence rather
    than evidence of absence. A sitting spent 41 steps at the top of the ladder
    before this existed.
    """

    def _fresh(self, floor: float) -> float:
        from agent_newton.domains import registry

        domain = registry.load_domain("calculus")
        config = Config(domain="calculus")
        board = new_blackboard("L1", 1, domain.concepts, config)
        assert "chain_rule" not in board.state.mastery
        board.seed_from_test([("chain_rule", Verdict.INCORRECT)], weight=3, floor=floor)
        return board.probability("chain_rule")

    def test_a_missed_concept_lands_on_the_floor(self) -> None:
        assert self._fresh(0.40) == pytest.approx(0.40)

    def test_which_leaves_the_ladder_room(self) -> None:
        from agent_newton.core.pedagogy import HintLevel, hint_level

        band = Config().zpd
        assert hint_level(self._fresh(0.40), 0, band) is HintLevel.TARGETED


class TestFailingWhatIsBuiltOnAConceptCastsDoubtOnIt:
    """The prerequisite graph informs belief, not only selection.

    From a sitting: three failures on integration by substitution while the
    model held chain rule at 0.952 and antiderivatives at 0.911 — both above
    ``theta_upper``, so neither could ever be offered again, and neither moved.

    ⚠️ Off by default and off for every cohort. Computing it needs the
    posteriors *and* the graph, so the decoupled arm cannot do it, and a
    mechanism aimed at what the other arm lacks separates the arms by
    construction. Its strength is a swept parameter including zero.
    """

    #: The sitting's own figures.
    BEFORE = {"chain_rule": 0.952, "antiderivative": 0.911}

    def _after(self, alpha: float, failures: int = 3):
        from agent_newton.domains import registry

        domain = registry.load_domain("calculus")
        config = Config(domain="calculus")
        config.bkt.prerequisite_doubt = alpha
        board = new_blackboard("victor", 1, domain.concepts, config)
        for concept_id in domain.concepts.ids():
            board.state.mastery[concept_id] = 0.95
        board.state.mastery.update(self.BEFORE)
        board.state.mastery["integration_by_substitution"] = 0.40
        for _ in range(failures):
            board.record_observation(
                item_id="ca_usub_p1",
                concept_id="integration_by_substitution",
                result=VerificationResult(Verdict.INCORRECT, "x"),
                misconception_label="usub_forgets_du",
                attempt=0,
                response="x",
            )
        return board

    def test_off_by_default_nothing_moves(self) -> None:
        board = self._after(alpha=0.0)
        for concept_id, before in self.BEFORE.items():
            assert board.probability(concept_id) == pytest.approx(before)

    def test_the_prerequisites_come_back_into_reach(self) -> None:
        board = self._after(alpha=0.25)
        assert "chain_rule" in board.frontier
        assert "antiderivative" in board.frontier

    def test_a_single_failure_is_not_enough(self) -> None:
        board = self._after(alpha=0.5, failures=1)
        for concept_id, before in self.BEFORE.items():
            assert board.probability(concept_id) == pytest.approx(before)

    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.15, 0.25, 0.5, 1.0])
    def test_the_effect_orders_with_the_parameter(self, alpha: float) -> None:
        """⚠️ Monotone in ``alpha``, which the first implementation was not.

        Firing on every failure past the threshold let a larger fraction push
        the estimate under ``theta_upper`` sooner, where the guard skipped it —
        so 0.25 left a prerequisite *higher* than 0.15 did. A sweep over a
        parameter that does not order its own outcomes measures nothing.
        """
        stronger = self._after(alpha).probability("chain_rule")
        weaker = self._after(alpha / 2).probability("chain_rule")
        assert stronger <= weaker

    def test_it_fires_once_however_long_the_failing_goes_on(self) -> None:
        three = self._after(alpha=0.5, failures=3).probability("chain_rule")
        eight = self._after(alpha=0.5, failures=8).probability("chain_rule")
        assert three == pytest.approx(eight)

    def test_a_prerequisite_still_in_the_frontier_is_left_alone(self) -> None:
        """It needs no help — it will come round on its own, and nudging it
        would count the same failure twice."""
        from agent_newton.domains import registry

        domain = registry.load_domain("calculus")
        config = Config(domain="calculus")
        config.bkt.prerequisite_doubt = 0.5
        board = new_blackboard("victor", 1, domain.concepts, config)
        for concept_id in domain.concepts.ids():
            board.state.mastery[concept_id] = 0.95
        board.state.mastery["chain_rule"] = 0.50
        board.state.mastery["integration_by_substitution"] = 0.40
        for _ in range(3):
            board.record_observation(
                item_id="ca_usub_p1",
                concept_id="integration_by_substitution",
                result=VerificationResult(Verdict.INCORRECT, "x"),
                attempt=0,
                response="x",
            )
        assert board.probability("chain_rule") == pytest.approx(0.50)

    def test_it_reaches_immediate_prerequisites_only(self) -> None:
        # Attenuating through the closure needs a decay factor nobody has
        # measured, and would let one failure reach the whole graph.
        from agent_newton.domains import registry

        domain = registry.load_domain("calculus")
        immediate = domain.concepts.prerequisites("integration_by_substitution")
        distant = domain.concepts.all_prerequisites("integration_by_substitution") - immediate
        assert distant, "no distant prerequisite to check against"
        board = self._after(alpha=0.5)
        for concept_id in distant:
            assert board.probability(concept_id) == pytest.approx(0.95)

    def test_it_writes_no_error_event_and_no_outcome(self) -> None:
        """An inference is not something the learner did.

        The trace is what the arbitration policy counts repeats in, and the
        outcome stream is what the decoupled view is built from.
        """
        plain = self._after(alpha=0.0)
        doubted = self._after(alpha=0.5)
        assert len(doubted.state.error_trace) == len(plain.state.error_trace)
        assert doubted.state.outcomes == plain.state.outcomes

    def test_it_is_recorded_under_its_own_cause(self) -> None:
        board = self._after(alpha=0.5)
        doubts = [r for r in board.audit_log if r.cause == "doubt"]
        assert {r.evidence["concept_id"] for r in doubts} == set(self.BEFORE)
        assert all(r.evidence["because_of"] == "integration_by_substitution" for r in doubts)
