"""Persistence across sessions.

The store exists so a plan can outlive the sitting that made it. What is tested
here is mostly the ways that could go wrong quietly: a learner resuming into the
wrong arm's history, a profile inherited across the comparison, or a state that
round-trips lossily and silently changes what the planner sees.
"""

from __future__ import annotations

import pytest

from agent_newton.config import Config, SimulatorConfig
from agent_newton.core.simulator import sample_profile
from agent_newton.core.state.schema import LearnerState, Plan, Utterance
from agent_newton.core.state.store import new_blackboard
from agent_newton.domains import registry
from agent_newton.domains.base import VerificationResult, Verdict
from agent_newton.store import LearnerStore
from agent_newton.store.ground_truth import ProfileStore


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


@pytest.fixture
def store(tmp_path):
    with LearnerStore(tmp_path / "learners.db") as db:
        yield db


def worked_state(toy, **mastery: float) -> LearnerState:
    config = Config.model_validate({"domain": "toy_algebra"})
    board = new_blackboard("L1", seed=1, graph=toy.concepts, config=config)
    board.record_observation(
        item_id="i1",
        concept_id="distribute",
        result=VerificationResult(Verdict.INCORRECT, "a"),
        misconception_label="distribute_first_term_only",
    )
    board.record_reflection("I was unsure", "i1", "distribute", kind="working")
    board.record_plan(Plan(goal="solve_linear"))
    board.state.mastery.update(mastery)
    return board.state


class TestSessionsAndResuming:
    def test_a_learner_with_no_history_resumes_nothing(self, store) -> None:
        # None means "never sat one", which is deliberately not the same as a
        # learner whose state happens to be empty.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        assert store.latest_state("L1", "coupled") is None

    def test_a_state_round_trips(self, store, toy) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        original = worked_state(toy, distribute=0.42)
        store.close_session(session, state=original, stop_reason="budget_spent")

        restored = store.latest_state("L1", "coupled")
        assert restored is not None
        assert restored.mastery == original.mastery
        assert restored.plan == original.plan
        assert restored.error_trace == original.error_trace
        assert restored.reflections == original.reflections
        assert restored.t == original.t

    def test_the_latest_session_is_the_one_resumed(self, store, toy) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        for value in (0.2, 0.5, 0.8):
            session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
            store.close_session(session, state=worked_state(toy, distribute=value))

        restored = store.latest_state("L1", "coupled")
        assert restored is not None
        assert restored.mastery["distribute"] == 0.8

    def test_the_sequence_index_advances(self, store) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        assert store.next_index("L1", "coupled") == 0
        store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        assert store.next_index("L1", "coupled") == 1

    def test_registering_a_learner_twice_is_harmless(self, store) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        store.ensure_learner("L1", "simulated", "toy_algebra")
        assert len(store.learners()) == 1


class TestTheArmsKeepSeparateHistories:
    """The same person under two architectures is two histories.

    If one arm resumed into the other's state, the paired comparison would be
    destroyed — and it would not look broken, it would look like a result.
    """

    def test_state_does_not_cross_arms(self, store, toy) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        store.close_session(session, state=worked_state(toy, distribute=0.9))

        assert store.latest_state("L1", "coupled") is not None
        assert store.latest_state("L1", "decoupled") is None

    def test_sequence_indices_are_per_arm(self, store) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        assert store.next_index("L1", "coupled") == 2
        assert store.next_index("L1", "decoupled") == 0

    def test_profiles_do_not_cross_arms(self, store, tmp_path, toy) -> None:
        # A profile remediated in one arm must never be inherited by the other:
        # the decoupled arm would start with the coupled arm's teaching already
        # applied.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        profile = sample_profile("L1", 1, toy.misconceptions, SimulatorConfig())
        store.close_session(session, state=worked_state(toy))

        with ProfileStore(store.path) as profiles:
            profiles.save(session, profile)
            assert profiles.latest("L1", "coupled") is not None
            assert profiles.latest("L1", "decoupled") is None


class TestProfilesArePersistedWithTheirStartingPoint:
    def test_firing_and_initial_both_survive(self, store, toy) -> None:
        # Remediation lowers firing and forgetting raises it, so without the
        # starting point there is nothing to report remediation as a proportion
        # of — `remediation_ratio` would silently rebase on the current values.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        profile = sample_profile("L1", 1, toy.misconceptions, SimulatorConfig())
        label = next(iter(profile.firing))
        profile.remediate(label, 0.5)
        store.close_session(session, state=worked_state(toy))

        with ProfileStore(store.path) as profiles:
            profiles.save(session, profile)
            restored = profiles.latest("L1", "coupled")

        assert restored is not None
        assert restored.firing == profile.firing
        assert dict(restored.initial) == dict(profile.initial)
        assert restored.remediation_ratio() == pytest.approx(profile.remediation_ratio())


class TestTheQueryableProjections:
    """⚠️ Each row must belong to the sitting it was said in.

    The utterance rows were projected from ``state.reflections``, and a resumed
    state carries every word the learner has ever said — so each new session
    wrote the whole history again under its own id. One real learner's table
    held 81 rows and 27 distinct texts across three sittings, two of which had
    said nothing at all. Nothing read the table yet, which is the only reason it
    never produced a wrong number.
    """

    def _sitting(self, store, toy, said: str | None) -> int:
        """One sitting that may or may not produce a word of its own.

        Built through the board, because that is how an utterance actually comes
        to exist: recorded on the blackboard, which writes it to the audit log,
        which is per sitting. The state is the resumed one either way.
        """
        config = Config.model_validate({"domain": "toy_algebra"})
        board = new_blackboard("L1", 1, toy.concepts, config)
        if said is not None:
            board.record_reflection(said, "i1", "distribute", kind="working")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        store.close_session(
            session, state=worked_state(toy), audit_log=board.audit_log
        )
        return session

    def test_utterances_are_queryable_across_sessions(self, store, toy) -> None:
        # The reason a database was chosen: planning is meant to read what the
        # learner said in *previous* sittings, not only this one.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        self._sitting(store, toy, "I was unsure about the second step")
        self._sitting(store, toy, "I think I have it now")

        said = store.utterances("L1", "coupled")
        assert len(said) == 2
        assert {row["concept_id"] for row in said} == {"distribute"}
        assert [row["text"] for row in said] == [
            "I was unsure about the second step",
            "I think I have it now",
        ]
        assert store.utterances("L1", "coupled", concept_id="nothing_here") == []

    def test_a_sitting_that_said_nothing_contributes_nothing(self, store, toy) -> None:
        # The resumed state still carries the earlier sitting's words, and they
        # must not be written a second time under this session's id.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        self._sitting(store, toy, "I was unsure about the second step")
        second = self._sitting(store, toy, None)

        said = store.utterances("L1", "coupled")
        assert len(said) == 1
        assert [row for row in said if row["session_id"] == second] == []

    def test_the_same_words_are_not_counted_twice(self, store, toy) -> None:
        # The shape the defect took: rows kept growing while distinct texts did
        # not, so anything counting what a learner said would have multiplied it
        # by the number of sittings since.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        for _ in range(3):
            self._sitting(store, toy, None)
        self._sitting(store, toy, "the only thing I ever said")

        said = store.utterances("L1", "coupled")
        assert len(said) == len({row["text"] for row in said}) == 1

    def test_events_are_queryable_by_cause(self, store, toy) -> None:
        store.ensure_learner("L1", "simulated", "toy_algebra")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        state = worked_state(toy)
        board = new_blackboard(
            "L1", 1, toy.concepts, Config.model_validate({"domain": "toy_algebra"})
        )
        store.close_session(session, state=state, audit_log=board.audit_log)

        assert list(store.events("L1", "coupled", cause="annotation"))

    def test_the_state_blob_is_authoritative(self, store, toy) -> None:
        # The projections are for querying. Anything the session resumes from
        # comes back out of the blob, so a projection that drifted could not
        # silently change what the planner sees.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        session = store.open_session(learner_id="L1", arm="coupled", config_hash="h")
        state = worked_state(toy)
        state.reflections.append(
            Utterance(text="only in the blob", item_id="i9", concept_id="distribute")
        )
        store.close_session(session, state=state)

        restored = store.latest_state("L1", "coupled")
        assert restored is not None
        assert "only in the blob" in [u.text for u in restored.reflections]


class TestTheStoreIsAFile:
    def test_it_survives_being_closed_and_reopened(self, tmp_path, toy) -> None:
        path = tmp_path / "nested" / "learners.db"
        with LearnerStore(path) as first:
            first.ensure_learner("L1", "human", "toy_algebra")
            session = first.open_session(learner_id="L1", arm="coupled", config_hash="h")
            first.close_session(session, state=worked_state(toy, distribute=0.66))

        with LearnerStore(path) as second:
            restored = second.latest_state("L1", "coupled")
            assert restored is not None
            assert restored.mastery["distribute"] == 0.66

    def test_humans_and_simulated_learners_are_distinguishable(self, store) -> None:
        # So the two can be compared later without inferring the kind from
        # whether a profile row happens to exist.
        store.ensure_learner("L1", "simulated", "toy_algebra")
        store.ensure_learner("H1", "human", "toy_algebra")
        assert [r["learner_id"] for r in store.learners(kind="human")] == ["H1"]
        assert [r["learner_id"] for r in store.learners(kind="simulated")] == ["L1"]
