"""Agents coordinate only through shared state.

This is the formal property that makes the architecture a blackboard rather
than a set of components that happen to share a store. If one agent held a
reference to another, information could move between them without passing
through the state — and without appearing in the audit log, which is what makes
a run reconstructible.

Composition *within* a role is fine and is not a back channel: the model-backed
planner wraps a deterministic one as its guardrail, which is one role deciding
in two stages, not two roles talking.
"""

from __future__ import annotations

from hypothesis import given as hyp_given
from hypothesis import settings
from hypothesis import strategies as st

from agent_newton.config import ArbitrationConfig, Config
from agent_newton.core.agents.diagnostic import NoisedOracleDiagnostic, OracleDiagnostic
from agent_newton.core.agents.llm import LLMDiagnostic, LLMPlanner, LLMTutor
from agent_newton.core.agents.planner import FixedOrderPlanner, FrontierPlanner
from agent_newton.core.agents.tutor import TemplateTutor
from agent_newton.core.arbitration.policy import ArbitrationPolicy
from agent_newton.core.orchestration.session import build_session
from agent_newton.core.state.schema import ErrorEvent
from agent_newton.core.state.zpd import Frontier
from agent_newton.domains import registry

ROLES = {
    "tutor": (TemplateTutor, LLMTutor),
    "diagnostic": (OracleDiagnostic, NoisedOracleDiagnostic, LLMDiagnostic),
    "planner": (FrontierPlanner, FixedOrderPlanner, LLMPlanner),
}


def _reachable(obj: object, depth: int = 4) -> list[object]:
    """Objects reachable through attributes, to a bounded depth."""
    if depth == 0 or not hasattr(obj, "__dict__"):
        return []
    found: list[object] = []
    for value in vars(obj).values():
        found.append(value)
        found.extend(_reachable(value, depth - 1))
    return found


def test_no_agent_reaches_an_agent_of_another_role() -> None:
    config = Config.model_validate(
        {
            "domain": "toy_algebra",
            "arm": "coupled",
            "simulator": {"surface": "symbolic"},
            "agents": {
                "tutor": {"impl": "llm", "provider": "ollama", "model": "m"},
                "diagnostic": {"impl": "llm", "provider": "ollama", "model": "m"},
                "planner": {"impl": "llm", "provider": "ollama", "model": "m"},
            },
        }
    )
    session = build_session("L0000", 1, registry.load_domain("toy_algebra"), config)

    for role, agent in (
        ("tutor", session.tutor),
        ("diagnostic", session.diagnostic),
        ("planner", session.planner),
    ):
        forbidden = tuple(
            cls for other, classes in ROLES.items() if other != role for cls in classes
        )
        offenders = [
            type(found).__name__
            for found in _reachable(agent)
            if isinstance(found, forbidden)
        ]
        assert not offenders, (
            f"the {role} can reach {offenders} without going through the "
            f"blackboard; information moved that way appears in no audit log"
        )


def test_the_check_would_catch_a_back_channel() -> None:
    """A guard that cannot fail proves nothing."""

    class ChattyTutor(TemplateTutor):
        def __init__(self, band, diagnostic) -> None:  # noqa: ANN001
            super().__init__(band)
            self.diagnostic = diagnostic

    from agent_newton.config import ZPDConfig

    tutor = ChattyTutor(ZPDConfig(), OracleDiagnostic())
    forbidden = tuple(ROLES["diagnostic"])
    assert any(isinstance(found, forbidden) for found in _reachable(tutor))


class TestPersistenceIsNotABackChannel:
    """The store keeps beliefs and ground truth in one database file.

    In memory the separation is structural: a profile is an object the agents are
    never handed. A shared database threatens that, because anything holding a
    connection could read the table the diagnostic agent exists to infer. So the
    split is restored in code — ``LearnerStore`` cannot touch profiles at all,
    and ``ProfileStore`` lives in a module agents may not import.
    """

    def _agent_sources(self) -> list[tuple[str, str]]:
        from pathlib import Path

        import agent_newton.core.agents as agents_pkg

        root = Path(agents_pkg.__file__).parent
        return [(p.name, p.read_text()) for p in sorted(root.glob("*.py"))]

    #: The import, not the words. ``OracleAccess.observe_ground_truth`` is a
    #: method name that legitimately appears throughout the agents, and matching
    #: on the bare phrase flagged the very capability that makes reading ground
    #: truth explicit.
    GROUND_TRUTH_IMPORT = "store.ground_truth"

    def test_no_agent_module_imports_ground_truth(self) -> None:
        offenders = [
            name
            for name, source in self._agent_sources()
            if self.GROUND_TRUTH_IMPORT in source
        ]
        assert not offenders, (
            f"{offenders} import the ground-truth store; the diagnostic agent "
            f"exists to infer what it holds, so an accuracy figure measured with "
            f"it in scope would mean nothing"
        )

    def test_no_agent_module_imports_the_store_at_all(self) -> None:
        # Even the belief side. The store is the runner's, and an agent holding
        # one could read across sessions without the state carrying it — which
        # is the same back channel, wearing a database.
        offenders = [
            name
            for name, source in self._agent_sources()
            if "agent_newton.store" in source
        ]
        assert not offenders, f"{offenders} import the store directly"

    def test_the_check_can_fail(self, tmp_path) -> None:
        # The shape it looks for, so the scan is not passing vacuously.
        stray = tmp_path / "stray_agent.py"
        stray.write_text("from agent_newton.store.ground_truth import ProfileStore\n")
        assert self.GROUND_TRUTH_IMPORT in stray.read_text()

    def test_the_check_does_not_fire_on_the_capability_name(self) -> None:
        # `observe_ground_truth` is how reading ground truth is made an explicit
        # capability rather than an argument every agent happens to receive. The
        # scan must not flag the thing that keeps the line honest.
        assert self.GROUND_TRUTH_IMPORT not in "def observe_ground_truth(self, label): ..."

    def test_the_learner_store_cannot_read_profiles(self) -> None:
        from agent_newton.store import LearnerStore

        surface = [name for name in dir(LearnerStore) if not name.startswith("_")]
        assert not [name for name in surface if "profile" in name.lower()]

    def test_a_profile_is_not_reachable_from_any_agent(self) -> None:
        # The in-memory property, restated now that profiles are persisted:
        # `build_session` samples one and hands it only to the learner and, when
        # the config names it, to the oracle planner.
        from agent_newton.core.simulator.profile import MisconceptionProfile

        config = Config.model_validate(
            {"domain": "toy_algebra", "arm": "coupled",
             "agents": {"tutor": {"impl": "template"},
                        "diagnostic": {"impl": "llm"},
                        "planner": {"impl": "goal_directed"}}}
        )
        session = build_session("L0000", 1, registry.load_domain("toy_algebra"), config)
        for role, agent in (
            ("tutor", session.tutor),
            ("diagnostic", session.diagnostic),
            ("planner", session.planner),
        ):
            assert not [
                f for f in _reachable(agent) if isinstance(f, MisconceptionProfile)
            ], f"the {role} can reach the learner's true profile"


class TestReplanningFiresExactlyWhenItShould:
    """The condition, checked over generated states rather than examples."""

    @settings(max_examples=200, deadline=None)
    @hyp_given(
        theta=st.floats(min_value=0.01, max_value=0.9),
        delta=st.floats(min_value=0.0, max_value=1.0),
        items_since=st.integers(min_value=0, max_value=10),
        min_items=st.integers(min_value=0, max_value=5),
    )
    def test_a_mastery_move_replans_iff_it_clears_both_gates(
        self, theta: float, delta: float, items_since: int, min_items: int
    ) -> None:
        policy = ArbitrationPolicy(
            ArbitrationConfig(theta=theta, min_items_between_replans=min_items)
        )
        policy.accept({"c": 0.1})
        for _ in range(items_since):
            policy.note_item()

        after = 0.1 + delta
        decision = policy.evaluate(
            current_concept="c",
            mastery={"c": after},
            frontier=Frontier(frozenset({"c"})),
            error_trace=[],
            prior=0.15,
        )

        # The expectation must use the same arithmetic the policy does.
        # `(0.1 + delta) - 0.1` is not `delta` in floating point, and hypothesis
        # finds the case where that difference straddles the threshold.
        observed_delta = abs(after - 0.1)
        over_threshold = observed_delta > theta
        rate_ok = items_since >= min_items
        assert decision.replan == (over_threshold and rate_ok)

    @settings(max_examples=100, deadline=None)
    @hyp_given(
        confirmed=st.integers(min_value=0, max_value=6),
        unconfirmed=st.integers(min_value=0, max_value=6),
        k=st.integers(min_value=1, max_value=4),
    )
    def test_only_verifier_confirmed_errors_count(
        self, confirmed: int, unconfirmed: int, k: int
    ) -> None:
        # A diagnostic label is an opinion that an error occurred; the verifier
        # is what establishes one. Unconfirmed events must never reach the
        # threshold, however many of them there are.
        policy = ArbitrationPolicy(
            ArbitrationConfig(k_repeats=k, min_items_between_replans=0, theta=0.99)
        )
        policy.accept({"c": 0.1})

        trace = [
            ErrorEvent(t=i, item_id="i", concept_id="c", misconception_label="m",
                       verifier_label="incorrect")
            for i in range(confirmed)
        ] + [
            ErrorEvent(t=100 + i, item_id="i", concept_id="c", misconception_label="m",
                       verifier_label="unparseable")
            for i in range(unconfirmed)
        ]

        decision = policy.evaluate(
            current_concept="c",
            mastery={"c": 0.1},
            frontier=Frontier(frozenset({"c"})),
            error_trace=trace,
            prior=0.15,
        )
        assert decision.replan == (confirmed >= k)
