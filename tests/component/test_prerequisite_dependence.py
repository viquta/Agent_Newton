"""Making sequencing matter to the simulated learner, on a dial.

⚠️ Read §7c before changing anything here. The generator had no notion of
sequencing at all: a step was a function of the profile, the item's probes and a
seeded roll, and a measurement found prerequisite order to make no difference to
what a learner ends up knowing — +0.0008, p = 1.0, one discordant pair in 137,
matched on coverage. So the outcome the architecture is judged on was one
nothing in the generator could move.

This is that dependence, made explicit and given a strength. The rules it has to
obey are what keep it a measurement rather than an assumption:

* **Zero is today.** Every number already measured must be reproduced exactly by
  the code that has the dial, or the dial is not a dial.
* **It reads the profile, never the learner model.** A mechanism keyed on the
  system's belief would let the coupled arm's own estimate drive the learner,
  and it would win by construction.
* **It is swept, never chosen.** A strength tuned until the coupled arm wins
  assumes the conclusion, which is the one thing the sweep exists to avoid.
"""

from __future__ import annotations

import pytest

from agent_newton.config import Config, SimulatorConfig
from agent_newton.core.orchestration.session import build_session
from agent_newton.core.simulator import SimulatedLearner, sample_profile
from agent_newton.core.simulator.profile import MisconceptionProfile, solidity
from agent_newton.domains import registry


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


def config_for(dependence: float = 0.0, **overrides) -> Config:
    return Config.model_validate(
        {
            "domain": "toy_algebra",
            "arm": "coupled",
            "simulator": {"surface": "symbolic", "prerequisite_dependence": dependence},
            "agents": {
                "tutor": {"impl": "template"},
                "diagnostic": {"impl": "oracle"},
                "planner": {"impl": "goal_directed"},
            },
            **overrides,
        }
    )


class TestZeroIsToday:
    """The load-bearing one. Every measured number was produced without this."""

    def test_the_default_is_off(self) -> None:
        assert Config().simulator.prerequisite_dependence == 0.0

    @pytest.mark.parametrize("arm", ["coupled", "decoupled"])
    @pytest.mark.parametrize("domain_name", ["toy_algebra", "calculus"])
    def test_a_cohort_at_zero_is_unchanged(self, arm: str, domain_name: str) -> None:
        domain = registry.load_domain(domain_name)
        for i in range(4):
            plain = Config.model_validate(
                {
                    "domain": domain_name, "arm": arm,
                    "simulator": {"surface": "symbolic"},
                    "agents": {
                        "tutor": {"impl": "template"},
                        "diagnostic": {"impl": "oracle"},
                        "planner": {"impl": "goal_directed"},
                    },
                }
            )
            dialled = plain.model_copy(deep=True)
            dialled.simulator.prerequisite_dependence = 0.0

            before = build_session(f"L{i:04d}", plain.seed, domain, plain).run()
            after = build_session(f"L{i:04d}", dialled.seed, domain, dialled).run()
            assert before.remediation_ratio == after.remediation_ratio
            assert before.items_attempted == after.items_attempted
            assert before.diagnoses == after.diagnoses
            assert before.gain == after.gain

    def test_at_zero_the_multiplier_is_the_configured_factor(self, toy) -> None:
        # And nothing is computed on that path: the untouched case is untouched.
        profile = sample_profile("L1", 1, toy.misconceptions, SimulatorConfig())
        learner = SimulatedLearner(profile, toy, SimulatorConfig())
        held = next(iter(profile.firing))
        assert learner._efficacy(held) == SimulatorConfig().remediation_factor


class TestTheMultiplier:
    """Its shape, at the ends where it has to be right."""

    def _learner(self, toy, dependence: float, firing: dict[str, float]):
        profile = MisconceptionProfile(learner_id="L1", seed=1, firing=dict(firing))
        config = SimulatorConfig(prerequisite_dependence=dependence)
        return SimulatedLearner(profile, toy, config), profile

    def test_solid_foundations_land_the_hint_in_full(self, toy) -> None:
        # A learner with nothing beneath the concept is never penalised for the
        # dial existing, whatever it is set to.
        learner, _ = self._learner(toy, 1.0, {"distribute_first_term_only": 0.8})
        assert learner._efficacy("distribute_first_term_only") == pytest.approx(
            SimulatorConfig().remediation_factor
        )

    def test_nothing_beneath_it_and_the_hint_does_nothing(self, toy) -> None:
        # `distribute` sits on `combine_like_terms`. A learner holding that one
        # unremediated has no foundation under the hint at all.
        learner, _ = self._learner(
            toy, 1.0, {"distribute_first_term_only": 0.8, "combine_unlike_terms": 0.8}
        )
        assert learner._efficacy("distribute_first_term_only") == pytest.approx(1.0)

    def test_teaching_the_prerequisite_first_makes_the_hint_land(self, toy) -> None:
        # The claim the whole mechanism exists to represent, as a number.
        learner, profile = self._learner(
            toy, 1.0, {"distribute_first_term_only": 0.8, "combine_unlike_terms": 0.8}
        )
        before = learner._efficacy("distribute_first_term_only")
        profile.firing["combine_unlike_terms"] = 0.0  # taught, and it took
        after = learner._efficacy("distribute_first_term_only")
        assert after < before
        assert after == pytest.approx(SimulatorConfig().remediation_factor)

    @pytest.mark.parametrize("dependence", [0.25, 0.5, 0.75])
    def test_it_is_monotone_in_the_dial(self, toy, dependence: float) -> None:
        weak, _ = self._learner(
            toy, dependence,
            {"distribute_first_term_only": 0.8, "combine_unlike_terms": 0.8},
        )
        strong, _ = self._learner(
            toy, 1.0,
            {"distribute_first_term_only": 0.8, "combine_unlike_terms": 0.8},
        )
        # Higher multiplier = less reduction = a weaker hint.
        assert (
            SimulatorConfig().remediation_factor
            < weak._efficacy("distribute_first_term_only")
            < strong._efficacy("distribute_first_term_only")
        )


class TestSolidity:
    def test_a_root_concept_is_always_solid(self, toy) -> None:
        # Nothing beneath it, so nothing can be shaky underneath it.
        profile = MisconceptionProfile(
            learner_id="L1", seed=1, firing={"combine_unlike_terms": 0.9}
        )
        assert solidity("integer_arithmetic", profile, toy.misconceptions, toy.concepts) == 1.0

    def test_a_learner_holding_nothing_beneath_is_solid(self, toy) -> None:
        profile = MisconceptionProfile(
            learner_id="L1", seed=1, firing={"distribute_first_term_only": 0.9}
        )
        # `distribute_first_term_only` is *on* distribute, not beneath it.
        assert solidity("distribute", profile, toy.misconceptions, toy.concepts) == 1.0

    def test_it_measures_the_whole_closure(self, toy) -> None:
        # `solve_linear` rests on `distribute`, which rests on
        # `combine_like_terms`. A grandparent counts: "the foundations" is
        # everything a concept stands on, not the storey below it.
        profile = MisconceptionProfile(
            learner_id="L1", seed=1, firing={"combine_unlike_terms": 0.9}
        )
        assert solidity("solve_linear", profile, toy.misconceptions, toy.concepts) == 0.0

    def test_it_rises_as_the_foundation_is_taught(self, toy) -> None:
        profile = MisconceptionProfile(
            learner_id="L1", seed=1, firing={"combine_unlike_terms": 0.8}
        )
        assert solidity("distribute", profile, toy.misconceptions, toy.concepts) == 0.0
        profile.remediate("combine_unlike_terms", 0.5)
        assert solidity("distribute", profile, toy.misconceptions, toy.concepts) == 0.5
        profile.firing["combine_unlike_terms"] = 0.0
        assert solidity("distribute", profile, toy.misconceptions, toy.concepts) == 1.0


class TestItReadsGroundTruthAndNotTheModel:
    """The circularity control, and the reason the result would mean anything.

    A mechanism keyed on the system's belief would let the coupled arm's own
    estimate drive the learner's behaviour: the arm that maintains a learner
    model would win because it maintains one, not because it routes better.
    """

    def test_the_mechanism_takes_no_mastery_argument(self) -> None:
        import inspect

        taken = set(inspect.signature(solidity).parameters)
        assert taken == {"concept_id", "profile", "catalogue", "graph"}

    def test_the_simulator_never_reads_a_posterior(self) -> None:
        """The grep §7c ran by hand, kept as a check — over code, not prose.

        `core/simulator/` may know the graph, because the mechanism needs the
        prerequisite relation. It must never read what the system *believes*.

        Read from the parse tree rather than the file text: the first version
        searched for the word "mastery" and failed on the docstring explaining
        why mastery must not be read. A check that cannot survive its own
        explanation is measuring the wrong thing.
        """
        import ast
        import pathlib

        forbidden = {
            "mastery",
            "error_trace",
            "frontier",
            "FullStateView",
            "ItemCorrectnessView",
            "Blackboard",
        }
        for path in pathlib.Path("src/agent_newton/core/simulator").glob("*.py"):
            tree = ast.parse(path.read_text())
            used = {
                node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
            } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            trespass = used & forbidden
            assert not trespass, f"{path.name} reads the learner model: {trespass}"

    def test_two_arms_see_the_same_learner_at_any_strength(self, toy) -> None:
        # The profile is drawn from (seed, learner_id) and nothing about the arm
        # enters it. True before the dial; it must stay true with it.
        config = SimulatorConfig(prerequisite_dependence=1.0)
        one = sample_profile("L0007", 20260812, toy.misconceptions, config)
        two = sample_profile("L0007", 20260812, toy.misconceptions, config)
        assert one.firing == two.firing
