"""The error-propagation study's run conditions.

The study only means anything if the three conditions differ in the diagnostic
and in nothing else. That is a property of the code building the configs, so it
is checked here rather than trusted: a condition that also moved the seed, the
item budget or the planner would produce a difference that looks like error
propagation and is not.
"""

from __future__ import annotations

import json

import pytest
from run_propagation import (
    ARMS,
    CONDITIONS,
    condition_config,
    measured_noise_rate,
    paired_differences,
    summarise,
)

from agent_newton.config import Config


@pytest.fixture(scope="module")
def base() -> Config:
    return Config.from_yaml("experiments/configs/calculus.yaml")


class TestConditionsVaryOneThing:
    @pytest.mark.parametrize("condition", CONDITIONS)
    @pytest.mark.parametrize("arm", ARMS)
    def test_only_the_diagnostic_and_the_arm_move(
        self, base: Config, condition: str, arm: str
    ) -> None:
        built = condition_config(base, condition, arm, noise_rate=0.25)
        left = built.model_dump(mode="json")
        right = base.model_dump(mode="json")
        # run_name carries the condition so run directories stay distinguishable.
        for key in ("arm", "run_name"):
            left.pop(key)
            right.pop(key)
        left_agents = left.pop("agents")
        right_agents = right.pop("agents")

        assert left == right, "a condition changed something other than the diagnostic"
        assert left_agents["tutor"] == right_agents["tutor"]
        assert left_agents["planner"] == right_agents["planner"]

    @pytest.mark.parametrize(
        ("condition", "impl"),
        [("oracle", "oracle"), ("noised", "noised_oracle"), ("llm", "llm")],
    )
    def test_each_condition_selects_its_implementation(
        self, base: Config, condition: str, impl: str
    ) -> None:
        built = condition_config(base, condition, "coupled", noise_rate=0.25)
        assert built.agents.diagnostic.impl == impl

    def test_the_noised_condition_carries_the_rate(self, base: Config) -> None:
        built = condition_config(base, "noised", "coupled", noise_rate=0.25)
        assert built.agents.diagnostic.noise_rate == 0.25

    def test_the_other_conditions_carry_no_rate(self, base: Config) -> None:
        # A noise rate left on an oracle would be inert but misleading in the
        # manifest, which is what a reader trusts afterwards.
        for condition in ("oracle", "llm"):
            built = condition_config(base, condition, "coupled", noise_rate=0.25)
            assert built.agents.diagnostic.noise_rate == 0.0

    def test_both_arms_keep_the_same_seed(self, base: Config) -> None:
        # The paired design rests on this: same seed, same learner, both arms.
        seeds = {
            condition_config(base, condition, arm, 0.25).seed
            for condition in CONDITIONS
            for arm in ARMS
        }
        assert seeds == {base.seed}

    def test_an_unknown_condition_is_refused(self, base: Config) -> None:
        with pytest.raises(ValueError, match="unknown condition"):
            condition_config(base, "perfect", "coupled", 0.25)


class TestTheNoiseRateComesFromTheMeasurement:
    def test_it_is_one_minus_accuracy(self, tmp_path) -> None:
        path = tmp_path / "summary.json"
        path.write_text(json.dumps({"accuracy": 0.6316, "cases": 38}))
        assert measured_noise_rate(path) == pytest.approx(0.3684)

    def test_an_empty_evaluation_is_refused(self, tmp_path) -> None:
        # Accuracy is 0.0 over no cases, which would silently become a noise
        # rate of 1.0 — a condition where the diagnostic is never right.
        path = tmp_path / "summary.json"
        path.write_text(json.dumps({"accuracy": 0.0, "cases": 0}))
        with pytest.raises(ValueError, match="no cases"):
            measured_noise_rate(path)


class TestPairing:
    def _arm(self, rows: list[tuple[str, float]]) -> dict:
        return {"per_learner": [{"learner_id": i, "remediation": v} for i, v in rows]}

    def test_learners_are_matched_by_id_not_position(self) -> None:
        coupled = self._arm([("L0000", 0.8), ("L0001", 0.2)])
        decoupled = self._arm([("L0001", 0.1), ("L0000", 0.5)])
        assert paired_differences(coupled, decoupled, "remediation") == pytest.approx(
            [0.3, 0.1]
        )

    def test_a_mismatched_cohort_is_refused(self) -> None:
        # Silently intersecting would compare a subset and report it as the whole.
        coupled = self._arm([("L0000", 0.8), ("L0001", 0.2)])
        decoupled = self._arm([("L0000", 0.5)])
        with pytest.raises(ValueError, match="same learners"):
            paired_differences(coupled, decoupled, "remediation")

    def test_the_split_counts_every_learner(self) -> None:
        summary = summarise([0.4, 0.0, -0.2, 0.0])
        assert summary["favour_coupled"] == 1
        assert summary["ties"] == 2
        assert summary["favour_decoupled"] == 1
        assert summary["n"] == 4
        # A mean alone hides the shape: a few large differences against a
        # majority of ties reads the same as a consistent small effect.
        assert summary["mean"] == pytest.approx(0.05)
