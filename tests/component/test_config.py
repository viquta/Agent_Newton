"""Config validators.

These guard design invariants, not typos. Each test below corresponds to a way a
plausible-looking config edit could silently invalidate results.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_newton.config import BKTConfig, Config, ZPDConfig, model_family


class TestModelFamily:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gemma4:12b", "google"),
            ("gemma4:31b-mlx", "google"),
            ("gpt-oss:20b", "openai"),
            ("gpt-4o", "openai"),
            ("claude-opus-5", "anthropic"),
            ("qwen2.5-coder:14b", "alibaba"),
        ],
    )
    def test_known_families(self, model: str, expected: str) -> None:
        assert model_family(model) == expected

    def test_unknown_models_never_collide(self) -> None:
        # Two unrecognised models must not land in a shared bucket, or the
        # circularity check would pass them as "different" only by accident —
        # or worse, flag unrelated models as the same lineage.
        assert model_family("someone-elses-model:7b") != model_family("another-model:7b")


class TestCircularityControl:
    """The simulated learner must not be produced by the model that diagnoses it."""

    def test_rejects_same_family(self) -> None:
        with pytest.raises(ValidationError, match="Circularity control violated"):
            Config.model_validate(
                {
                    "simulator": {"surface": "llm", "surface_model": {"model": "gemma4:12b"}},
                    "agents": {"diagnostic": {"impl": "llm", "model": "gemma4:12b"}},
                }
            )

    def test_accepts_different_families(self) -> None:
        config = Config.model_validate(
            {
                "simulator": {"surface": "llm", "surface_model": {"model": "gpt-oss:20b"}},
                "agents": {"diagnostic": {"impl": "llm", "model": "gemma4:12b"}},
            }
        )
        assert config.uses_llm()

    def test_inapplicable_when_simulator_is_symbolic(self) -> None:
        # No model generates the student's step, so there is no lineage to share.
        config = Config.model_validate(
            {
                "simulator": {"surface": "symbolic", "surface_model": {"model": "gemma4:12b"}},
                "agents": {"diagnostic": {"impl": "llm", "model": "gemma4:12b"}},
            }
        )
        assert config.simulator.surface == "symbolic"

    def test_inapplicable_when_diagnostic_is_an_oracle(self) -> None:
        # An oracle reads the injected label rather than inferring it, so it
        # cannot be contaminated by the simulator's lineage.
        config = Config.model_validate(
            {
                "simulator": {"surface": "llm", "surface_model": {"model": "gemma4:12b"}},
                "agents": {"diagnostic": {"impl": "oracle", "model": "gemma4:12b"}},
            }
        )
        assert config.agents.diagnostic.impl == "oracle"


class TestZPDBand:
    def test_rejects_empty_band(self) -> None:
        with pytest.raises(ValidationError, match="ZPD band is empty"):
            ZPDConfig(theta_lower=0.9, theta_upper=0.7)

    def test_rejects_degenerate_band(self) -> None:
        with pytest.raises(ValidationError, match="ZPD band is empty"):
            ZPDConfig(theta_lower=0.8, theta_upper=0.8)

    def test_accepts_ordered_band(self) -> None:
        band = ZPDConfig(theta_lower=0.7, theta_upper=0.9)
        assert band.theta_lower < band.theta_upper


class TestBKT:
    def test_rejects_degenerate_parameters(self) -> None:
        # guess + slip >= 1 makes evidence update backwards (Beck & Chang, 2007):
        # a correct answer would *lower* the mastery estimate.
        with pytest.raises(ValidationError, match="degenerate"):
            BKTConfig(p_guess=0.6, p_slip=0.5)

    def test_accepts_standard_parameters(self) -> None:
        assert BKTConfig(p_guess=0.20, p_slip=0.10).p_transit > 0


class TestNoisedOracle:
    def test_rejects_zero_noise(self) -> None:
        # A noised oracle with no noise is just an oracle; silently allowing it
        # would collapse the three diagnostic conditions to two.
        with pytest.raises(ValidationError, match="just an oracle"):
            Config.model_validate(
                {"agents": {"diagnostic": {"impl": "noised_oracle", "noise_rate": 0.0}}}
            )

    def test_accepts_nonzero_noise(self) -> None:
        config = Config.model_validate(
            {"agents": {"diagnostic": {"impl": "noised_oracle", "noise_rate": 0.23}}}
        )
        assert config.agents.diagnostic.noise_rate == pytest.approx(0.23)


class TestUsesLLM:
    def test_fully_deterministic_run_uses_no_llm(self) -> None:
        config = Config.model_validate(
            {
                "simulator": {"surface": "symbolic"},
                "agents": {
                    "tutor": {"impl": "template"},
                    "diagnostic": {"impl": "oracle"},
                    "planner": {"impl": "goal_directed"},
                },
            }
        )
        assert not config.uses_llm()

    @pytest.mark.parametrize(
        "override",
        [
            {"agents": {"tutor": {"impl": "llm"}}},
            {"agents": {"diagnostic": {"impl": "llm"}}},
            {"agents": {"planner": {"impl": "llm"}}},
            {"simulator": {"surface": "llm"}},
        ],
    )
    def test_any_llm_component_flips_the_flag(self, override: dict) -> None:
        base = {
            "simulator": {"surface": "symbolic"},
            "agents": {
                "tutor": {"impl": "template"},
                "diagnostic": {"impl": "oracle"},
                "planner": {"impl": "goal_directed"},
            },
        }
        for key, value in override.items():
            base[key] = {**base.get(key, {}), **value}
        assert Config.model_validate(base).uses_llm()


class TestContentHash:
    def test_is_stable_across_equal_configs(self) -> None:
        assert Config().content_hash() == Config().content_hash()

    def test_changes_with_any_field(self) -> None:
        assert Config().content_hash() != Config(seed=1).content_hash()


class TestSmokeConfig:
    def test_shipped_smoke_config_is_valid_and_model_free(self) -> None:
        config = Config.from_yaml("experiments/configs/smoke.yaml")
        assert not config.uses_llm(), "the smoke config must never invoke a model"
        assert config.domain == "toy_algebra"


CONFIG_DIR = Path("experiments/configs")
#: The one config a person sits in front of. Everywhere else is an experiment.
HUMAN_CONFIGS = {"demo.yaml"}


class TestPretestSeedingStaysOutOfExperiments:
    """Seeding from the pre-test must not reach a cohort by accident.

    It updates no state a test was not entitled to update, but it changes the
    *starting frontier* — and the coupled planner routes from the frontier while
    the decoupled one structurally cannot. A cohort run with it on would report
    an advantage handed to one arm before the first item, indistinguishable from
    the effect under study.
    """

    def _experiment_configs(self) -> list[Path]:
        return sorted(
            path
            for path in CONFIG_DIR.glob("*.yaml")
            if path.name not in HUMAN_CONFIGS
        )

    def test_there_are_experiment_configs_to_check(self) -> None:
        # Without this the scan below passes vacuously if the directory moves.
        assert self._experiment_configs()

    def test_no_experiment_config_seeds_from_the_pretest(self) -> None:
        for path in self._experiment_configs():
            config = Config.from_yaml(path)
            assert not config.cohort.seed_from_pretest, (
                f"{path.name} seeds the learner model from the pre-test; that "
                f"moves the starting frontier and only one arm can use one"
            )

    def test_the_default_is_off(self) -> None:
        assert not Config().cohort.seed_from_pretest

    def test_the_check_can_fail(self, tmp_path: Path) -> None:
        # A guard that cannot fail proves nothing. This is the shape the scan
        # above is looking for, and it must be rejected.
        stray = tmp_path / "stray.yaml"
        stray.write_text("domain: toy_algebra\ncohort:\n  seed_from_pretest: true\n")
        assert Config.from_yaml(stray).cohort.seed_from_pretest


class TestTheDwellingCapStaysOutOfExperiments:
    """A cap on how long a concept may be worked changes what one arm does.

    Only the coupled planner dwells — the decoupled one advances on consecutive
    correct answers and never revisits — so a cap set for a cohort would change
    one arm's behaviour and not the other's, on top of the manipulation under
    test. It stays a swept parameter whose default is today's behaviour, so that
    every number already measured was measured without it.
    """

    def _experiment_configs(self) -> list[Path]:
        return sorted(
            path
            for path in CONFIG_DIR.glob("*.yaml")
            if path.name not in HUMAN_CONFIGS
        )

    def test_no_experiment_config_caps_dwelling(self) -> None:
        for path in self._experiment_configs():
            config = Config.from_yaml(path)
            assert config.cohort.max_visits_per_concept is None, (
                f"{path.name} caps how long a concept may be worked; only the "
                f"coupled arm dwells, so that changes one arm and not the other"
            )

    def test_the_default_is_unlimited(self) -> None:
        # None rather than a large number: unlimited is what every measured
        # result was produced under, and a large default would be a policy
        # nobody chose.
        assert Config().cohort.max_visits_per_concept is None

    def test_the_check_can_fail(self, tmp_path: Path) -> None:
        stray = tmp_path / "stray.yaml"
        stray.write_text(
            "domain: toy_algebra\ncohort:\n  max_visits_per_concept: 3\n"
        )
        assert Config.from_yaml(stray).cohort.max_visits_per_concept == 3


class TestTheScaffoldingLadderStaysOutOfExperiments:
    """A cohort must run the ladder its numbers were produced under.

    Nothing here can move a cohort number today — the simulated learner improves
    only through ``receive_hint``, so the support level reaches nothing it
    responds to, and there is an integration test proving it. That inertness is
    a property of *this* simulator, though, not of the design. The moment a
    mechanism is added that responds to how much help was given, a cohort
    quietly running the richer ladder stops being the run every measured figure
    came from — and it would look like the numbers moved rather than like the
    policy changed.
    """

    def _experiment_configs(self) -> list[Path]:
        return sorted(
            path
            for path in CONFIG_DIR.glob("*.yaml")
            if path.name not in HUMAN_CONFIGS
        )

    def test_no_experiment_config_changes_the_ladder(self) -> None:
        for path in self._experiment_configs():
            config = Config.from_yaml(path)
            assert config.scaffolding.policy == "banded", (
                f"{path.name} runs the {config.scaffolding.policy!r} scaffolding "
                f"ladder; every measured result was produced under 'banded'"
            )

    def test_no_experiment_config_offers_support_at_presentation(self) -> None:
        for path in self._experiment_configs():
            config = Config.from_yaml(path)
            assert not config.scaffolding.offer_at_presentation, (
                f"{path.name} shows the rule beside the question; that is "
                f"support no cohort run has ever been given"
            )

    def test_the_defaults_are_todays_behaviour(self) -> None:
        assert Config().scaffolding.policy == "banded"
        assert not Config().scaffolding.offer_at_presentation

    def test_the_human_config_turns_both_on(self) -> None:
        # Otherwise the scan above passes because nothing anywhere uses them,
        # and it would keep passing after the feature was removed.
        demo = Config.from_yaml(CONFIG_DIR / "demo.yaml")
        assert demo.scaffolding.policy == "banded_plus"
        assert demo.scaffolding.offer_at_presentation

    def test_the_check_can_fail(self, tmp_path: Path) -> None:
        # A guard that cannot fail proves nothing. Both shapes the scans look
        # for, and both must be rejected.
        stray = tmp_path / "stray.yaml"
        stray.write_text(
            "domain: toy_algebra\nscaffolding:\n  policy: banded_plus\n"
            "  offer_at_presentation: true\n"
        )
        loaded = Config.from_yaml(stray)
        assert loaded.scaffolding.policy == "banded_plus"
        assert loaded.scaffolding.offer_at_presentation


class TestReviewOnRequestStaysOutOfExperiments:
    """Reopening a mastered concept relaxes the band, so a cohort must not.

    Doubly inert there — nothing outside the demo records a request, so the set
    it acts on is empty — and the knob exists so that is a decision rather than
    an accident. It is also the one relaxation a learner can trigger, and the
    coupled arm is the only one that could act on it.
    """

    def _experiment_configs(self) -> list[Path]:
        return sorted(
            path
            for path in CONFIG_DIR.glob("*.yaml")
            if path.name not in HUMAN_CONFIGS
        )

    def test_no_experiment_config_reopens_on_request(self) -> None:
        for path in self._experiment_configs():
            config = Config.from_yaml(path)
            assert not config.cohort.review_on_request, (
                f"{path.name} reopens concepts the band has closed; no measured "
                f"result was produced with the upper bound relaxed"
            )

    def test_the_default_is_off(self) -> None:
        assert not Config().cohort.review_on_request

    def test_the_human_config_turns_it_on(self) -> None:
        assert Config.from_yaml(CONFIG_DIR / "demo.yaml").cohort.review_on_request

    def test_the_check_can_fail(self, tmp_path: Path) -> None:
        stray = tmp_path / "stray.yaml"
        stray.write_text("domain: toy_algebra\ncohort:\n  review_on_request: true\n")
        assert Config.from_yaml(stray).cohort.review_on_request


class TestPrerequisiteDoubtStaysOutOfExperiments:
    """⚠️ The most consequential of these scans, and the reason is the claim.

    Charging a repeated failure back to a prerequisite needs the posteriors
    *and* the graph, so the decoupled arm structurally cannot do it. A mechanism
    aimed squarely at what the other arm lacks would separate the arms **by
    construction** — which the prerequisite-dependence sweep names explicitly as
    the follow-up not to go looking for.

    So it is off here, and if it is ever run in a cohort its strength must be
    swept across a range **including zero**, with the whole sweep reported. A
    single favourable point would be assuming the conclusion.
    """

    def _experiment_configs(self) -> list[Path]:
        return sorted(
            path
            for path in CONFIG_DIR.glob("*.yaml")
            if path.name not in HUMAN_CONFIGS
        )

    def test_no_experiment_config_doubts_prerequisites(self) -> None:
        for path in self._experiment_configs():
            config = Config.from_yaml(path)
            assert config.bkt.prerequisite_doubt == 0.0, (
                f"{path.name} charges failures back to prerequisites; only one "
                f"arm can do that, so it would separate them by construction"
            )

    def test_the_default_is_off(self) -> None:
        assert Config().bkt.prerequisite_doubt == 0.0

    def test_the_check_can_fail(self, tmp_path: Path) -> None:
        stray = tmp_path / "stray.yaml"
        stray.write_text("domain: toy_algebra\nbkt:\n  prerequisite_doubt: 0.25\n")
        assert Config.from_yaml(stray).bkt.prerequisite_doubt == 0.25


class TestTheTeachingLayerStaysOutOfExperiments:
    """A cohort must not be taught, and the reason is stated in advance.

    Nothing here can move a cohort number today. A lesson targets a *concept*,
    and the simulated learner improves only when a hint names a misconception it
    holds — so the exposition reaches nothing it responds to, and there is a
    test on that beside the feature.

    It stays off anyway, because that inertness is a property of *this*
    simulator rather than of the design. The moment a mechanism is added that
    responds to having been taught, a cohort quietly running with teaching on
    stops being the run every measured figure came from — and it would look like
    the numbers moved rather than like the policy changed.
    """

    def _experiment_configs(self) -> list[Path]:
        return sorted(
            path
            for path in CONFIG_DIR.glob("*.yaml")
            if path.name not in HUMAN_CONFIGS
        )

    def test_no_experiment_config_explains_anything(self) -> None:
        for path in self._experiment_configs():
            config = Config.from_yaml(path)
            assert config.teaching.explain_after == 0, (
                f"{path.name} explains a concept after "
                f"{config.teaching.explain_after} errors; no cohort run has ever "
                f"been given exposition, and a run that was is not comparable "
                f"with the ones the results came from"
            )

    def test_the_default_is_off(self) -> None:
        # Zero rather than a plausible threshold: off is what every measured
        # result was produced under, and a default nobody chose is a policy
        # nobody chose.
        assert Config().teaching.explain_after == 0

    def test_the_check_can_fail(self, tmp_path: Path) -> None:
        # A guard that cannot fail proves nothing. This is the shape the scan
        # above is looking for, and it must be rejected.
        stray = tmp_path / "stray.yaml"
        stray.write_text("domain: toy_algebra\nteaching:\n  explain_after: 3\n")
        assert Config.from_yaml(stray).teaching.explain_after == 3
