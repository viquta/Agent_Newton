"""Config validators.

These guard design invariants, not typos. Each test below corresponds to a way a
plausible-looking config edit could silently invalidate results.
"""

from __future__ import annotations

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
                    "planner": {"impl": "deterministic"},
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
                "planner": {"impl": "deterministic"},
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
