"""The three framings: what each writer stores, and which seed framing C accepts.

Both guards here have to be shown able to fail. `record` exists because the
dose-matched writers used to keep four fields of the twelve, so framing B was
stored without its interval, its effect size or its discordant counts — a test
that only checked the fields it happens to write would have passed against the
narrow version too. And framing C's seed check is the inverse of every other
seed guard in the project: it refuses a *fresh* seed, so a test asserting only
that some seeds are refused would pass on a guard that refused everything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from framing_c import confirmatory_seed, cycling
from run_paired import record

from agent_newton.config import Config
from agent_newton.core.evaluation.statistics import ALPHA

#: Every quantity the analysis declares it will report, plus the identity of
#: the outcome and the two p-values kept apart. A framing stored without one of
#: these cannot be put in a table beside another framing.
DECLARED = frozenset(
    {
        "outcome",
        "mean_difference",
        "ci95",
        "n_pairs",
        "ties",
        "favouring_coupled",
        "favouring_decoupled",
        "sign_effect_size",
        "sign_p",
        "wilcoxon_p",
        "holm_p",
        "significant",
    }
)


@dataclass
class _Result:
    """Stands in for `PairedResult`, whose fields `record` reads."""

    outcome: str = "remediation"
    n_pairs: int = 160
    discordant: int = 17
    favouring_first: int = 16
    favouring_second: int = 1
    mean_difference: float = 0.0259
    ci95: tuple[float, float] = (0.0143, 0.0396)
    sign_p: float = 0.000275
    wilcoxon_p: float = 0.000352
    sign_effect_size: float = 0.8824

    @property
    def ties(self) -> int:
        return self.n_pairs - self.discordant


class TestWhatARecordStores:
    def test_every_declared_quantity_is_present(self) -> None:
        assert set(record(_Result(), 0.00055)) == DECLARED

    def test_the_two_p_values_stay_apart(self) -> None:
        # The correction replaces neither. A summary carrying only `holm_p`
        # cannot be audited, and one carrying only `sign_p` overstates.
        stored = record(_Result(sign_p=0.000275, wilcoxon_p=0.000352), 0.00055)
        assert stored["sign_p"] == pytest.approx(0.000275)
        assert stored["wilcoxon_p"] == pytest.approx(0.000352)
        assert stored["holm_p"] == pytest.approx(0.00055)

    def test_significance_follows_the_adjusted_value_not_the_raw_one(self) -> None:
        # The rule the analysis applies is Holm over the family, so a raw value
        # inside alpha whose adjusted value is outside it is not significant.
        raw_inside = ALPHA / 2
        assert record(_Result(sign_p=raw_inside), raw_inside)["significant"]
        assert not record(_Result(sign_p=raw_inside), ALPHA * 2)["significant"]

    def test_ties_are_stored_rather_than_left_to_be_derived(self) -> None:
        # The sign test's real sample size is the discordant count, and the tie
        # count is what tells a reader how few learners a result rests on.
        assert record(_Result(n_pairs=160, discordant=17), 1.0)["ties"] == 143


class TestFramingCTakesOnlyTheReportedCohort:
    def _config(self, tmp_path, seed: int | None) -> Config:
        config = Config.from_yaml("experiments/configs/calculus.yaml").model_copy(
            update={"paths": Config.from_yaml(
                "experiments/configs/calculus.yaml"
            ).paths.model_copy(update={"results_dir": tmp_path})}
        )
        if seed is not None:
            directory = tmp_path / f"paired_{config.domain}"
            directory.mkdir(parents=True)
            (directory / "summary.json").write_text(json.dumps({"seed": seed}))
        return config

    def test_it_reads_the_seed_from_the_reported_summary(self, tmp_path) -> None:
        declared, path = confirmatory_seed(self._config(tmp_path, 20260811))
        assert declared == 20260811
        assert path.name == "summary.json"

    def test_it_refuses_when_there_is_no_comparison_to_re_frame(self, tmp_path) -> None:
        # A framing of nothing is not a framing. Without this the script would
        # run a cohort and report it as an alternative accounting of a result
        # that does not exist.
        with pytest.raises(SystemExit, match="nothing to re-frame"):
            confirmatory_seed(self._config(tmp_path, None))

    def test_a_different_stored_seed_moves_what_is_accepted(self, tmp_path) -> None:
        # The guard tracks the summary rather than a constant, so a comparison
        # re-run on a new seed does not silently keep licensing the old one.
        assert confirmatory_seed(self._config(tmp_path, 20260901))[0] == 20260901


class TestCyclingChangesOneThing:
    def test_it_sets_on_exhaustion_and_nothing_else(self) -> None:
        config = Config.from_yaml("experiments/configs/calculus.yaml")
        cycled = cycling(config)
        assert config.agents.planner.on_exhaustion == "stop"
        assert cycled.agents.planner.on_exhaustion == "cycle"
        # `advance_after` decides when the walk steps forward, so moving it too
        # would make this a second manipulation rather than a framing.
        assert cycled.agents.planner.advance_after == config.agents.planner.advance_after
        assert cycled.agents.planner.emphasis == config.agents.planner.emphasis
        assert cycled.simulator == config.simulator
        assert cycled.cohort == config.cohort

    def test_the_original_config_is_not_mutated(self) -> None:
        config = Config.from_yaml("experiments/configs/calculus.yaml")
        cycling(config)
        assert config.agents.planner.on_exhaustion == "stop"
