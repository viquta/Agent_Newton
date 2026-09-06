"""The paired statistics, against cases whose answers are known independently.

Every p-value the cohort comparison reports comes from this module, and until
now nothing under ``tests/`` imported it. The cases here are hand-worked rather
than generated: each one states an answer that can be checked by arithmetic, so
a failure says which property broke rather than only that something moved.

The Holm case reads the committed paired summary and re-derives its ``holm_p``
column from its ``sign_p`` column, which ties the implementation to the numbers
already published in ``results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agent_newton.core.evaluation.statistics import (
    ALPHA,
    bootstrap_ci,
    compare,
    holm_bonferroni,
    paired_differences,
    rank_biserial,
    sign_test,
    wilcoxon,
)

ROOT = Path(__file__).resolve().parents[2]
PAIRED_SUMMARY = ROOT / "results" / "paired_calculus" / "summary.json"


def arm(**values: float) -> dict[str, dict]:
    """One arm's per-learner rows, keyed by learner id."""
    return {learner: {"outcome": value} for learner, value in values.items()}


class TestTheSignTest:
    def test_every_pair_tied_returns_one(self) -> None:
        assert sign_test(np.zeros(10)) == 1.0

    def test_five_positive_and_none_negative(self) -> None:
        # Two-sided exact binomial on five successes out of five: 2 * 0.5**5.
        assert sign_test(np.ones(5)) == pytest.approx(0.0625)

    def test_ties_are_dropped_rather_than_counted(self) -> None:
        # Padding with zeros changes the pair count but not the discordant set,
        # so the p-value must not move.
        assert sign_test(np.array([1.0, 1.0, 1.0, 1.0, 1.0])) == sign_test(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        )

    def test_it_is_two_sided(self) -> None:
        assert sign_test(np.ones(5)) == sign_test(-np.ones(5))


class TestWilcoxon:
    def test_every_pair_tied_returns_one(self) -> None:
        assert wilcoxon(np.zeros(10)) == 1.0

    def test_a_symmetric_sample_does_not_separate_the_arms(self) -> None:
        assert wilcoxon(np.array([1.0, -1.0, 2.0, -2.0])) == pytest.approx(1.0)

    def test_it_uses_magnitude_where_the_sign_test_does_not(self) -> None:
        # Same signs in both samples, so the sign test cannot tell them apart;
        # Wilcoxon can, because the magnitudes differ.
        lopsided = np.array([9.0, 9.0, 9.0, -1.0])
        balanced = np.array([1.0, 1.0, 1.0, -9.0])
        assert sign_test(lopsided) == sign_test(balanced)
        assert wilcoxon(lopsided) < wilcoxon(balanced)


class TestRankBiserial:
    def test_all_discordant_pairs_favouring_the_first_arm(self) -> None:
        assert rank_biserial(np.ones(7)) == 1.0

    def test_all_favouring_the_second(self) -> None:
        assert rank_biserial(-np.ones(7)) == -1.0

    def test_an_even_split_is_zero(self) -> None:
        assert rank_biserial(np.array([1.0, 1.0, -1.0, -1.0])) == 0.0

    def test_ties_are_outside_the_denominator(self) -> None:
        # The share is of *discordant* pairs. Ties must not pull it toward zero,
        # or a high tie rate would read as a small effect.
        assert rank_biserial(np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])) == 1.0

    def test_no_discordant_pairs_at_all(self) -> None:
        assert rank_biserial(np.zeros(5)) == 0.0


class TestTheBootstrapInterval:
    def test_constant_differences_collapse_the_interval(self) -> None:
        rng = np.random.default_rng(0)
        low, high = bootstrap_ci(np.full(50, 2.5), rng)
        assert low == pytest.approx(2.5)
        assert high == pytest.approx(2.5)

    def test_no_differences_at_all(self) -> None:
        assert bootstrap_ci(np.array([]), np.random.default_rng(0)) == (0.0, 0.0)

    def test_the_same_seed_reproduces_the_interval(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        first = bootstrap_ci(values, np.random.default_rng(20260811))
        second = bootstrap_ci(values, np.random.default_rng(20260811))
        assert first == second

    def test_it_brackets_the_observed_mean(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        low, high = bootstrap_ci(values, np.random.default_rng(1), draws=4000)
        assert low < values.mean() < high


class TestHolmBonferroni:
    def test_it_reproduces_the_committed_summary(self) -> None:
        """The published ``holm_p`` column, re-derived from its ``sign_p`` column."""
        stored = json.loads(PAIRED_SUMMARY.read_text())["results"]
        adjusted = holm_bonferroni([row["sign_p"] for row in stored])
        for row, value in zip(stored, adjusted):
            assert value == pytest.approx(row["holm_p"], rel=1e-12), row["outcome"]

    def test_the_smallest_p_pays_the_full_family_size(self) -> None:
        assert holm_bonferroni([0.001, 0.4, 0.5, 0.6])[0] == pytest.approx(0.004)

    def test_a_single_hypothesis_is_unchanged(self) -> None:
        assert holm_bonferroni([0.03]) == [0.03]

    def test_adjusted_values_never_decrease(self) -> None:
        # Scaling alone would give 0.06 then 0.04. Holm is a step-down procedure:
        # once a rank is raised, no later rank may fall below it.
        assert holm_bonferroni([0.04, 0.03]) == pytest.approx([0.06, 0.06])

    def test_it_clamps_at_one(self) -> None:
        assert holm_bonferroni([0.6, 0.7]) == pytest.approx([1.0, 1.0])

    def test_it_returns_values_in_the_order_given(self) -> None:
        # Input is deliberately unsorted: the caller zips the result against its
        # own outcome list, so a sorted return would mislabel every row.
        assert holm_bonferroni([0.5, 0.001, 0.2]) == pytest.approx([0.5, 0.003, 0.4])

    def test_it_is_never_more_lenient_than_the_raw_p(self) -> None:
        raw = [0.001, 0.02, 0.3, 0.9]
        assert all(a >= p for a, p in zip(holm_bonferroni(raw), raw))


class TestPairedDifferences:
    def test_learners_are_matched_by_id_not_position(self) -> None:
        first = arm(L0=1.0, L1=5.0)
        reordered = {"L1": {"outcome": 2.0}, "L0": {"outcome": 3.0}}
        assert list(paired_differences(first, reordered, "outcome")) == [-2.0, 3.0]

    def test_it_refuses_arms_that_ran_different_learners(self) -> None:
        with pytest.raises(ValueError, match="did not run the same learners"):
            paired_differences(arm(L0=1.0, L1=1.0), arm(L0=1.0, L2=1.0), "outcome")

    def test_a_missing_value_counts_as_zero(self) -> None:
        first = {"L0": {"outcome": None}}
        assert list(paired_differences(first, arm(L0=2.0), "outcome")) == [-2.0]


class TestTheDirectionConvention:
    """``favouring_first`` means the first arm's value is larger — nothing more.

    On a lower-is-better outcome the winning arm is therefore ``favouring_second``,
    and a reader who takes the field name at face value gets the result backwards.
    """

    def test_favouring_first_means_the_first_arm_is_larger(self) -> None:
        result = compare(
            "outcome", arm(L0=3.0), arm(L0=1.0), np.random.default_rng(0)
        )
        assert result.favouring_first == 1
        assert result.favouring_second == 0
        assert result.rank_biserial == 1.0

    def test_on_a_lower_is_better_outcome_the_winner_is_favouring_second(self) -> None:
        # The first arm reaches a smaller distance-to-goal, which is the better
        # result, and the count that records it is favouring_second.
        result = compare(
            "outcome",
            arm(L0=1.0, L1=2.0),
            arm(L0=7.0, L1=8.0),
            np.random.default_rng(0),
        )
        assert result.favouring_second == 2
        assert result.rank_biserial == -1.0
        assert result.mean_difference < 0


class TestThePairedResult:
    def test_significance_reads_the_sign_test_not_wilcoxon(self) -> None:
        # The primary test is the sign test; the Wilcoxon p sits beside it and
        # does not decide the verdict.
        result = compare(
            "outcome",
            arm(**{f"L{i}": 1.0 for i in range(6)}),
            arm(**{f"L{i}": 0.0 for i in range(6)}),
            np.random.default_rng(0),
        )
        assert result.sign_p < ALPHA
        assert result.significant is (result.sign_p < ALPHA)

    def test_ties_are_the_pairs_the_arms_treated_identically(self) -> None:
        result = compare(
            "outcome",
            arm(L0=1.0, L1=0.0, L2=0.0),
            arm(L0=0.0, L1=0.0, L2=0.0),
            np.random.default_rng(0),
        )
        assert result.n_pairs == 3
        assert result.discordant == 1
        assert result.ties == 2
