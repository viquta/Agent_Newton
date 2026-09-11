"""The paired statistics, against cases whose answers are known independently.

Every p-value the cohort comparison reports comes from this module. The cases
here are hand-worked rather than generated: each one states an answer that can
be checked by arithmetic, so a failure says which property broke rather than
only that something moved.

Three kinds of check, and each procedure gets whichever apply:

* **Hand-worked values** — inputs small enough that the answer is a fraction.
* **An outside implementation** that shares no code with the module —
  ``math.comb`` for the exact binomial, ``scipy.stats.wilcoxon`` called
  directly on the discordant pairs with ``method="exact"``,
  ``scipy.stats.bootstrap`` for the percentile interval, and statsmodels'
  ``multipletests`` for Holm.
* **Invariants** that must hold for any input, checked with Hypothesis.

The committed paired summary is read twice. Once to re-derive its ``holm_p``
column from its ``sign_p`` column, and once to re-derive ``sign_p`` and the
effect size from the discordant counts alone, with the binomial formula rather
than the module. Both tie the implementation to numbers already in ``results/``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.multitest import multipletests

from agent_newton.core.evaluation.statistics import (
    ALPHA,
    bootstrap_ci,
    compare,
    holm_bonferroni,
    paired_differences,
    sign_effect_size,
    sign_test,
    wilcoxon,
)

ROOT = Path(__file__).resolve().parents[2]
PAIRED_SUMMARY = ROOT / "results" / "paired_calculus" / "summary.json"


def arm(**values: float) -> dict[str, dict]:
    """One arm's per-learner rows, keyed by learner id."""
    return {learner: {"outcome": value} for learner, value in values.items()}


def stored_rows() -> list[dict]:
    return json.loads(PAIRED_SUMMARY.read_text())["results"]


def stored_effect_size(row: dict) -> float:
    """The effect size, under its current key or the one it had before 2026-09-10."""
    return row["sign_effect_size"] if "sign_effect_size" in row else row["rank_biserial"]


def two_sided_binomial(positive: int, trials: int) -> float:
    """Exact two-sided binomial p-value at p = 0.5, from the definition.

    Twice the smaller tail, clamped at one. At p = 0.5 the distribution is
    symmetric, so this coincides with the "outcomes no more likely than the
    observed one" definition scipy uses — which is what lets a sum over
    ``math.comb`` stand as an outside check on ``binomtest``.
    """
    if trials == 0:
        return 1.0
    lower = sum(math.comb(trials, k) for k in range(positive + 1)) / 2**trials
    upper = sum(math.comb(trials, k) for k in range(positive, trials + 1)) / 2**trials
    return min(1.0, 2 * min(lower, upper))


def rank_biserial(differences: np.ndarray) -> float:
    """The matched-pairs rank-biserial correlation, ``(W+ - W-) / (W+ + W-)``.

    Here only to show that the module's effect size is *not* this. It ranks the
    absolute differences; the module's statistic counts directions.
    """
    discordant = differences[differences != 0]
    ranks = stats.rankdata(np.abs(discordant))
    w_plus = ranks[discordant > 0].sum()
    w_minus = ranks[discordant < 0].sum()
    return float((w_plus - w_minus) / (w_plus + w_minus))


#: Small signed integers with zeros — the shape of a paired difference vector.
small_differences = st.lists(st.integers(-3, 3), min_size=1, max_size=40).map(
    lambda values: np.array(values, dtype=float)
)


class TestTheSignTest:
    def test_every_pair_tied_returns_one(self) -> None:
        assert sign_test(np.zeros(10)) == 1.0

    def test_five_positive_and_none_negative(self) -> None:
        # Two-sided exact binomial on five successes out of five: 2 * 0.5**5.
        assert sign_test(np.ones(5)) == pytest.approx(0.0625)

    def test_a_mixed_sample_by_hand(self) -> None:
        # Four of five favour the first arm. Both tails at or beyond that:
        # (C(5,4) + C(5,5) + C(5,1) + C(5,0)) / 32 = 12 / 32.
        assert sign_test(np.array([1.0, 1.0, 1.0, 1.0, -1.0])) == pytest.approx(0.375)

    def test_one_discordant_pair_cannot_reject(self) -> None:
        assert sign_test(np.array([0.0, 0.0, 1.0])) == 1.0

    def test_an_even_split_is_one(self) -> None:
        assert sign_test(np.array([1.0, -1.0])) == 1.0
        assert sign_test(np.array([3.0, -0.5, 2.0, -7.0])) == 1.0

    def test_ties_are_dropped_rather_than_counted(self) -> None:
        # Padding with zeros changes the pair count but not the discordant set,
        # so the p-value must not move.
        assert sign_test(np.array([1.0, 1.0, 1.0, 1.0, 1.0])) == sign_test(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        )

    def test_it_is_two_sided(self) -> None:
        assert sign_test(np.ones(5)) == sign_test(-np.ones(5))

    def test_magnitudes_do_not_enter(self) -> None:
        assert sign_test(np.array([100.0, 0.001, 5.0, -0.2])) == sign_test(
            np.array([1.0, 1.0, 1.0, -1.0])
        )

    @settings(max_examples=100, deadline=None)
    @given(differences=small_differences)
    def test_it_is_the_exact_binomial_on_the_discordant_pairs(
        self, differences: np.ndarray
    ) -> None:
        positive = int((differences > 0).sum())
        negative = int((differences < 0).sum())
        expected = two_sided_binomial(positive, positive + negative)
        assert sign_test(differences) == pytest.approx(expected)
        assert 0.0 < sign_test(differences) <= 1.0

    @settings(max_examples=100, deadline=None)
    @given(differences=small_differences)
    def test_only_the_signs_matter(self, differences: np.ndarray) -> None:
        assert sign_test(differences) == sign_test(np.sign(differences))
        assert sign_test(differences) == sign_test(-differences)
        assert sign_test(differences) == sign_test(np.concatenate([differences, np.zeros(5)]))

    def test_it_reproduces_the_committed_summary_from_its_counts(self) -> None:
        """The published ``sign_p``, re-derived from the two discordant counts.

        With the binomial formula, not the module — on the primary that is
        16 of 17 one way, ``2 * (C(17,16) + C(17,17)) / 2**17 = 36 / 131072``.
        """
        for row in stored_rows():
            expected = two_sided_binomial(
                row["favouring_coupled"], row["favouring_coupled"] + row["favouring_decoupled"]
            )
            assert row["sign_p"] == pytest.approx(expected, rel=1e-12), row["outcome"]


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

    def test_all_positive_by_hand(self) -> None:
        # Every rank on one side: two tails of one arrangement each, out of 2**n.
        assert wilcoxon(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == pytest.approx(2 / 32)
        assert wilcoxon(np.arange(1.0, 7.0)) == pytest.approx(2 / 64)

    def test_ties_are_removed_before_scipy_chooses_a_method(self) -> None:
        """The p-value must not depend on how many tied pairs surround the
        discordant ones.

        scipy picks exact-versus-asymptotic from the length of the array it is
        handed. Passing the tied pairs through — as this function did until
        2026-09-11 — made a cohort of 160 with 7 discordant pairs look like a
        sample of 160, and every recorded Wilcoxon p was the normal approximation.
        """
        discordant = np.array([0.3, 0.1, 0.25, 0.05, 0.2, 0.15, -0.12])
        padded = np.concatenate([discordant, np.zeros(153)])
        assert wilcoxon(padded) == wilcoxon(discordant)

    def test_with_equal_magnitudes_it_reduces_to_the_sign_test(self) -> None:
        # All ranks tie, so the rank sum is a count of directions and the two
        # tests are the same test. scipy enumerates the arrangements up to 13
        # discordant pairs, which is where the equality is exact.
        discordant = np.array([1.0] * 6 + [-1.0])
        cohort = np.concatenate([discordant, np.zeros(153)])
        assert wilcoxon(cohort) == pytest.approx(sign_test(cohort))
        assert wilcoxon(cohort) == pytest.approx(0.125)

    def test_it_agrees_with_the_exact_distribution_on_distinct_magnitudes(self) -> None:
        # Seventeen discordant pairs, sixteen one way, no two alike — the
        # primary outcome's shape — against scipy asked for the exact
        # distribution outright.
        rng = np.random.default_rng(0)
        discordant = np.concatenate([rng.uniform(0.01, 0.1, 16), [-0.05]])
        cohort = np.concatenate([discordant, np.zeros(143)])
        reference = stats.wilcoxon(discordant, method="exact")
        assert wilcoxon(cohort) == pytest.approx(
            float(reference.pvalue)  # pyright: ignore[reportAttributeAccessIssue]
        )

    def test_it_is_two_sided(self) -> None:
        sample = np.array([0.3, 0.1, 0.25, -0.05, 0.2])
        assert wilcoxon(sample) == pytest.approx(wilcoxon(-sample))


class TestTheSignEffectSize:
    def test_all_discordant_pairs_favouring_the_first_arm(self) -> None:
        assert sign_effect_size(np.ones(7)) == 1.0

    def test_all_favouring_the_second(self) -> None:
        assert sign_effect_size(-np.ones(7)) == -1.0

    def test_an_even_split_is_zero(self) -> None:
        assert sign_effect_size(np.array([1.0, 1.0, -1.0, -1.0])) == 0.0

    def test_a_mixed_sample_by_hand(self) -> None:
        # Three of four one way: 2 * 0.75 - 1. Sixteen of seventeen: 2 * 16/17 - 1.
        assert sign_effect_size(np.array([1.0, 1.0, 1.0, -1.0])) == pytest.approx(0.5)
        assert sign_effect_size(np.array([1.0] * 16 + [-1.0])) == pytest.approx(15 / 17)

    def test_ties_are_outside_the_denominator(self) -> None:
        # The share is of *discordant* pairs. Ties must not pull it toward zero,
        # or a high tie rate would read as a small effect.
        assert sign_effect_size(np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])) == 1.0

    def test_no_discordant_pairs_at_all(self) -> None:
        assert sign_effect_size(np.zeros(5)) == 0.0

    def test_it_is_the_discordant_margin_over_the_discordant_count(self) -> None:
        result = compare(
            "outcome",
            arm(L0=3.0, L1=2.0, L2=1.0, L3=0.0, L4=5.0),
            arm(L0=1.0, L1=1.0, L2=4.0, L3=0.0, L4=1.0),
            np.random.default_rng(0),
        )
        margin = result.favouring_first - result.favouring_second
        assert result.sign_effect_size == pytest.approx(margin / result.discordant)

    def test_it_is_not_the_rank_biserial_correlation(self) -> None:
        """The two coincide when every discordant pair goes one way and part
        company when the split is mixed, which is what the rename recorded."""
        lopsided = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        assert sign_effect_size(lopsided) == rank_biserial(lopsided) == 1.0
        # Three small one way, one large the other: directions say 0.5, ranks
        # say (1 + 2 + 3 - 4) / 10.
        mixed = np.array([0.1, 0.2, 0.3, -0.9])
        assert sign_effect_size(mixed) == pytest.approx(0.5)
        assert rank_biserial(mixed) == pytest.approx(0.2)

    @settings(max_examples=100, deadline=None)
    @given(differences=small_differences)
    def test_only_the_signs_matter(self, differences: np.ndarray) -> None:
        assert sign_effect_size(differences) == sign_effect_size(np.sign(differences))
        assert sign_effect_size(differences) == pytest.approx(-sign_effect_size(-differences))
        assert sign_effect_size(differences) == sign_effect_size(
            np.concatenate([differences, np.zeros(5)])
        )
        assert -1.0 <= sign_effect_size(differences) <= 1.0

    def test_it_reproduces_the_committed_summary_from_its_counts(self) -> None:
        for row in stored_rows():
            first, second = row["favouring_coupled"], row["favouring_decoupled"]
            expected = (first - second) / (first + second)
            assert stored_effect_size(row) == pytest.approx(expected, rel=1e-12), row["outcome"]


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

    def test_it_covers_a_known_mean_at_the_stated_rate(self) -> None:
        """Four hundred samples of thirty from a distribution whose mean is
        known; the interval must contain it about 95 % of the time.

        This is the check the bracketing test above cannot make — an interval
        at the wrong percentiles still brackets the observed mean. With these
        draws a 5th/95th interval covers 0.895, below the bound.
        """
        rng = np.random.default_rng(20260811)
        true_mean, replicates = 0.5, 400
        covered = 0
        for _ in range(replicates):
            sample = rng.normal(true_mean, 1.0, 30)
            low, high = bootstrap_ci(sample, rng)
            covered += low <= true_mean <= high
        assert 0.91 <= covered / replicates <= 0.98

    def test_it_agrees_with_scipy_percentile_bootstrap(self) -> None:
        sample = np.random.default_rng(3).normal(0.5, 1.0, 80)
        low, high = bootstrap_ci(sample, np.random.default_rng(1), draws=20000)
        reference = stats.bootstrap(
            (sample,),
            np.mean,
            method="percentile",
            n_resamples=20000,
            confidence_level=0.95,
            rng=np.random.default_rng(2),
        ).confidence_interval
        # Different resampling streams, so agreement is to Monte Carlo error —
        # a few tenths of a percent of the width, checked against 2 %.
        width = float(reference.high - reference.low)
        assert low == pytest.approx(float(reference.low), abs=0.02 * width)
        assert high == pytest.approx(float(reference.high), abs=0.02 * width)

    def test_it_narrows_as_the_cohort_grows(self) -> None:
        small = bootstrap_ci(np.random.default_rng(5).normal(0.5, 1.0, 20), np.random.default_rng(0))
        large = bootstrap_ci(np.random.default_rng(5).normal(0.5, 1.0, 200), np.random.default_rng(0))
        assert large[1] - large[0] < small[1] - small[0]

    def test_it_lies_within_the_range_of_the_data(self) -> None:
        values = np.array([-2.0, 0.0, 0.0, 0.0, 1.0, 7.0])
        low, high = bootstrap_ci(values, np.random.default_rng(0))
        assert values.min() <= low <= high <= values.max()

    def test_the_interval_depends_on_the_generator_state(self) -> None:
        """``run_paired.analyse`` threads one generator through the outcomes in
        order, so every stored ``ci95`` after the first depends on the order of
        ``OUTCOMES``. Reordering them would move those intervals — and nothing
        else — which is why the order is fixed."""
        first = arm(**{f"L{i}": float(i) for i in range(10)})
        second = arm(**{f"L{i}": 0.0 for i in range(10)})
        shared = np.random.default_rng(0)
        assert (
            compare("outcome", first, second, shared).ci95
            != compare("outcome", first, second, shared).ci95
        )
        assert (
            compare("outcome", first, second, np.random.default_rng(0)).ci95
            == compare("outcome", first, second, np.random.default_rng(0)).ci95
        )


class TestHolmBonferroni:
    def test_it_reproduces_the_committed_summary(self) -> None:
        """The published ``holm_p`` column, re-derived from its ``sign_p`` column."""
        stored = stored_rows()
        adjusted = holm_bonferroni([row["sign_p"] for row in stored])
        for row, value in zip(stored, adjusted):
            assert value == pytest.approx(row["holm_p"], rel=1e-12), row["outcome"]

    def test_the_smallest_p_pays_the_full_family_size(self) -> None:
        assert holm_bonferroni([0.001, 0.4, 0.5, 0.6])[0] == pytest.approx(0.004)

    def test_a_single_hypothesis_is_unchanged(self) -> None:
        assert holm_bonferroni([0.03]) == [0.03]

    def test_an_empty_family_is_empty(self) -> None:
        assert holm_bonferroni([]) == []

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

    def test_tied_p_values_are_adjusted_alike(self) -> None:
        # The sort breaks the tie arbitrarily, so the two get multipliers 3 and
        # 2 — and the running maximum is what makes them come out equal.
        assert holm_bonferroni([0.01, 0.01, 0.5]) == pytest.approx([0.03, 0.03, 0.5])

    @settings(max_examples=100, deadline=None)
    @given(p_values=st.lists(st.floats(0.0, 1.0), min_size=1, max_size=8))
    def test_it_agrees_with_statsmodels(self, p_values: list[float]) -> None:
        reference = np.asarray(multipletests(p_values, method="holm")[1], dtype=float)
        assert holm_bonferroni(p_values) == pytest.approx(reference.tolist())

    @settings(max_examples=100, deadline=None)
    @given(p_values=st.lists(st.floats(0.0, 1.0), min_size=1, max_size=8))
    def test_the_invariants_hold_for_any_family(self, p_values: list[float]) -> None:
        adjusted = holm_bonferroni(p_values)
        assert all(a >= p for a, p in zip(adjusted, p_values))
        assert all(a <= 1.0 for a in adjusted)
        # A permutation of the input permutes the output and changes nothing else.
        assert holm_bonferroni(p_values[::-1]) == pytest.approx(adjusted[::-1])


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
        assert result.sign_effect_size == 1.0

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
        assert result.sign_effect_size == -1.0
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

    def test_the_counts_partition_the_pairs(self) -> None:
        result = compare(
            "outcome",
            arm(L0=1.0, L1=0.0, L2=2.0, L3=0.0, L4=4.0),
            arm(L0=0.0, L1=0.0, L2=5.0, L3=0.0, L4=4.0),
            np.random.default_rng(0),
        )
        assert result.ties + result.discordant == result.n_pairs
        assert result.favouring_first + result.favouring_second == result.discordant
