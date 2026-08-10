"""Paired statistics for the cohort comparison.

The comparison is paired by construction — the same seed produces the same
learner in both arms — so every learner is their own control and the unit of
analysis is the *difference*.

What those differences look like decides the test, and here they are not a bell
curve. Most learners record exactly zero: their misconceptions were either
reached and remediated under both architectures or under neither, and the effect
lives in a minority of discordant pairs. A paired t-test would be assuming a
shape the data contradicts.

So the primary test is the **exact sign test** on discordant pairs. It uses only
the direction of each difference, which is all a tie-heavy paired sample
supports, and it is exact rather than asymptotic — which matters when the
discordant count is small. Wilcoxon's signed-rank is reported beside it: it uses
the magnitudes too, so it has more power when they are informative, at the cost
of treating differences as comparable across learners.

Effect size is the **rank-biserial correlation**, the share of discordant pairs
favouring one arm rescaled to [-1, 1]. It does not pretend the differences sit
on an interval scale, which Cohen's d would.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

#: Conventional two-sided threshold, fixed here so no analysis picks its own.
ALPHA = 0.05


@dataclass(frozen=True, slots=True)
class PairedResult:
    """One outcome, compared across arms."""

    outcome: str
    n_pairs: int
    #: Pairs where the two arms differed at all. The sign test's real sample
    #: size, and usually far below ``n_pairs``.
    discordant: int
    favouring_first: int
    favouring_second: int
    mean_difference: float
    ci95: tuple[float, float]
    sign_p: float
    wilcoxon_p: float
    rank_biserial: float

    @property
    def significant(self) -> bool:
        """On the primary test, which is the sign test."""
        return self.sign_p < ALPHA

    @property
    def ties(self) -> int:
        return self.n_pairs - self.discordant


def sign_test(differences: np.ndarray) -> float:
    """Two-sided exact sign test. Ties are dropped, as the test requires."""
    discordant = differences[differences != 0]
    if discordant.size == 0:
        return 1.0
    positive = int((discordant > 0).sum())
    result = stats.binomtest(positive, discordant.size, 0.5)
    return float(result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def wilcoxon(differences: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank. 1.0 when every pair is tied."""
    if not np.any(differences != 0):
        return 1.0
    try:
        outcome = stats.wilcoxon(differences, zero_method="wilcox")
        return float(outcome.pvalue)  # pyright: ignore[reportAttributeAccessIssue]
    except ValueError:  # pragma: no cover - guarded by the check above
        return 1.0


def rank_biserial(differences: np.ndarray) -> float:
    """Share of discordant pairs favouring the first arm, rescaled to [-1, 1]."""
    discordant = differences[differences != 0]
    if discordant.size == 0:
        return 0.0
    return float(2 * (discordant > 0).mean() - 1)


def bootstrap_ci(
    differences: np.ndarray, rng: np.random.Generator, draws: int = 2000
) -> tuple[float, float]:
    """Percentile confidence interval for the mean paired difference.

    Bootstrapped rather than taken from a t distribution, for the same reason
    the test is non-parametric: the differences are not normal.
    """
    if differences.size == 0:
        return (0.0, 0.0)
    picks = rng.integers(0, differences.size, size=(draws, differences.size))
    means = differences[picks].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_differences(
    first: dict[str, dict], second: dict[str, dict], key: str
) -> np.ndarray:
    """``first - second`` per learner, matched by id.

    Matching by id rather than position means a change to cohort ordering cannot
    silently pair the wrong learners, which would look like noise rather than
    like a bug.
    """
    shared = sorted(set(first) & set(second))
    if len(shared) != len(first) or len(shared) != len(second):
        raise ValueError("the two arms did not run the same learners")
    return np.array(
        [(first[k][key] or 0) - (second[k][key] or 0) for k in shared], dtype=float
    )


def compare(
    outcome: str,
    first: dict[str, dict],
    second: dict[str, dict],
    rng: np.random.Generator,
    draws: int = 2000,
) -> PairedResult:
    """Full paired comparison of one outcome between two arms."""
    differences = paired_differences(first, second, outcome)
    return PairedResult(
        outcome=outcome,
        n_pairs=int(differences.size),
        discordant=int((differences != 0).sum()),
        favouring_first=int((differences > 0).sum()),
        favouring_second=int((differences < 0).sum()),
        mean_difference=float(differences.mean()) if differences.size else 0.0,
        ci95=bootstrap_ci(differences, rng, draws),
        sign_p=sign_test(differences),
        wilcoxon_p=wilcoxon(differences),
        rank_biserial=rank_biserial(differences),
    )


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values, in the order given.

    Several outcomes are reported from one cohort, so the family-wise error rate
    needs controlling or the secondary outcomes inflate the chance that
    *something* reaches significance. Holm rather than plain Bonferroni: it is
    uniformly more powerful and rests on no extra assumption.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        scaled = (len(p_values) - rank) * p_values[index]
        running = max(running, min(1.0, scaled))
        adjusted[index] = running
    return adjusted
