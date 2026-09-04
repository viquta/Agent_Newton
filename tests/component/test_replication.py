"""The seed guard, and the replication's pre-registered shape.

Learner profiles come from ``(seed, learner_id)``, so re-using a seed a stored
summary already reports means re-reporting learners some existing analysis has
already read a result off. ``run_paired`` refused only the *config's* seed — the
one the power analysis sized from — which left every other spent seed reachable,
the confirmatory seed included.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from replicate_paired import (  # noqa: E402
    BASELINE_SEED,
    PRIMARY,
    RECORDED,
    SEEDS,
    distribution,
)
from run_paired import OUTCOMES, spent_seeds  # noqa: E402


def _summary(directory: Path, name: str, payload: dict) -> None:
    (directory / name).mkdir(parents=True, exist_ok=True)
    (directory / name / "summary.json").write_text(json.dumps(payload))


class TestTheSeedGuardCanFail:
    """A guard that cannot fail proves nothing."""

    def test_a_seed_a_stored_summary_reports_is_found(self, tmp_path: Path) -> None:
        _summary(tmp_path, "some_study", {"seed": 12345})
        assert 12345 in spent_seeds(tmp_path)

    def test_it_says_which_study_spent_it(self, tmp_path: Path) -> None:
        # The error message names the study, so the person picking a new seed
        # can see what would have been double-counted.
        _summary(tmp_path, "paired_calculus", {"seed": 12345})
        assert spent_seeds(tmp_path)[12345] == {"paired_calculus"}

    def test_an_unused_seed_is_not_refused(self, tmp_path: Path) -> None:
        _summary(tmp_path, "some_study", {"seed": 12345})
        assert 999 not in spent_seeds(tmp_path)

    def test_the_pilot_and_baseline_fields_count_too(self, tmp_path: Path) -> None:
        # A sweep records its baseline seed under its own name; a seed spent as
        # someone else's baseline is just as spent.
        _summary(tmp_path, "a_sweep", {"baseline_seed": 222, "pilot_seed": 333})
        found = spent_seeds(tmp_path)
        assert 222 in found and 333 in found

    def test_reruns_do_not_count(self, tmp_path: Path) -> None:
        # `results/reruns/` is where the container's reproduce helper writes.
        # Those are re-runs of studies already counted, so treating them as
        # spending a seed would refuse the seed a study legitimately owns.
        _summary(tmp_path / "reruns", "paired_calculus", {"seed": 4444})
        assert 4444 not in spent_seeds(tmp_path)

    def test_a_malformed_summary_is_skipped_rather_than_fatal(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "broken").mkdir()
        (tmp_path / "broken" / "summary.json").write_text("{not json")
        _summary(tmp_path, "fine", {"seed": 77})
        assert spent_seeds(tmp_path) == {77: {"fine"}}


class TestTheReplicationIsPreRegistered:
    """The commitments that make this a replication rather than another attempt.

    All three are properties of the file, so they can be checked rather than
    trusted: fixed seeds, a baseline that must reproduce, and a summary quantity
    that is the estimate rather than a count of significant runs.
    """

    def test_the_seeds_are_fixed_and_owned_by_nothing_else(self) -> None:
        """No other study may have read a result off these seeds.

        ⚠️ Not "unspent" — once the replication has run, its own summary reports
        every one of them, and that is correct. An earlier version of this test
        asserted the precondition rather than the property, so running the thing
        it guards made it fail. What must hold is that *no other study* owns
        them.
        """
        assert len(SEEDS) == len(set(SEEDS)) == 10
        spent = spent_seeds(Path("results"))
        others = {
            seed: sorted(studies - {"replication_paired"})
            for seed in SEEDS
            for studies in [spent.get(seed, set())]
            if studies - {"replication_paired"}
        }
        assert not others, f"a declared seed is owned by another study: {others}"

    def test_a_study_reporting_many_seeds_spends_all_of_them(
        self, tmp_path: Path
    ) -> None:
        # The plural field. Reading only the singular one would leave a
        # multi-seed study's seeds silently reusable.
        _summary(tmp_path, "replication_paired", {"seeds": [1, 2, 3], "seed": 9})
        found = spent_seeds(tmp_path)
        assert {1, 2, 3, 9} <= set(found)

    def test_run_directories_do_not_spend_a_seed(self, tmp_path: Path) -> None:
        # A study's summary records that a result was read off a seed. The
        # timestamped run directories beside it are byproducts — a reproduction
        # pass writes hundreds — and counting them would refuse a seed for
        # having been re-run rather than reported on.
        _summary(tmp_path, "20260831T100502_smoke_coupled_53c98352", {"seed": 555})
        assert 555 not in spent_seeds(tmp_path)

    def test_the_baseline_is_the_confirmatory_seed(self) -> None:
        # Run first and required to reproduce, so a harness that measured
        # something else could not report a beautifully consistent wrong answer.
        assert BASELINE_SEED == 20260811
        assert set(RECORDED) == set(OUTCOMES)

    def test_the_primary_is_the_declared_one(self) -> None:
        assert PRIMARY == OUTCOMES[0] == "remediation"

    def test_the_summary_quantity_is_the_estimate_not_a_win_count(self) -> None:
        # ⚠️ The pre-registered figure is the median and spread. The count of
        # seeds clearing correction is reported because a reader will want it,
        # and must never become the headline: at this tie rate significance is
        # close to a lottery, which is the thing being characterised.
        points = [
            {"outcomes": {"remediation": {"mean_difference": v, "significant": s}}}
            for v, s in [(0.01, True), (0.03, False), (0.02, True)]
        ]
        summary = distribution(points, "remediation")
        assert summary["median"] == pytest.approx(0.02)
        assert summary["min"] == pytest.approx(0.01)
        assert summary["max"] == pytest.approx(0.03)
        assert summary["values"] == sorted(summary["values"])
        assert summary["seeds_clearing_correction"] == 2

    def test_every_declared_seed_reaches_the_summary(self) -> None:
        # The failure this design exists to prevent is a dropped seed, so the
        # count is asserted rather than eyeballed.
        points = [
            {"outcomes": {"remediation": {"mean_difference": 0.0, "significant": False}}}
            for _ in SEEDS
        ]
        assert distribution(points, "remediation")["n_seeds"] == len(SEEDS)
