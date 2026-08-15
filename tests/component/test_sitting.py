"""Reading a sitting back.

Every figure quoted about the human sittings came out of a script written for
the occasion, because the record is tens of kilobytes of JSON. What made that
worse than tedious is that the record could not answer the question the sittings
kept raising — *why* was the support pitched there — so the answer had to be
reconstructed by wrapping the tutor and running it again.

So the log carries the level's own inputs and the learner's own answer, and one
renderer turns it into prose. The renderer is shared with the scaffold probe, so
a replay and a real sitting read identically.
"""

from __future__ import annotations

import pytest

from agent_newton.core.evaluation.sitting import narrate, summarise
from agent_newton.domains import registry


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


def record(cause: str, **evidence) -> dict:
    return {"cause": cause, "summary": f"{cause} happened", "evidence": evidence}


def sitting() -> list[dict]:
    return [
        record("annotation", requested=["distribute"]),
        record("plan", goal="solve_linear"),
        record(
            "annotation",
            item_id="ta_dist_p1",
            concept_id="distribute",
            reflection="I multiplied only the first term",
            kind="working",
        ),
        record(
            "observation",
            item_id="ta_dist_p1",
            concept_id="distribute",
            verdict="incorrect",
            response="3x + 4",
            mastery_before=0.40,
            mastery_after=0.26,
            misconception_label="distribute_first_term_only",
        ),
        record(
            "tutor",
            item_id="ta_dist_p1",
            concept_id="distribute",
            move="remediate",
            level="targeted",
            text="Here is what went wrong: the factor reaches every term.",
            mastery=0.40,
            prior_failures=0,
        ),
    ]


class TestTheAccountIsReadable:
    def test_it_shows_what_the_learner_wrote(self, toy) -> None:
        # The thing a transcript is for, and the log did not carry it: a verdict
        # without the answer says a step was wrong and not what was wrong.
        text = narrate(sitting(), toy, learner_id="victor")
        assert "`3x + 4`" in text
        assert "incorrect" in text

    def test_it_says_what_the_level_was_chosen_from(self, toy) -> None:
        # The question two sittings could not answer from their own record.
        text = narrate(sitting(), toy, learner_id="victor")
        assert "belief of 0.40" in text
        assert "0 earlier readable failure(s)" in text

    def test_an_older_sitting_says_the_inputs_are_missing(self, toy) -> None:
        # Rather than defaulting them: a belief of zero is a claim, and every
        # sitting before this week has none recorded.
        older = [r for r in sitting() if r["cause"] != "tutor"]
        older.append(
            record(
                "tutor", item_id="ta_dist_p1", concept_id="distribute",
                move="hint", level="worked_step", text="Look again.",
            )
        )
        assert "predates" in narrate(older, toy)

    def test_the_learner_s_own_words_are_quoted(self, toy) -> None:
        assert "I multiplied only the first term" in narrate(sitting(), toy)

    def test_concepts_are_named_not_keyed(self, toy) -> None:
        # A reader should not have to know the ids. The panel that quoted two
        # remarks without naming their subject was read as noise.
        text = narrate(sitting(), toy, learner_id="victor")
        assert "Distributing a factor over a sum" in text

    def test_the_request_is_in_the_account(self, toy) -> None:
        assert "Asked to work on" in narrate(sitting(), toy)

    def test_it_survives_a_concept_the_graph_no_longer_has(self, toy) -> None:
        # Content moves; a stored sitting must still render. Falling over while
        # reading a record is how a record stops being read.
        stale = [
            record(
                "observation", item_id="gone", concept_id="no_such_concept",
                verdict="correct", response="7",
            )
        ]
        assert "no_such_concept" in narrate(stale, toy)


class TestTheSummary:
    def test_it_counts_the_levels(self, toy) -> None:
        # The count that would have shown both collapses without anyone reading
        # a transcript: every turn at one level.
        assert summarise(sitting())["levels"] == {"targeted": 1}

    def test_it_counts_the_verdicts(self, toy) -> None:
        assert summarise(sitting())["verdicts"] == {"incorrect": 1}

    def test_an_empty_log_summarises_to_nothing(self, toy) -> None:
        assert summarise([]) == {"levels": {}, "moves": {}, "verdicts": {}}
