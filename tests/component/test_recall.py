"""Finding what a learner said before.

Two strategies, built to be compared rather than argued about.
``bonus_lesson_idea.md`` closed retrieval for *lesson content* and that argument
stands — fifteen lessons keyed by concept id is a dict lookup, and an
approximate index would put a non-deterministic step in the instructional path
for nothing. This is the other case the same note names as the one that *would*
earn an index: a corpus nobody keyed, queried in the learner's own words.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import pytest

from agent_newton.core.evaluation.recall import RecallGold, load_gold, score
from agent_newton.core.recall import EmbeddedRecall, KeyedRecall, Recall, cosine
from agent_newton.core.state.schema import Utterance

GOLD = Path("tests/fixtures/gold/calculus_recall_cases.yaml")


def _said(
    text: str,
    concept_id: str,
    kind: Literal["reflection", "working", "lesson"] = "lesson",
) -> Utterance:
    return Utterance(text=text, item_id="", concept_id=concept_id, kind=kind)


class Axes:
    """An embedder with no model in it: one axis per keyword.

    Enough to test ranking, thresholds and determinism without a service, and
    without the tests depending on what a particular embedding model happens to
    think two sentences have in common.
    """

    label = "fake/axes"
    WORDS = ("gradient", "ratio", "secant")

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [
            [1.0 if word in text.lower() else 0.0 for word in self.WORDS]
            for text in texts
        ]


class TestCosine:
    def test_identical_vectors(self) -> None:
        assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_a_zero_vector_is_similar_to_nothing(self) -> None:
        # Rather than raising. An utterance with no vector is a measurement that
        # failed, and it must not take a recall down with it.
        assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestKeyedRecall:
    def test_it_returns_what_was_said_about_this_concept(self) -> None:
        corpus = [_said("what is a gradient", "limit_concept"),
                  _said("something else", "power_rule")]
        [found] = KeyedRecall().about(corpus, "limit_concept", "anything")
        assert found.text == "what is a gradient"

    def test_most_recent_first(self) -> None:
        corpus = [_said("older", "power_rule"), _said("newer", "power_rule")]
        assert [u.text for u in KeyedRecall().about(corpus, "power_rule")] == [
            "newer",
            "older",
        ]

    def test_the_query_changes_nothing(self) -> None:
        """Deliberately, or the comparison would measure two changes at once.

        This strategy *is* keying on the concept. A version that also read the
        query would be a third strategy, and neither of the two being compared.
        """
        corpus = [_said("about gradients", "power_rule")]
        keyed = KeyedRecall()
        assert keyed.about(corpus, "power_rule", "gradients") == keyed.about(
            corpus, "power_rule", "nothing whatever to do with it"
        )

    def test_it_cannot_see_across_concepts(self) -> None:
        """⚠️ The limitation the whole comparison exists to price.

        A learner who asked what a gradient was while working on limits, and
        meets gradients again under the power rule, has said something that
        bears on the new question. As far as a key is concerned they never said
        it.
        """
        corpus = [_said("wait, what is a gradient", "limit_concept")]
        assert KeyedRecall().about(corpus, "power_rule", "what is the gradient?") == ()


class TestEmbeddedRecall:
    def test_it_finds_a_remark_filed_under_another_concept(self) -> None:
        corpus = [_said("wait, what is a gradient", "limit_concept")]
        found = EmbeddedRecall(Axes()).about(corpus, "power_rule", "the gradient here")
        assert [u.text for u in found] == ["wait, what is a gradient"]

    def test_it_leaves_out_what_the_query_is_not_about(self) -> None:
        corpus = [_said("what does ratio mean", "average_rate_of_change"),
                  _said("what is a gradient", "limit_concept")]
        found = EmbeddedRecall(Axes()).about(corpus, "power_rule", "about the gradient")
        assert [u.text for u in found] == ["what is a gradient"]

    def test_no_query_returns_nothing(self) -> None:
        """Rather than falling back to the concept.

        Falling back would make this the keyed strategy under another name, at
        exactly the moment the comparison is being taken.
        """
        corpus = [_said("what is a gradient", "limit_concept")]
        assert EmbeddedRecall(Axes()).about(corpus, "limit_concept", "") == ()

    def test_nothing_similar_enough_returns_nothing(self) -> None:
        # A recaller that always fills its quota looks good on recall and hands
        # a tutor an unrelated remark, which is worse than silence because the
        # tutor will try to use it.
        corpus = [_said("what does ratio mean", "average_rate_of_change")]
        assert EmbeddedRecall(Axes(), threshold=0.5).about(
            corpus, "power_rule", "about the gradient"
        ) == ()

    def test_the_threshold_is_the_dial(self) -> None:
        corpus = [_said("gradient and ratio", "limit_concept")]
        query = "gradient"
        assert EmbeddedRecall(Axes(), threshold=0.5).about(corpus, "c", query)
        assert not EmbeddedRecall(Axes(), threshold=0.9).about(corpus, "c", query)

    def test_it_is_deterministic(self) -> None:
        """⚠️ Which is the answer to the objection recorded against retrieval.

        The note's worry was a non-deterministic step in the instructional path
        of a project whose rolls are hashes and whose model cache is keyed by
        prompt. That belongs to *approximate* search. This scans the whole
        corpus exactly, so the same corpus and query give the same ranking every
        time, and the only remaining variable is the embedding model — which
        goes in the manifest like every other model.
        """
        corpus = [_said("gradient", "a"), _said("gradient again", "b")]
        recall = EmbeddedRecall(Axes())
        first = recall.about(corpus, "c", "gradient")
        for _ in range(5):
            assert recall.about(corpus, "c", "gradient") == first

    def test_the_label_names_the_model_and_the_threshold(self) -> None:
        # Two strategies are not one result, and neither are two thresholds.
        assert EmbeddedRecall(Axes(), 0.7).label == "embedded/fake/axes@0.7"

    def test_it_satisfies_the_protocol(self) -> None:
        assert isinstance(EmbeddedRecall(Axes()), Recall)
        assert isinstance(KeyedRecall(), Recall)


class TestTheGoldSet:
    @pytest.fixture(scope="class")
    @classmethod
    def gold(cls) -> RecallGold:
        return load_gold(GOLD)

    def test_it_loads(self, gold) -> None:
        assert gold.corpus and gold.cases

    def test_returning_the_whole_corpus_scores_badly(self, gold) -> None:
        """The property that makes the set worth scoring against.

        A strategy that hands back everything gets full recall for free, so the
        corpus has to be much larger than what is relevant or precision cannot
        punish it.
        """

        class ReturnsEverything:
            label = "everything"

            def about(self, corpus, concept_id, query="", limit=3):  # noqa: ANN001
                return corpus

        report = score(gold, ReturnsEverything())
        assert report.recall == pytest.approx(1.0)
        assert report.precision < 0.15

    def test_some_case_has_nothing_to_find(self, gold) -> None:
        # Invisible to both precision and recall, and the easiest thing to get
        # wrong: filling the quota fails it while recall is untouched.
        assert any(not case.relevant for case in gold.cases)

    def test_a_renamed_utterance_fails_loudly(self, tmp_path) -> None:
        # Referential integrity at load time rather than at scoring time, so a
        # rename shrinks the set visibly instead of silently.
        stray = tmp_path / "bad.yaml"
        stray.write_text(
            "corpus:\n  - {id: a, concept_id: c, text: hello}\n"
            "cases:\n  - {id: x, concept_id: c, query: q, relevant: [b]}\n"
        )
        with pytest.raises(ValueError, match="unknown utterances"):
            load_gold(stray)


class TestScoring:
    def _gold(self) -> RecallGold:
        corpus = (_said("one", "c"), _said("two", "c"), _said("three", "c"))
        from agent_newton.core.evaluation.recall import RecallCase

        return RecallGold(
            corpus,
            ("a", "b", "d"),
            (
                RecallCase("case1", "c", "q", frozenset({"a"})),
                RecallCase("empty", "c", "q", frozenset()),
            ),
        )

    def test_precision_and_recall_are_reported_apart(self) -> None:
        """Never averaged, and that is a statement about what this is for.

        An unrelated remark handed to a tutor as context is worse than silence,
        so a strategy that finds everything with half of it noise is worse here
        than one that finds less and means it. One number would hide that.
        """
        from agent_newton.core.evaluation.recall import RecallReport

        report = RecallReport(rows=[("x", frozenset({"a", "b"}), frozenset({"a"}))])
        assert report.precision == pytest.approx(0.5)
        assert report.recall == pytest.approx(1.0)
        assert not hasattr(report, "f1")

    def test_silence_on_an_empty_case_is_counted(self) -> None:
        class SaysNothing:
            label = "nothing"

            def about(self, corpus, concept_id, query="", limit=3):  # noqa: ANN001
                return ()

        report = score(self._gold(), SaysNothing())
        assert report.returned_nothing_correctly == 1
        assert report.noise == 0

    def test_noise_is_what_a_tutor_would_try_to_use(self) -> None:
        class SaysEverything:
            label = "everything"

            def about(self, corpus, concept_id, query="", limit=3):  # noqa: ANN001
                return corpus

        report = score(self._gold(), SaysEverything())
        assert report.noise == 5  # 2 wrong on case1, 3 on the empty one
        assert report.returned_nothing_correctly == 0


class TestACohortHasNothingToRecall:
    """⚠️ Structural, not a switch, and it is the guarantee worth having.

    A simulated learner writes nothing in either prose channel, so the corpus
    any recall strategy could search is empty by construction. Not a flag that
    is off — a learner with no words.
    """

    def test_a_simulated_learner_writes_no_prose(self) -> None:
        from agent_newton.config import SimulatorConfig
        from agent_newton.core.simulator import SimulatedLearner, sample_profile
        from agent_newton.domains import registry

        domain = registry.load_domain("calculus")
        profile = sample_profile("L1", 1, domain.misconceptions, SimulatorConfig())
        learner = SimulatedLearner(profile, domain, SimulatorConfig())
        item = domain.items.for_concept("power_rule", "practice")[0]
        assert learner.show_working(item, "wrong", required=True) is None
        assert learner.reflect(item, "which part?") is None
        assert learner.discuss("power_rule", "what do you think?") is None

    def test_so_both_strategies_return_nothing(self) -> None:
        for strategy in (KeyedRecall(), EmbeddedRecall(Axes())):
            assert strategy.about((), "power_rule", "the gradient") == ()


class Finds:
    """A recall that returns the whole corpus and records what it was asked.

    For the wiring tests: whether a recalled utterance reaches the prompt, and
    how it is labelled when it does, are separate questions from whether the
    ranking finds the right one.
    """

    label = "stub/finds"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def about(self, corpus, concept_id, query="", limit=3):  # noqa: ANN001
        self.queries.append(query)
        return tuple(corpus[:limit])


class TestTheTutorUsesIt:
    """Wiring the measured strategy in, off by default."""

    def _tutor(self, recall=None):  # noqa: ANN001
        from agent_newton.config import ZPDConfig
        from agent_newton.core.agents.llm import LLMTutor

        class Replies:
            label = "fake/model"

            def __init__(self) -> None:
                self.prompts: list[str] = []

            def generate(self, prompt, schema, system):  # noqa: ANN001
                import json

                from agent_newton.llm.base import Completion

                self.prompts.append(prompt)
                return Completion(
                    text=json.dumps({"text": "look again."}),
                    model="fake",
                    provider="fake",
                )

        provider = Replies()
        return LLMTutor(provider, ZPDConfig(), recall=recall), provider

    def _view(self, *utterances: Utterance):
        from agent_newton.core.state.views import FullStateView
        from agent_newton.core.state.zpd import Frontier

        return FullStateView(
            mastery={"power_rule": 0.4},
            error_trace=(),
            frontier=Frontier(frozenset({"power_rule"})),
            outcomes=(),
            version=1,
            reflections=utterances,
        )

    def _respond(self, tutor, view, domain):  # noqa: ANN001
        from agent_newton.core.agents.base import Diagnosis

        item = domain.items.for_concept("power_rule", "practice")[0]
        return tutor.respond(
            item,
            Diagnosis(None),
            view,
            domain,
            response="5x^5",
            mastery=0.4,
            prior_failures=0,
            moves_this_item=[],
        )

    @pytest.fixture(scope="class")
    @classmethod
    def calculus(cls):
        from agent_newton.domains import registry

        return registry.load_domain("calculus")

    def test_off_by_default_it_reads_only_this_concept(self, calculus) -> None:
        # What every measured result was produced under. A remark filed under
        # another concept is invisible, which is `said_about`'s whole behaviour.
        tutor, provider = self._tutor()
        view = self._view(_said("wait, what is a gradient", "limit_concept"))
        self._respond(tutor, view, calculus)
        assert "gradient" not in provider.prompts[0]

    def test_with_recall_a_remark_from_another_concept_reaches_the_tutor(
        self, calculus
    ) -> None:
        # The case the ideas note asks for, end to end. A stub strategy, because
        # what is under test here is the wiring — whether a recalled utterance
        # reaches the prompt at all — and the ranking is tested on its own above.
        tutor, provider = self._tutor(Finds())
        view = self._view(_said("wait, what is a gradient", "limit_concept"))
        self._respond(tutor, view, calculus)
        assert "wait, what is a gradient" in provider.prompts[0]

    def test_the_query_carries_the_question_and_not_only_the_answer(
        self, calculus
    ) -> None:
        """⚠️ A response is usually an expression, and prose does not rank
        against algebra.

        The words a learner used when they were confused are in the question's
        vocabulary — "gradient", "ratio", "secant" — not in `5x^5`. Ranking
        against the answer alone matched almost nothing, which made recall look
        useless when it was being asked the wrong question.
        """
        asked = Finds()
        tutor, _ = self._tutor(asked)
        self._respond(tutor, self._view(), calculus)
        assert "Differentiate" in asked.queries[0]
        assert "5x^5" in asked.queries[0]

    def test_and_it_is_labelled_as_being_about_something_else(
        self, calculus
    ) -> None:
        """⚠️ Or it is §7i's leak one level out.

        Keyed reading could only return remarks about the current concept, so
        nothing ever had to say otherwise. Recall reaches across, and an
        unlabelled remark about limits arriving while the learner works the
        power rule is a tutor being handed context it will misuse — which is how
        a learner came to be asked about their derivative of u^4 on a question
        containing no u^4.
        """
        tutor, provider = self._tutor(Finds())
        view = self._view(_said("wait, what is a gradient", "limit_concept"))
        self._respond(tutor, view, calculus)
        assert "while working on limit_concept" in provider.prompts[0]

    def test_nothing_relevant_hands_the_tutor_nothing(self, calculus) -> None:
        # Silence rather than the least irrelevant thing available. Real
        # strategy here, because "returns nothing" is the behaviour under test.
        tutor, provider = self._tutor(EmbeddedRecall(Axes(), threshold=0.5))
        view = self._view(_said("what does ratio mean", "average_rate_of_change"))
        self._respond(tutor, view, calculus)
        assert "ratio" not in provider.prompts[0]


class TestRecallCannotCrossTheArms:
    """⚠️ The property the whole comparison rests on, and the one this could break.

    The learner's words live on ``FullStateView`` and nowhere else — it is
    something they told us about themselves, and the decoupled arm is *defined*
    by doing without it. Recall reads the history from inside that branch. Had it
    reached the store or the session instead, the decoupled arm would have been
    handed what it is defined by lacking, and it would have looked like the
    coupling advantage growing rather than like a leak.
    """

    def test_the_decoupled_view_carries_no_words_to_recall_from(self) -> None:
        from agent_newton.core.state.views import ItemCorrectnessView

        view = ItemCorrectnessView(outcomes=(True,), version=1)
        assert not hasattr(view, "reflections")
        assert not hasattr(view, "said_about")

    def test_a_decoupled_tutor_reads_nothing_however_recall_is_set(self) -> None:
        from agent_newton.config import ZPDConfig
        from agent_newton.core.agents.base import Diagnosis
        from agent_newton.core.agents.llm import LLMTutor
        from agent_newton.core.state.views import ItemCorrectnessView
        from agent_newton.domains import registry

        class Replies:
            label = "fake/model"

            def __init__(self) -> None:
                self.prompts: list[str] = []

            def generate(self, prompt, schema, system):  # noqa: ANN001
                import json

                from agent_newton.llm.base import Completion

                self.prompts.append(prompt)
                return Completion(
                    text=json.dumps({"text": "look again."}), model="f", provider="f"
                )

        class Everything:
            label = "everything"

            def about(self, corpus, concept_id, query="", limit=3):  # noqa: ANN001
                raise AssertionError(
                    "recall must not be consulted for the decoupled arm"
                )

        domain = registry.load_domain("calculus")
        provider = Replies()
        tutor = LLMTutor(provider, ZPDConfig(), recall=Everything())
        tutor.respond(
            domain.items.for_concept("power_rule", "practice")[0],
            Diagnosis(None),
            ItemCorrectnessView(outcomes=(), version=1),
            domain,
            response="5x^5",
            mastery=0.0,
            prior_failures=0,
            moves_this_item=[],
        )
        assert provider.prompts, "the tutor still answered"


class TestItIsOffForEveryCohort:
    def test_the_default_is_off(self) -> None:
        from agent_newton.config import Config

        assert Config().teaching.recall.strategy == "off"

    def test_a_run_that_does_not_want_it_builds_nothing(self) -> None:
        # No embedding connection opened and nothing embedded, the same way the
        # confusion detector is built only where it is asked for.
        from agent_newton.config import Config
        from agent_newton.core.orchestration.session import _recall_for

        assert _recall_for(Config()) is None

    def test_the_measured_default_is_what_gets_built(self) -> None:
        from agent_newton.config import Config
        from agent_newton.core.orchestration.session import _recall_for

        config = Config()
        config.teaching.recall.strategy = "embedded"
        built = _recall_for(config)
        assert built is not None
        # 0.7 from the sweep, chosen for precision: 80% precision at 36% recall,
        # and correctly silent when nothing is relevant.
        assert "@0.7" in built.label


class TestADeadEmbedderDoesNotEndTheSitting:
    """A sitting must survive a dead backend, and recall is not the exception.

    ⚠️ The container found this rather than a test. `demo.yaml` sets
    `recall.strategy: embedded`, `./newton up-models` pulled only the chat
    model, and the reachability check looked at only that one — so a sitting
    started, ran until the first tutor reply, and then raised out of
    `_remembered` from *outside* the `try` that catches a dead chat model.

    Falling back costs context and nothing else: `said_about` is the pre-recall
    behaviour and what every measured result was produced under.
    """

    class Dead:
        """A strategy whose backend is not there."""

        label = "ollama/nomic-embed-text@0.7"

        def about(self, utterances, concept_id, query, limit=3):  # noqa: ANN001, ARG002
            from agent_newton.llm.base import ProviderError

            raise ProviderError("ollama embed failed for nomic-embed-text")

    def test_the_turn_is_still_written(self, calculus) -> None:
        helper = TestTheTutorUsesIt()
        tutor, _provider = helper._tutor(self.Dead())
        view = helper._view(_said("i keep dropping the exponent", "power_rule"))
        hint = helper._respond(tutor, view, calculus)
        assert hint.text

    def test_it_falls_back_to_said_about(self, calculus) -> None:
        # Not merely "it did not raise": the words filed under *this* concept
        # still reach the prompt, which is what makes the fallback a degradation
        # rather than a silent loss of the channel.
        helper = TestTheTutorUsesIt()
        tutor, provider = helper._tutor(self.Dead())
        view = helper._view(_said("i keep dropping the exponent", "power_rule"))
        helper._respond(tutor, view, calculus)
        assert "dropping the exponent" in provider.prompts[0]

    def test_the_failure_is_counted(self, calculus) -> None:
        # ⚠️ The point of the counter. A run that quietly stopped recalling and
        # one where nothing happened to match are indistinguishable without it,
        # and the first is a broken setup while the second is a normal sitting.
        helper = TestTheTutorUsesIt()
        tutor, _provider = helper._tutor(self.Dead())
        view = helper._view(_said("i keep dropping the exponent", "power_rule"))
        assert tutor.recall_failures == 0
        helper._respond(tutor, view, calculus)
        assert tutor.recall_failures == 1

    @pytest.fixture(scope="class")
    @classmethod
    def calculus(cls):
        from agent_newton.domains import registry

        return registry.load_domain("calculus")
