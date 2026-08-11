"""Model-backed agents.

Driven by a scripted fake provider, so the wiring, the guardrails and the
failure handling are tested without inference. Whether a real model gives good
answers is a measurement, not a test — that is what the component evaluation is
for.
"""

from __future__ import annotations

import json

import pytest

from agent_newton.config import BKTConfig, Config, ZPDConfig
from agent_newton.core.agents.base import Diagnosis, OracleAccess
from agent_newton.core.agents.llm import LLMDiagnostic, LLMPlanner, LLMTutor
from agent_newton.core.agents.schemas import (
    MAX_CANDIDATES,
    UNKNOWN,
    diagnosis_schema,
    plan_schema,
)
from agent_newton.core.orchestration.session import build_session
from agent_newton.core.pedagogy import HintLevel, TutorMove
from agent_newton.core.state import bkt
from agent_newton.core.state.schema import Plan
from agent_newton.core.state.store import new_blackboard
from agent_newton.domains import registry
from agent_newton.llm.base import Completion

BAND = ZPDConfig()
PRIOR = bkt.initial(BKTConfig())


class Scripted:
    """Replies from a list; repeats the last one once exhausted."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies) or ["{}"]
        self.calls: list[str] = []
        self.systems: list[str] = []

    @property
    def label(self) -> str:
        return "fake/model-1"

    def generate(self, prompt: str, schema, system):  # noqa: ANN001
        self.calls.append(prompt)
        self.systems.append(system or "")
        text = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        return Completion(text=text, model="model-1", provider="fake")


@pytest.fixture(scope="module")
def toy():
    return registry.load_domain("toy_algebra")


def view_for(toy, arm: str = "coupled", goal: str | None = "solve_linear"):
    config = Config.model_validate({"domain": "toy_algebra", "arm": arm})
    board = new_blackboard("L1", 1, toy.concepts, config)
    if goal is not None:
        # The session sets the goal before asking for an item, so a view without
        # one is not a state any planner is ever handed.
        board.record_plan(Plan(goal=goal))
    return board.view()


class TestTheDiagnosticCannotSeeGroundTruth:
    """The circularity control, enforced by structure rather than discipline."""

    def test_it_is_not_an_oracle(self) -> None:
        # If it satisfied OracleAccess the session would hand it the injected
        # label, and the accuracy it reports would be meaningless.
        assert not isinstance(LLMDiagnostic(Scripted()), OracleAccess)

    def test_it_has_no_channel_to_the_label(self) -> None:
        agent = LLMDiagnostic(Scripted())
        assert not hasattr(agent, "observe_ground_truth")

    def test_the_prompt_never_contains_the_answer_label(self, toy) -> None:
        # The strongest form: even the text sent to the model must not carry it.
        provider = Scripted(json.dumps({"misconception_id": "unknown", "confidence": 0.1}))
        item = toy.items.get("ta_dist_p1")  # probes distribute_first_term_only
        LLMDiagnostic(provider).diagnose(item, "3x + 4", toy)
        prompt = provider.calls[0]
        assert "3x + 4" in prompt, "the student's step should be shown"
        # The catalogue is offered as options, but nothing marks which is right.
        assert "the answer is distribute_first_term_only" not in prompt.lower()


class TestLLMDiagnostic:
    def test_returns_the_chosen_label(self, toy) -> None:
        reply = json.dumps({"misconception_id": "combine_unlike_terms", "confidence": 0.9})
        result = LLMDiagnostic(Scripted(reply)).diagnose(
            toy.items.get("ta_clt_p2"), "9x", toy
        )
        assert result.misconception_id == "combine_unlike_terms"
        assert result.confidence == pytest.approx(0.9)

    def test_unknown_is_no_diagnosis_rather_than_a_label(self, toy) -> None:
        # A forced guess is worse than an admission, for tutoring and for
        # measurement alike.
        reply = json.dumps({"misconception_id": "unknown", "confidence": 0.2})
        result = LLMDiagnostic(Scripted(reply)).diagnose(
            toy.items.get("ta_clt_p2"), "9x", toy
        )
        assert result.misconception_id is None
        assert not result.named

    def test_an_unusable_reply_is_not_a_wrong_answer(self, toy) -> None:
        # Nothing was inferred, so nothing may be scored. Counted separately so
        # the failure rate stays visible.
        agent = LLMDiagnostic(Scripted("gibberish"))
        result = agent.diagnose(toy.items.get("ta_clt_p2"), "9x", toy)
        assert result.misconception_id is None
        assert agent.failures == 1

    def test_the_schema_forbids_an_invented_label(self, toy) -> None:
        schema = diagnosis_schema(toy.name, tuple(toy.misconceptions.ids()))
        with pytest.raises(Exception):
            schema.model_validate({"misconception_id": "not_a_real_id", "confidence": 0.5})


class TestTheCandidateListIsBounded:
    """An unbounded array is an unbounded reply.

    Constrained decoding follows the schema's grammar, so an array with no
    maximum permits a model to enumerate labels until it hits the context
    limit — minutes of inference for a reply that should take one second.
    """

    def test_too_many_candidates_are_rejected(self, toy) -> None:
        schema = diagnosis_schema(toy.name, tuple(toy.misconceptions.ids()))
        label = toy.misconceptions.ids()[0]
        with pytest.raises(Exception):
            schema.model_validate(
                {
                    "considered": [label] * (MAX_CANDIDATES + 1),
                    "misconception_id": label,
                    "confidence": 0.5,
                }
            )

    def test_the_bound_reaches_the_decoder(self, toy) -> None:
        # The JSON schema is what becomes the decoding grammar. A bound enforced
        # only by pydantic would reject the long reply *after* generating it,
        # which is the cost this exists to avoid.
        schema = diagnosis_schema(toy.name, tuple(toy.misconceptions.ids()))
        rendered = schema.model_json_schema()
        assert rendered["properties"]["considered"]["maxItems"] == MAX_CANDIDATES

    def test_a_reply_within_the_bound_still_passes(self, toy) -> None:
        schema = diagnosis_schema(toy.name, tuple(toy.misconceptions.ids()))
        label = toy.misconceptions.ids()[0]
        reply = schema.model_validate(
            {"considered": [label], "misconception_id": label, "confidence": 0.5}
        )
        assert reply.misconception_id == label  # pyright: ignore[reportAttributeAccessIssue]


class TestTheLabelSpaceOnOffer:
    """Which misconceptions a diagnosis may choose between.

    With the whole catalogue on offer the agent names a misconception from an
    unrelated concept rather than abstaining — three of them in one human
    sitting, each shown to the learner as an explanation of their error.
    """

    def _diagnose(self, toy, label_space, label):
        provider = Scripted(
            json.dumps({"misconception_id": label, "confidence": 0.9})
        )
        item = next(i for i in toy.items.bank("practice") if i.probes)
        LLMDiagnostic(provider, label_space=label_space).diagnose(item, "3*x", toy)
        return item, provider

    def test_the_narrow_space_offers_only_the_item_s_concept(self, toy) -> None:
        item, provider = self._diagnose(toy, "concept", item_label(toy))
        offered = {m.id for m in toy.misconceptions.for_concept(item.concept_id)}
        elsewhere = {m.id for m in toy.misconceptions.all()} - offered

        assert offered, "the fixture item's concept has no misconceptions"
        assert all(label in provider.calls[0] for label in offered)
        assert not any(label in provider.calls[0] for label in elsewhere)

    def test_the_wide_space_offers_everything(self, toy) -> None:
        _, provider = self._diagnose(toy, "catalogue", item_label(toy))
        assert all(m.id in provider.calls[0] for m in toy.misconceptions.all())

    def test_a_label_from_another_concept_is_not_even_expressible(self, toy) -> None:
        # The schema becomes the decoding grammar, so under the narrow space an
        # incoherent label is impossible rather than merely discouraged.
        item = next(i for i in toy.items.bank("practice") if i.probes)
        own = tuple(m.id for m in toy.misconceptions.for_concept(item.concept_id))
        schema = diagnosis_schema(toy.name, own)
        rendered = schema.model_json_schema()

        allowed = set(rendered["properties"]["misconception_id"]["enum"])
        assert allowed == set(own) | {UNKNOWN}

    def test_a_concept_with_no_entry_falls_back_to_the_catalogue(self, toy) -> None:
        # Offering nothing would leave 'unknown' the only legal reply, which is
        # not a measurement. toy_algebra has such a concept.
        bare = next(
            i
            for i in toy.items.bank("practice")
            if not toy.misconceptions.for_concept(i.concept_id)
        )
        provider = Scripted(json.dumps({"misconception_id": UNKNOWN, "confidence": 0.1}))
        LLMDiagnostic(provider, label_space="concept").diagnose(bare, "3*x", toy)
        assert all(m.id in provider.calls[0] for m in toy.misconceptions.all())


def item_label(toy) -> str:
    item = next(i for i in toy.items.bank("practice") if i.probes)
    return item.probes[0]


class TestLLMTutor:
    def _hint(self, toy, diagnosis, moves=(), attempts=1, response="3*x + 4"):
        provider = Scripted(json.dumps({"text": "Look at the second term."}))
        tutor = LLMTutor(provider, BAND)
        hint = tutor.respond(
            toy.items.get("ta_dist_p1"),
            diagnosis,
            view_for(toy),
            toy,
            response=response,
            failed_attempts=attempts,
            moves_this_item=list(moves),
        )
        return hint, provider

    def test_uses_the_model_for_wording(self, toy) -> None:
        hint, _ = self._hint(toy, Diagnosis("distribute_first_term_only", 0.9))
        assert hint.text == "Look at the second term."

    def test_the_rules_choose_the_move_not_the_model(self, toy) -> None:
        # First turn after a confirmed misconception must be reflection, whatever
        # the model would rather say.
        hint, _ = self._hint(toy, Diagnosis("distribute_first_term_only", 0.9))
        assert hint.move is TutorMove.REFLECT
        assert hint.targets is None

    def test_remediation_follows_reflection(self, toy) -> None:
        hint, _ = self._hint(
            toy, Diagnosis("distribute_first_term_only", 0.9), moves=[TutorMove.REFLECT]
        )
        assert hint.move is TutorMove.REMEDIATE
        assert hint.targets == "distribute_first_term_only"

    def test_the_rules_choose_the_support_level(self, toy) -> None:
        # Level comes from mastery and failures, not from the prompt. A
        # constraint a model can talk itself out of is not a constraint.
        hint, provider = self._hint(toy, Diagnosis(None), attempts=0)
        assert hint.level is HintLevel.WORKED_STEP
        assert "work the step through" in provider.calls[0]

    def test_a_failed_call_still_produces_a_turn(self, toy) -> None:
        # A hint is prose; losing it should not end a session. What matters is
        # that the targeting survives.
        provider = Scripted("not json")
        hint = LLMTutor(provider, BAND).respond(
            toy.items.get("ta_dist_p1"),
            Diagnosis("distribute_first_term_only", 0.9),
            view_for(toy),
            toy,
            response="3*x + 4",
            failed_attempts=1,
            moves_this_item=[TutorMove.REFLECT],
        )
        assert hint.text
        assert hint.targets == "distribute_first_term_only"

    def test_the_step_being_responded_to_reaches_the_model(self, toy) -> None:
        # Without this the tutor has only the misconception's description and
        # reconstructs a step to match it. A human session was told its
        # calculation was correct when it had in fact multiplied.
        _, provider = self._hint(
            toy, Diagnosis("distribute_first_term_only", 0.9), response="3*x + 4"
        )
        assert "3*x + 4" in provider.calls[0]

    def test_the_model_is_told_not_to_narrate_the_working(self, toy) -> None:
        # The response alone is not enough: a model handed a step will still
        # assert which part of it the student got right. The instruction is
        # what stops that, so it has to be present.
        provider = Scripted(json.dumps({"text": "Look again."}))
        LLMTutor(provider, BAND).respond(
            toy.items.get("ta_dist_p1"),
            Diagnosis(None),
            view_for(toy),
            toy,
            response="3*x + 4",
            failed_attempts=1,
            moves_this_item=[],
        )
        assert "unless their step shows it" in provider.systems[0]


class TestLLMPlannerGuardrails:
    def _planner(self, *replies):
        return LLMPlanner(Scripted(*replies), BAND, PRIOR)

    def test_an_in_band_proposal_is_honoured(self, toy) -> None:
        reply = json.dumps({"concept_id": "integer_arithmetic", "reason": "start here"})
        planner = self._planner(reply)
        item = planner.select(view_for(toy), toy, {})
        assert item is not None and item.concept_id == "integer_arithmetic"
        assert planner.overrides == 0

    def test_an_out_of_band_proposal_is_overridden(self, toy) -> None:
        # solve_linear sits behind unmet prerequisites at the start.
        reply = json.dumps({"concept_id": "solve_linear", "reason": "looks fun"})
        planner = self._planner(reply)
        item = planner.select(view_for(toy), toy, {})
        assert item is not None and item.concept_id != "solve_linear"
        assert planner.overrides == 1

    def test_a_failed_call_falls_back_rather_than_stalling(self, toy) -> None:
        planner = self._planner("not json")
        assert planner.select(view_for(toy), toy, {}) is not None
        assert planner.overrides == 1

    def test_the_override_rate_is_reported(self, toy) -> None:
        # A high rate means the model is not usefully planning, whatever the
        # outcome numbers say.
        reply = json.dumps({"concept_id": "solve_linear", "reason": "no"})
        planner = self._planner(reply)
        for _ in range(4):
            planner.select(view_for(toy), toy, {})
        assert planner.override_rate == pytest.approx(1.0)

    def test_it_refuses_the_decoupled_view(self, toy) -> None:
        with pytest.raises(TypeError):
            self._planner("{}").select(view_for(toy, "decoupled"), toy, {})


class TestSessionAssembly:
    def _config(self, **agents) -> Config:
        base = {
            "tutor": {"impl": "template"},
            "diagnostic": {"impl": "oracle"},
            "planner": {"impl": "goal_directed"},
        }
        return Config.model_validate(
            {
                "domain": "toy_algebra",
                "simulator": {"surface": "symbolic"},
                "agents": {**base, **agents},
            }
        )

    def test_model_backed_roles_are_now_built(self, toy) -> None:
        config = self._config(
            tutor={"impl": "llm", "provider": "ollama", "model": "gemma4:12b"},
            diagnostic={"impl": "llm", "provider": "ollama", "model": "gemma4:12b"},
            planner={"impl": "llm", "provider": "ollama", "model": "gemma4:12b"},
        )
        session = build_session("L0000", 1, toy, config)
        assert isinstance(session.tutor, LLMTutor)
        assert isinstance(session.diagnostic, LLMDiagnostic)
        assert isinstance(session.planner, LLMPlanner)

    def test_building_a_provider_does_not_contact_it(self, toy) -> None:
        # Assembly must be free; only running a session should need a server.
        config = self._config(
            diagnostic={"impl": "llm", "provider": "ollama", "model": "gemma4:12b"}
        )
        build_session("L0000", 1, toy, config)

    def test_the_decoupled_arm_ignores_the_llm_planner(self, toy) -> None:
        # Its view has no frontier, so a frontier-based planner cannot run
        # whatever the config says.
        from agent_newton.core.agents.planner import FixedOrderPlanner

        config = self._config(planner={"impl": "llm", "provider": "ollama", "model": "m"})
        config = config.model_copy(update={"arm": "decoupled"})
        assert isinstance(build_session("L0000", 1, toy, config).planner, FixedOrderPlanner)
