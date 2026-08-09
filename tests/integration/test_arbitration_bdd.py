"""Step definitions for `features/arbitration.feature`.

The feature file is the specification; these bind it to the policy. Keeping the
spec executable is what stops it drifting from the code it describes — a
document can go stale silently, a failing scenario cannot.
"""

from __future__ import annotations

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from agent_newton.config import ArbitrationConfig
from agent_newton.core.arbitration.policy import ArbitrationPolicy, Decision
from agent_newton.core.state.schema import ErrorEvent
from agent_newton.core.state.zpd import Frontier

scenarios("features/arbitration.feature")

PRIOR = 0.15


@pytest.fixture
def world() -> dict:
    return {
        "concept": None,
        "in_frontier": True,
        "baseline": PRIOR,
        "mastery": PRIOR,
        "errors": [],
        "items_since": 99,  # generous unless a scenario says otherwise
        "config": {},
        "decision": None,
    }


# -- Background ------------------------------------------------------------


@given(parsers.parse('a learner working on "{concept}"'))
def working_on(world: dict, concept: str) -> None:
    world["concept"] = concept


@given(parsers.parse("the replanning threshold is {theta:f}"))
def threshold(world: dict, theta: float) -> None:
    world["config"]["theta"] = theta


@given(parsers.parse("a replan requires {n:d} items since the last one"))
def rate_limit(world: dict, n: int) -> None:
    world["config"]["min_items_between_replans"] = n


@given(parsers.parse("a misconception must recur {k:d} times"))
def repeats(world: dict, k: int) -> None:
    world["config"]["k_repeats"] = k


# -- Preconditions ---------------------------------------------------------


@given("no concept has been planned")
def no_plan(world: dict) -> None:
    world["concept"] = None


@given(parsers.parse('"{concept}" is no longer in the frontier'))
def left_frontier(world: dict, concept: str) -> None:
    world["in_frontier"] = False


@given(parsers.parse("{n:d} items have been worked since the last replan"))
def items_worked(world: dict, n: int) -> None:
    world["items_since"] = n


@given(parsers.parse("only {n:d} items have been worked since the last replan"))
def only_items_worked(world: dict, n: int) -> None:
    world["items_since"] = n


@given(parsers.parse('mastery of "{concept}" has moved by {delta:f}'))
def mastery_moved(world: dict, concept: str, delta: float) -> None:
    world["mastery"] = world["baseline"] + delta


@given(parsers.parse('"{label}" has been confirmed {n:d} times'))
def confirmed_errors(world: dict, label: str, n: int) -> None:
    world["errors"] = [
        ErrorEvent(
            t=i,
            item_id=f"i{i}",
            concept_id=world["concept"] or "c",
            misconception_label=label,
            verifier_label="incorrect",
        )
        for i in range(n)
    ]


@given(parsers.parse('"{label}" has been diagnosed but not confirmed {n:d} times'))
def unconfirmed_errors(world: dict, label: str, n: int) -> None:
    # The verifier could not read the response, so no error was established.
    # A diagnostic label alone must not move the plan.
    world["errors"] = [
        ErrorEvent(
            t=i,
            item_id=f"i{i}",
            concept_id=world["concept"] or "c",
            misconception_label=label,
            verifier_label="unparseable",
        )
        for i in range(n)
    ]


# -- Action ----------------------------------------------------------------


@when("the policy is consulted")
def consult(world: dict) -> None:
    policy = ArbitrationPolicy(ArbitrationConfig(**world["config"]))
    concept = world["concept"]

    if concept is not None:
        policy.accept({concept: world["baseline"]})
        for _ in range(world["items_since"]):
            policy.note_item()

    frontier = Frontier(frozenset({concept} if world["in_frontier"] and concept else set()))
    world["decision"] = policy.evaluate(
        current_concept=concept,
        mastery={concept: world["mastery"]} if concept else {},
        frontier=frontier,
        error_trace=world["errors"],
        prior=PRIOR,
    )


# -- Outcomes --------------------------------------------------------------


def _decision(world: dict) -> Decision:
    decision = world["decision"]
    assert decision is not None, "the policy was never consulted"
    return decision


@then("it replans")
def replans(world: dict) -> None:
    assert _decision(world).replan


@then("it does not replan")
def does_not_replan(world: dict) -> None:
    assert not _decision(world).replan


@then(parsers.parse('the trigger is "{trigger}"'))
def trigger_is(world: dict, trigger: str) -> None:
    assert _decision(world).trigger == trigger


@then("no trigger fired")
def no_trigger(world: dict) -> None:
    assert _decision(world).trigger is None


@then(parsers.parse('it was suppressed by "{guard}"'))
def suppressed_by(world: dict, guard: str) -> None:
    assert _decision(world).suppressed_by == guard


@then("the evidence names the concept and the threshold")
def evidence_is_complete(world: dict) -> None:
    evidence = _decision(world).evidence
    assert evidence.get("concept")
    assert "theta" in evidence and "delta" in evidence
