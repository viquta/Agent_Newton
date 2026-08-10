# The shared learner state

Reference for `core/state/`. Concept and misconception identifiers are opaque
strings throughout — nothing here knows what a derivative is, which is what lets
the same state serve any domain.

## Structure

| Field | Meaning |
|---|---|
| `learner_id`, `seed` | Identity and the seed the learner was generated from |
| `mastery` | `concept_id -> P(mastery)`. Absent concepts sit at the BKT prior. |
| `error_trace` | Recent errors, oldest first, bounded by `arbitration.error_trace_length` |
| `plan` | What the learner is working toward. `None` before the first plan is set. |
| `version` | Monotonic; bumped by every mutation |
| `t` | Steps taken; timestamps error events |

An `ErrorEvent` carries the item and concept, the misconception label (or `None`
when the diagnostic named nothing — distinct from a label of `"unknown"`), a
confidence, and the verifier's verdict.

A `Plan` carries the `goal` — a terminal concept the domain declares — the
`emphasis` (`consolidate` or `advance`), the version it was set at, and a
reason. It holds no sequence of concepts: the way to the goal is derived from
the rest of the state whenever it is needed, so evidence arriving mid-session
changes the next step rather than invalidating a stored one. See `route.py` and
[architecture.md](architecture.md).

## Mastery estimation

Bayesian Knowledge Tracing (Corbett & Anderson, 1995) with four parameters:
`p_init`, `p_transit`, `p_guess`, `p_slip`. Each observation revises the
posterior by Bayes, then the transition carries it forward: what is not yet
known may be learned at each opportunity.

Estimates are clamped away from 0 and 1. At exactly 1.0 no later evidence could
move the estimate, so one lucky streak would freeze a concept as mastered and
the frontier could never reopen it.

`p_guess + p_slip >= 1` is rejected at config load — the model degenerates and a
correct answer would *lower* the estimate.

## The frontier

```
frontier = { c : P(c) < theta_upper  and  for all p in prereqs(c): P(p) > theta_lower }
```

Not yet independently mastered, but every prerequisite is in place. Note that
the two thresholds differ, so a concept between them counts as a *met
prerequisite* while remaining unmastered itself — progress unlocks the next
concept before it is finished.

Computed as a pure function of mastery and the graph, and cached against the
state version, so every agent reads the same zone within a step.

**An empty frontier means the learner is done, not that something failed.** That
case is distinguished from failure explicitly. The recovery branch — falling back
to the earliest unmastered concept in topological order — should be unreachable:
topological order places prerequisites before dependants, so the first unmastered
concept cannot have an unmastered prerequisite and is therefore always in the
zone. `Frontier.fallback` is `True` if it ever fires, which indicates the graph
and the mastery map disagree. A property test asserts the invariant across mastery
configurations.

`is_unconstrained()` reports a band admitting essentially the whole graph. Not an
error — a learner at the start legitimately has many reachable concepts — but if
it holds throughout a run, the band is not selecting anything.

## The blackboard

Every mutation goes through `Blackboard`, so no path exists that changes state
without bumping the version and appending to the audit log. Agents hold a view,
never the state.

```python
board.record_observation(item_id=..., concept_id=..., result=..., misconception_label=...)
board.record_replan("threshold crossed", concept=..., delta=...)
board.annotate("...", **evidence)
```

`record_observation` returns whether the result counted as evidence.

**Only evidence updates the learner model.** A `VerificationResult` whose
`is_evidence` is `False` — unreadable input, or an answer the verifier could not
decide — is a failure to measure, not information about the learner. It is
counted in `board.unmeasurable` and written to the audit log, because the attempt
happened and a rising rate means the verifier is failing rather than the learner,
but it moves no estimate, appends no outcome and adds no error event.

`audit_log` returns a tuple, so callers cannot rewrite history. Every replan
records the evidence that triggered it, which is what makes a decision
reconstructible after a run rather than only observable during it.

## Views

Agents receive a view chosen by `config.arm`. Both are windows onto the *same*
state object, so the two configurations cannot drift apart in any other respect.

| | `FullStateView` (coupled) | `ItemCorrectnessView` (decoupled) |
|---|---|---|
| Per-concept mastery | yes | — |
| Error trace and labels | yes | — |
| Frontier | yes | — |
| Outcome stream | yes | yes |
| `consecutive_correct()` | yes | yes |
| Plan (goal and emphasis) | yes | yes |

The second is not a coarser version of the first. It has neither the posteriors
nor the graph, so **no frontier can be computed from it** — there is no method
that would return one. A planner given this view is structurally incapable of
frontier-based selection rather than merely discouraged from it.

The same limit extends to the plan. Both views carry it, because a goal is
curriculum and withholding it would make the comparison measure ignorance of
the target rather than inability to route toward it. But routing toward that
goal needs the posteriors, and honouring the emphasis needs the posteriors or
the error trace, so a planner on the second view produces identical selections
whichever emphasis was configured.
