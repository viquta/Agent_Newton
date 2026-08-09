# The arbitration policy

Reference for `core/arbitration/`. The specification is executable: every rule
below has a scenario in `tests/integration/features/arbitration.feature`. A
document can go stale silently; a failing scenario cannot.

## What it decides

Whether the planner may be reopened. Without it the planner is consulted after
every item and the plan changes continuously, which leaves nothing for a
threshold to govern.

A **trigger** proposes a replan. A **guardrail** suppresses one. A guardrail can
never cause a replan, only prevent it.

## Triggers

| Trigger | Fires when | Suppressible |
|---|---|---|
| `no_plan` | No concept has been planned yet | no |
| `frontier_crossed` | The concept being worked has left the frontier | **no** |
| `mastery_delta` | Mastery of the current concept moved by more than `theta` since the plan was set | yes |
| `misconception_repeat` | A verifier-confirmed misconception recurred `k_repeats` times in the rolling window | yes |

`frontier_crossed` is not suppressible because continuing would mean handing out
items for a concept that has left the zone — either mastered, or with a
prerequisite that has since lapsed.

## Guardrails

**Rate limit.** At least `min_items_between_replans` items must have been worked
since the last replan. Without it, a threshold low enough to be sensitive also
makes the planner thrash between concepts on single observations.

A trigger held back this way is recorded in the audit log with
`suppressed_by: rate_limited`, and counted in `suppressed`. That distinction
matters when interpreting a threshold sweep: it separates *the threshold*
deciding from *the rate limit* deciding.

**Verifier confirmation.** Only errors the verifier judged incorrect count
toward `misconception_repeat`. A diagnostic agent's label is an opinion that an
error occurred; the verifier is what establishes there was one. Without this,
diagnostic error would propagate straight into planning — a mislabelled
unreadable response could move the plan.

## The audit trail

Every replan writes its trigger and evidence:

```json
{"cause": "replan", "version": 42,
 "summary": "replan triggered by mastery_delta",
 "evidence": {"concept": "chain_rule", "before": 0.31, "after": 0.58,
              "delta": 0.27, "theta": 0.15}}
```

Suppressed triggers are recorded too, as annotations. A replan that cannot be
explained after the run is not auditable, and the audit trail is the point.

## Reading a threshold sweep

**The triggers compete, so total replan count is a poor summary.** As `theta`
rises and suppresses `mastery_delta`, `misconception_repeat` takes up the slack,
while `frontier_crossed` contributes a floor that does not depend on `theta` at
all. Observed on the calculus domain over six learners:

| `theta` | `frontier_crossed` | `mastery_delta` | `misconception_repeat` |
|---|---|---|---|
| 0.02 | 54 | 58 | 1 |
| 0.30 | 54 | 55 | 4 |
| 0.90 | 54 | 0 | 18 |

Totals are nearly flat while the composition changes completely. Any analysis
reading only totals would conclude the threshold does nothing.

`SessionOutcome.triggers` and the cohort runner's `replans_by_trigger` therefore
report the breakdown, and a sweep should read it rather than the total. To
isolate the threshold's own effect, set `k_repeats` high enough to disable the
repeat pathway — `frontier_crossed` cannot be disabled, but it is a constant
offset rather than a confound.
