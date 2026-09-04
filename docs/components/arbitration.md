# Arbitration

Decides *when* the plan may reopen. What it is handed, how it works, and what it produces.

Part of the component reference — [architecture.md](../architecture.md) is
the map, and [the index](README.md) lists the rest.

Decides **when** the plan may be reopened. Not an agent — it holds no view and
makes no proposal about content. It governs how often the planner is consulted
at all.

`core/arbitration/policy.py` · called from `session.py` :: `run`

**There are two things to understand and one picture cannot hold both:** what a
single call decides, and the lifecycle *between* calls — which is where the
baseline reset lives, and why `theta` measures change since the last replan
rather than since the last item.

⚠️ **It is not an agent.** It holds no view and makes no proposal about content;
it is handed values and gates the planner. The natural mistake is to draw it as
a fourth agent beside Tutor, Diagnostic and Planner, each handed a view of the
shared state. Arbitration is not one of those.

---

## Scope

**In:** whether the planner is consulted again on this step, and which trigger is
credited for it.

**Out:** everything about *what* is then chosen. Arbitration never sees an item,
a concept or a goal — reopening the plan and deciding what the reopened plan says
are different jobs, and the second belongs to the planner. It is also not the
only thing that can reopen a plan: `frontier_crossed` is not suppressible by
design, since continuing would mean giving items for a concept that has left the
zone.

---

## Inputs

```python
def evaluate(
    self, *,
    current_concept: str | None,
    mastery: dict[str, float],
    frontier: Frontier,
    error_trace: list[ErrorEvent],
    prior: float,
) -> Decision
```

Read straight from the board by the session — the policy holds no reference to
it. Plus two bookkeeping calls:

- `note_item()` — one item was given (`session.py` :: `run`)
- `accept(mastery)` — a replan happened: **reset the baseline and the rate
  limit** (`session.py` :: `run`)

Configured by `ArbitrationConfig`: `theta` (0.15), `k_repeats` (2),
`min_items_between_replans`, `error_trace_length` (20).

---

## How it works

**A trigger proposes a replan; a guardrail suppresses one.** The asymmetry is
worth holding on to: a guardrail can never *cause* a replan, only prevent one, so
every replan in the log is traceable to a trigger and every absence to either no
trigger or a named guardrail.


Four checks, in order.

**1. No plan** → `Decision(True, trigger=NO_PLAN)`. Nothing to protect.

**2. The concept left the frontier** → `FRONTIER_CROSSED`, and this one is **not
suppressible**: continuing would mean giving items for a concept that has left
the zone.

**3. A trigger, from `_find_trigger`:**

| trigger | condition |
|---|---|
| `mastery_delta` | `abs(after - before) > theta`, against the baseline taken at the **last replan** |
| `misconception_repeat` | one label appears `>= k_repeats` times in the rolling window |

⚠️ Only verifier-confirmed errors count:

```python
# policy.py :: _find_trigger — a diagnostic label is an opinion about an error;
# the verifier is what establishes there was one.
confirmed = [e for e in window if e.verifier_label == "incorrect"]
```

**4. The rate limit.** A trigger firing within `min_items_between_replans` is
**suppressed and counted** — `Decision(False, trigger=…, suppressed_by=RATE_LIMITED)`.
The session annotates it, so a trigger that fired and was held back stays in the
record.

---

## Output

```python
Decision(replan: bool, trigger: str | None, suppressed_by: str | None, evidence: dict)
```

`summary` renders it for the log: `"replan triggered by mastery_delta"`,
`"mastery_delta suppressed by rate_limited"`, `"no trigger"`.

The session branches on `decision.replan or set_aside`:

- **true** → `_retarget()`, then `planner.select(board.view(), …)` — the one
  arrow that leaves the Session lane for the Planner
- **false** → `_next_item_for(working)`, the least-practised item on the same
  concept, with no planner involved

---

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

---

## Reading a threshold sweep

**The triggers compete, so a total replan count is a poor summary.** As `theta`
rises and suppresses `mastery_delta`, `misconception_repeat` takes up the slack,
while `frontier_crossed` contributes a floor that does not depend on `theta` at
all. Observed on the calculus domain over six learners:

| `theta` | `frontier_crossed` | `mastery_delta` | `misconception_repeat` |
|---|---|---|---|
| 0.02 | 54 | 58 | 1 |
| 0.30 | 54 | 55 | 4 |
| 0.90 | 54 | 0 | 18 |

Totals are nearly flat while the composition changes completely. **Any analysis
reading only totals would conclude the threshold does nothing** — and would be
describing a substitution between triggers rather than an absence of effect.

`SessionOutcome.triggers` and the cohort runner's `replans_by_trigger` therefore
report the breakdown, and a sweep should read that rather than the total. To
isolate the threshold's own effect, set `k_repeats` high enough to disable the
repeat pathway; `frontier_crossed` cannot be disabled, but it is a constant
offset rather than a confound.

---

## What is not arbitration

`concept_set_aside` is its own trigger name, not one of the three above. It fires
when the dwelling cap sets a concept aside, and it exists as a separate name so
the threshold sweep reads it apart from the triggers arbitration owns. It cannot
fire without `cohort.max_visits_per_concept`, which no experiment config sets.

---
