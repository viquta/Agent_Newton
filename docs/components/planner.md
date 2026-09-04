# Planner

Chooses the goal and the next item. What it is handed, how it works, and what it produces.

Part of the component reference — [architecture.md](../architecture.md) is
the map, and [the index](README.md) lists the rest.

Chooses **what to work toward** and **what to work on next**. The only agent that
decides what the learner meets.

`core/agents/base.py` (the Protocol) · `core/state/schema.py` (`Plan`) ·
`core/agents/planner.py` (six implementations) · `core/agents/llm.py`
(`LLMPlanner`) · `core/state/route.py` (the routing) · `core/state/zpd.py`
(the frontier)

---

## Scope

⚠️ **Both columns.** These files are more comment than code by design, so a
whole-definition count measures documented surface rather than logic.

**The planner proper**

| | lines | non-comment |
|---|---|---|
| `agents/base.py` :: `Planner` — a Protocol, **never executes** | 21 | 17 |
| `state/schema.py` :: `Plan` — a dataclass, built on every replan | 17 | 12 |
| `planner.py` :: `GoalDirectedPlanner` 85, `FixedOrderPlanner` 106 | 191 | 140 |
| `planner.py` :: `_ClosureWalker` 44, `ReverseOrderPlanner` 11, `ShuffledPlanner` 17 | 72 | 61 |
| `planner.py` :: `OraclePlanner` 67, `FrontierPlanner` 32 | 99 | 83 |
| `llm.py` :: `LLMPlanner` | 90 | 75 |
| | **490** | **388** |

⚠️ **The first two rows are different kinds of thing.** `Planner`'s method bodies
are `...` — nothing instantiates it and nothing calls it; it constrains the
implementations through pyright and has no run-time effect. `Plan` is a
dataclass, built whenever the plan is set. "`base.py` never runs" is true of the
Protocols in it and false of the dataclasses beside them.

⚠️ **`agents/base.py`, not `base.py`.** Four files carry that name —
`core/agents/`, `core/recall/`, `domains/`, `llm/`.

**What it composes, and what is written about it**

| | lines |
|---|---|
| `core/state/route.py` — the four narrowings live here, not in the planner | 325 |
| `core/state/zpd.py` — the frontier | 157 |
| `core/evaluation/planning.py` — the reference policy and the shadow | 248 |

`route.py` stands to the planner as `pedagogy/policy.py` does to the tutor: the
rules it composes rather than part of it. A planner that inlined `next_step`
would still work and would stop being swappable, which is the whole point of the
role.

---

## Inputs

```python
def plan(self, view: StateView, domain: Domain) -> Plan | None
def select(self, view: StateView, domain: Domain, given: Mapping[str, int]) -> Item | None
```

| input | where it comes from | what it carries |
|---|---|---|
| `view` | `board.view()`, `store.py` :: `Blackboard.view` | **the arm.** `FullStateView` (10 fields) or `ItemCorrectnessView` (3) |
| `domain` | `registry.load_domain(...)` | graph, item bank, catalogue — *curriculum*, identical in both arms |
| `given` | `state.items_given` | lifetime count per item id, so a returning learner gets fresh numbers |

⚠️ **`view` is the entire independent variable of the study.** Both arms get the
same domain, the same learner, the same seed. One `if` in `Blackboard.view`
decides which object arrives here.

The coupled view carries `mastery`, `error_trace`, `frontier`, `reflections`,
`weaknesses`, `requested`, `reviewing`. The decoupled view carries `outcomes`
(a right/wrong bit-stream), `version` and `plan`.

### What it does with `FullStateView`

Eight of the ten fields are read, and `select` is where most of it happens.
Counted from `planner.py`:

| field | read | what for |
|---|---|---|
| `mastery` | ×6 | which goal is reached, and every ranking key |
| `frontier` | ×6 | the reachable set, and `may_select`'s own check |
| `plan` | ×7 | the goal and emphasis to route toward |
| `requested` | ×3 | moves which **goal** comes next, not which concept |
| `error_trace` | ×1 | `CONSOLIDATE`'s "recent errors first" |
| `weaknesses` | ×1 | concepts already set aside, deprioritised |
| `reviewing` | ×1 | relaxes the upper bound so a reviewed concept stays in the zone |
| `outcomes` | ×1 | via `consecutive_correct`, the decoupled walk's advance signal |
| `reflections` | — | **never read.** The learner's words are the tutor's input |
| `version` | — | **never read.** Ordering is the store's concern |

⚠️ `requested` moving the *goal* rather than the ranking is the subtle one, and
`next_goal`'s docstring gives the reason: re-ranking within a frontier could not
honour a request at all, because a concept off the way to the current goal is not
a candidate in the first place. The request would look accepted and change
nothing.

To see the difference concretely, build a session on each arm and print what
`board.view()` returns: the coupled view answers `frontier`, `mastery` and
`error_trace`; on the decoupled view those attributes do not exist.

---

## How it works


### `GoalDirectedPlanner`, the coupled arm

Four narrowings, each discarding something.

⚠️ **The goal is one concept at a time, not "finish the graph".**
`concepts.yaml` declares **five terminal goals, in order** — on calculus:
`negative_fractional_exponents`, `stationary_points`, `quotient_rule`,
`implicit_differentiation`, `integration_by_substitution`. `next_goal` takes the
first not yet reached and the order is the domain's curriculum decision, so the
function takes the goals as given rather than choosing among them. That is why
the outcome is `goals_mastered` out of five, and why `distance_to_goal` is a
distance to *a* goal rather than to the end of the DAG.

**1. Goal** — `route.next_goal` (`route.py` :: `next_goal`). The first declared goal not yet
reached, except that a *live* request outranks the order, and a **reviewed**
request is a fallback after that.

**2. Relevance** — `route.relevant(goal, graph)` is the goal's prerequisite
closure plus itself. On calculus, `integration_by_substitution` pulls in 10 of 15
concepts.

**3. Reachability** — `route.candidates` intersects that closure with the
frontier: not mastered, prerequisites met. Typically 1–3 concepts survive.

**4. Ranking** — `route.rank` (`route.py` :: `rank`), by `Emphasis`:

- `ADVANCE` — deepest first. As soon as something further opens, take it.
- `CONSOLIDATE` — recent errors, then shallowest, then **touched but
  unfinished**, then furthest from mastery.

Then `_least_used` (`planner.py` :: `_least_used`) turns the concept into an item — the
least-practised one, on the lifetime count.

Finally the planner checks its own proposal:

```python
# planner.py :: GoalDirectedPlanner.select
if not step.fallback and may_select(step.concept_id, full.frontier) is not None:
    return None
```

### The decoupled arm — `FixedOrderPlanner`

Walks the syllabus. Advances on consecutive correct answers. Its whole
justification is `"walking the syllabus; position 0"`. It implements `Resumable`,
because its position in the walk is the only progress signal it has and must
survive between sittings — the coupled planner holds nothing, since everything it
routes from is on the board.

---

### Implementations

Eight of them, and only two run in a cohort.

| class | used for |
|---|---|
| `GoalDirectedPlanner` | the coupled arm |
| `FixedOrderPlanner` | the decoupled arm |
| `ReverseOrderPlanner`, `ShuffledPlanner` | ordering probes (§7c) — same material, different order |
| `FrontierPlanner` | selects from the frontier without a goal |
| `OraclePlanner` | reference policy for planner-vs-oracle; holds the live profile |
| `LLMPlanner` | model-backed, wrapping the deterministic guardrail |
| `ShadowedPlanner` | ⚠️ the eighth — `core/evaluation/planning.py`. Delegates to the planner under test and asks the reference the same question on the way past |

All eight satisfy the same Protocol **structurally** — none inherits from a
Planner base class, and there is no such class. ⚠️ `ReverseOrderPlanner` and
`ShuffledPlanner` do share an internal `_ClosureWalker`, which is ordinary code
reuse *within* a role and not the Protocol being satisfied by inheritance. The
distinction matters: the roles are duck-typed, so an implementation is a Planner
because it has the two methods, not because of what it descends from.

⚠️ **`ShadowedPlanner` is an agent and an instrument at once**, which is why
it was missed: it lives under `core/evaluation/`, so it reads as measurement
rather than as an implementation. It satisfies the Protocol, the session runs
unchanged around it, and there is a test asserting a shadowed session produces
identical outcomes to a plain one. That is what makes the planner-vs-reference
comparison collectable through the genuine loop instead of a duplicate of it.

---

## Output

`Plan(goal, emphasis, set_at_version, reason)` and `Item | None`.

`None` from `select` means nothing is left to teach — the session records
`nothing_left_to_select` and stops. The planner writes nothing itself; the
session records the plan and the consequences.

⚠️ `reason` is prose for the audit log, and it is what makes a run
reconstructible: `"consolidate toward integration_by_substitution: chose
integration_by_substitution from 2 candidate(s)"`.

### What a learner sees of all this

In the demo, the *what* and never the *why*. Above every question the board panel
shows:

- the **goal by name**, with `route.remaining` as `"n concept(s) to go"`, and the
  emphasis beside it;
- **`▶` markers** on the frontier concepts in the mastery table;
- **`↻ goal set to 'quotient_rule' (consolidate)`** whenever a `plan` or `replan`
  record appears in the audit log since the last question.

⚠️ **`Plan.reason` is not among them.** It goes into the audit record's
`evidence`; the panel prints `record.summary`, which `record_plan` sets to
`f"goal set to {goal!r} ({emphasis})"`. So a learner is told the target and never
the deliberation — not which candidates were considered, not how many there were,
not why this one won. Whether that is the right amount to show is a design
question and an open one; what matters here is that the shortlist is *available*
in the log and deliberately not surfaced.

---
