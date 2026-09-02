# Architecture

## Coordination model

The learner state is a **blackboard**. Agents do not call one another; each
reads a view of the shared state and writes observations back to it. Adding a
direct call between two agents would be a design error, so
`tests/integration/` asserts no such path exists.

```
                    ┌──────────────────────────────────┐
   Simulated        │   SHARED LEARNER STATE           │
   learner  ──┐     │   mastery estimate per concept   │
              │     │   rolling error trace            │
              ▼     │   versioned + append-only audit  │
   Verifier ──────► │   ── derived: mastery frontier ──│
   (domain)  │      └───┬──────────┬──────────┬────────┘
             │          │ view     │ view     │ view
   Tutor ◄───┘      Tutor      Diagnostic   Planner
                                               ▲
                                               │ gates
              Arbitration policy: thresholds, guardrails, audit
```

## Components

Each of the six decision-making parts has its own page under
[`components/`](components/README.md) — what it is handed, how it works and what
it produces. **This section is the map**: what exists, where it lives and how the
pieces fit. Where a component has a page, the detail is there and not repeated
here, so that there is one description of each mechanism rather than two.


### Shared learner state (`core/state/`)

| Module | Holds |
|---|---|
| `schema.py` | `LearnerState`, `ErrorEvent`, `Plan` — pydantic models. Concept and misconception ids are opaque strings; no subject knowledge here. |
| `bkt.py` | Bayesian Knowledge Tracing — per-concept mastery posterior from a stream of correct/incorrect observations |
| `zpd.py` | The mastery frontier, derived from the posteriors and the prerequisite graph |
| `route.py` | The way to the goal, derived from the posteriors, the error trace and the graph |
| `decay.py` | What a gap between sittings does to a posterior. Both estimates relax toward the BKT prior — belief going stale, not the learner forgetting. Nothing schedules a review: a decayed posterior falls back below `θ_upper` and the route passes through it again. |
| `store.py` | The blackboard. Every mutation bumps `version` and appends to an immutable audit log. |
| `views.py` | `FullStateView` and `ItemCorrectnessView` |

**Views** are how the same state is exposed differently to different
configurations. `FullStateView` carries per-concept posteriors, the error trace,
misconception labels and the frontier; it also carries what the learner said
(`reflections`, reachable per concept through `said_about`), the concepts worked
often enough to be set aside (`weaknesses`), and the concepts a learner asked for
and is being given again (`requested`, `reviewing`). `ItemCorrectnessView`
carries only the correct/incorrect stream. They are two views over one state object, not two
implementations, so a configuration that changes the view changes what an agent
can see and nothing else.

Both views carry the `Plan`. A goal is curriculum, like the item bank and the
graph, so it is not part of what the ablation withholds.

### Goals and the route to them

A domain declares terminal concepts in `concepts.yaml`, in the order they are
worked toward:

```yaml
goals: [negative_fractional_exponents, stationary_points, quotient_rule]
```

Absent, the graph's sinks are used. `ConceptGraph.goals()` returns them.

The `Plan` on `LearnerState` holds the goal currently aimed at and the
`Emphasis` — how the learner wants to reach it. It holds no sequence of
concepts: the way there is recomputed from the state each time it is needed,
the same way the frontier is.

```
relevant(goal)  = all_prerequisites(goal) ∪ {goal}
candidates      = frontier ∩ relevant(goal)
next step       = rank(candidates) by emphasis
```

| `Emphasis` | Ranking among candidates |
|---|---|
| `consolidate` | most errors in the trace, then shallowest, then furthest from mastery |
| `advance` | deepest reachable first |

The two differ because the band has width. A concept opens once its
prerequisites clear `θ_lower` but stays selectable until it clears `θ_upper`,
so there is a range in which a learner may either move on or stay.

Ranking uses `depth` rather than position in the topological order: the latter
is only *a* total order consistent with the graph, and among concepts at equal
depth it comes from the order the YAML happens to be written in.

Computing any of this needs the posteriors and the error trace, so a planner
holding `ItemCorrectnessView` produces the same selections whichever emphasis
was configured.

### The mastery frontier

A concept is in the frontier when it is not yet independently mastered but every
prerequisite is:

```
frontier(state, graph) = { c : P(mastery_c) < θ_upper
                           ∧ ∀p ∈ prereqs(c): P(mastery_p) > θ_lower }
```

Computed as a pure function of the state and cached per `version`, so every
agent sees one consistent frontier within a step. It constrains item selection
and drives hint level.

Computing it requires per-concept posteriors *and* the prerequisite graph, so it
is available only through `FullStateView`.

**Degenerate cases.** Badly chosen thresholds can empty the frontier (nothing
selectable) or admit the whole graph (no constraint). The planner falls back to
the topologically shallowest unmastered concept and logs the fallback; fallback
frequency is reported as a run diagnostic.

### Pedagogy predicates (`core/pedagogy/`)

Instructional rules expressed as checkable predicates rather than prose, so they
can be asserted in tests and logged per decision:

| Predicate | Rule |
|---|---|
| Band membership | An item may be selected only if its concept is in the frontier |
| Scaffolding | Hint level is chosen from position within the band and the recent error trace. Two ladders: `banded` steps at `θ_lower` and `θ_lower / 2`; `banded_plus` reads every cut point off the band instead, disclosing nothing above `θ_upper` and capping escalation at `targeted` inside it |
| Support at presentation | A second axis, decided *before* the first attempt rather than after a failure: below `θ_lower` the rule is shown beside the question, and further down a solved example on other numbers |
| Fading | Hint level is monotonically non-increasing in mastery, all else equal |
| Error-first | A reflective prompt is required after a confirmed misconception, before remediation |

### Agents (`core/agents/`)

Prompts are parameterised by the loaded domain; none contain subject-specific
text.

One page each: [planner](components/planner.md) ·
[tutor](components/tutor.md) · [diagnostic](components/diagnostic.md). The
[confusion detector](components/confusion.md) sits beside them and is not an
agent — [the index](components/README.md) says why.

| Agent | Function | Invoked |
|---|---|---|
| `tutor` | Hints and step-level feedback at the level the scaffolding predicate selects | Every step |
| `diagnostic` | Classifies an incorrect step into the domain's misconception catalogue, structured as elicit → differentiate → remediate → verify | Only on incorrect steps |
| `planner` | Names the goal, and selects the next concept and item on the way to it | `select()` every item; `plan()` only when arbitration allows a replan, or an item was set aside |

A planner makes two decisions at two timescales. `plan()` names the target —
the first declared goal not yet reached. `select()` chooses the item on the way
to it. Both read the same view, so a planner that cannot see the learner model
is limited at both.

| Implementation | Behaviour |
|---|---|
| `goal_directed` | Routes toward the declared goals from the posteriors and the error trace |
| `greedy` | Frontier selection with no target — the undirected predecessor, kept as a baseline |
| `llm` | Proposes among the goal-directed candidates; the guardrail decides whether the proposal stands |
| `oracle` | Selects against the simulated learner's profile — the ceiling a selection policy could reach, not a condition any real agent could occupy |
| `reverse`, `shuffled` | Ordering probes. Same material, deliberately worse order, so an outcome that depends on sequencing can be told from one that does not |

The model-backed planner is hybrid: the model proposes, and a deterministic
guardrail layer rejects out-of-band, off-route or thrashing choices and falls
back. The model's latitude is bounded by rules that cannot be prompted away, and
it does not choose the goal.

The decoupled arm's planner walks the union of every goal's prerequisite closure
in topological order, advancing on consecutive correct answers. It is restricted
to the same material — that is curriculum — but cannot skip what this learner
has already mastered or return to what they are struggling with.

The union is deliberately not narrowed to the *current* goal: the walk only
moves forward, so narrowing would step past a concept a later goal needs and
never return to it. Note that on `calculus` the restriction is currently inert —
every concept is an ancestor of some declared goal, so the union is the whole
graph. It bites only in a domain carrying material that lies on the way to no
goal.

A planner that walks off the end of its list returns nothing and the session
ends, which is why the decoupled arm can attempt fewer items than the budget
allows. That is an implementation choice, not a consequence of the missing
learner model.

### Arbitration policy (`core/arbitration/`)

Decides when new evidence may revise the plan — the triggers, the guardrails, the
audit trail and how to read a threshold sweep are all on
[components/arbitration.md](components/arbitration.md).

**The policy reads the board, not the arm's view.** It is handed
`board.state.mastery` and `board.frontier` in *both* configurations, so the
decoupled arm receives replan timing derived from information its own planner
cannot see. That is deliberate: the guardrail layer is held constant so that the
planner's view is the single thing that varies. The consequence is that the
decoupled arm is being compared at its best — it is told *when* to reconsider by
a policy better informed than itself — which makes any difference measured
between the arms a conservative estimate.

Setting a goal is recorded under the audit cause `plan`, not `replan`, so it
stays out of the trigger counts a threshold analysis reads.

### The session loop (`core/orchestration/`)

`session.py` is the only thing that moves information between agents, and the
single file to read first for control flow. Each step: the planner selects, the
learner answers, the verifier grades, the diagnostic labels an incorrect step,
the board updates, the arbitration policy decides whether the plan may reopen,
and the tutor replies at the level the scaffolding predicate chose.

`SessionObserver` is how a session is watched without being altered — the demo's
live panel is one, and a run's `events.jsonl` another. Nothing an observer does
reaches the loop.

A session ends for one of four reasons, and which one is an outcome rather than
an implementation detail:

| Stop reason | Meaning |
|---|---|
| `budget_spent` | The item budget ran out — the ordinary ending |
| `every_goal_reached` | Every declared goal cleared `θ_upper`. Only a planner that names goals can reach this |
| `nothing_left_to_select` | The planner returned no item: the frontier emptied, or a fixed-order walk ran off the end of its list |
| `learner_ended_it` | A person stopped |

`nothing_left_to_select` is why the decoupled arm can attempt fewer items than
the budget allows, and it is not the same event as `every_goal_reached`. Only one
of the two planners can tell them apart, which is why both are recorded.

### Evaluation (`core/evaluation/`)

Each component is scored against something that is not another model's opinion.

| Module | Scores |
|---|---|
| `verifier.py` | The symbolic verifier against a hand-labelled gold set, including correct answers written differently |
| `diagnostic.py` | Inferred labels against the injected ones. The only place the two ever meet |
| `planning.py` | A planner's selections against a reference policy |
| `outcomes.py` | Learning outcomes per learner — gain, normalised gain, remediation |
| `teaching.py` | What was taught per concept, so appropriate instruction can be established for a learner who never grasps one |
| `sitting.py` | A stored audit log rendered back as prose |
| `tutor.py` | Hints against deterministic checks, and against a judge for the two questions no predicate can settle |
| `statistics.py` | Paired tests over the arms, with correction across outcomes |

### Verifier

Supplied by the domain and called by the orchestrator after every step. Never a
tool an agent elects to invoke — see
[domain_interface.md](domain_interface.md) for the boundary, and
[components/verifier.md](components/verifier.md) for the three-stage comparison
and the three-valued verdict.

### Simulated learner (`core/simulator/`)

Hybrid by design:

- `profile.py` samples a per-learner misconception profile from a seed.
- `engine.py` is deterministic and seedable. It decides whether a step is
  correct or which buggy rule fires, and owns remediation: a misconception's
  firing probability drops only when a hint correctly targets it.
- `surface.py` renders the decided step into natural student language.
- `human.py` puts a person behind the same `Learner` protocol, which is how the
  demo runs the cohorts' session loop with someone answering at a keyboard.

The rule engine owns behaviour; the model only phrases it. Setting
`simulator.surface: symbolic` removes the model entirely, making runs fast and
exactly reproducible. The simulated learner's profile is never visible to any
agent.

## Reproducibility

Each run writes a manifest recording the git SHA and working-tree cleanliness,
configuration hash, domain name, content hashes for the concept graph,
misconception catalogue and item banks, resolved provider/model per role, seeds
and timestamps.

Model responses are cached on disk keyed by `(provider, model, prompt hash)`, so
re-running an analysis over stored results costs nothing and returns identical
output.

Structured events go to `events.jsonl` in the run directory — one JSON object
per line, carrying arbitration decisions, replanning triggers and state
transitions. This is the audit log in machine-readable form.
