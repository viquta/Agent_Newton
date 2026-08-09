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
              Arbitration policy: thresholds, guardrails, audit
```

## Components

### Shared learner state (`core/state/`)

| Module | Holds |
|---|---|
| `schema.py` | `LearnerState`, `ErrorEvent` — pydantic models. Concept and misconception ids are opaque strings; no subject knowledge here. |
| `bkt.py` | Bayesian Knowledge Tracing — per-concept mastery posterior from a stream of correct/incorrect observations |
| `zpd.py` | The mastery frontier, derived from the posteriors and the prerequisite graph |
| `store.py` | The blackboard. Every mutation bumps `version` and appends to an immutable audit log. |
| `views.py` | `FullStateView` and `ItemCorrectnessView` |

**Views** are how the same state is exposed differently to different
configurations. `FullStateView` carries per-concept posteriors, the error trace,
misconception labels and the frontier. `ItemCorrectnessView` carries only the
correct/incorrect stream. They are two views over one state object, not two
implementations, so a configuration that changes the view changes what an agent
can see and nothing else.

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
| Scaffolding | Hint level is chosen from position within the band and the recent error trace |
| Fading | Hint level is monotonically non-increasing in mastery, all else equal |
| Error-first | A reflective prompt is required after a confirmed misconception, before remediation |

### Agents (`core/agents/`)

Prompts are parameterised by the loaded domain; none contain subject-specific
text.

| Agent | Function | Invoked |
|---|---|---|
| `tutor` | Hints and step-level feedback at the level the scaffolding predicate selects | Every step |
| `diagnostic` | Classifies an incorrect step into the domain's misconception catalogue, structured as elicit → differentiate → remediate → verify | Only on incorrect steps |
| `planner` | Proposes the next concept and item from its state view | Per item, and on replan |

The planner is hybrid: the model proposes, and a deterministic guardrail layer
rejects out-of-band, prerequisite-violating or thrashing choices and falls back.
The model's latitude is bounded by rules that cannot be prompted away.

### Arbitration policy (`core/arbitration/`)

Decides when new evidence may revise the plan.

**Triggers** — a concept enters or leaves the frontier; a mastery estimate moves
by more than `theta`; a misconception recurs `k_repeats` times within the rolling
window.

**Guardrails** — never demote below the prerequisite floor; rate-limit
replanning to at most once per `min_items_between_replans`; require verifier
confirmation, not the diagnostic agent's judgement alone, before any demotion.

Every decision writes its triggering evidence to the audit log, so a replan can
be reconstructed after the fact.

### Verifier

Supplied by the domain and called by the orchestrator after every step. Never a
tool an agent elects to invoke — see
[domain_interface.md](domain_interface.md).

### Simulated learner (`core/simulator/`)

Hybrid by design:

- `profile.py` samples a per-learner misconception profile from a seed.
- `engine.py` is deterministic and seedable. It decides whether a step is
  correct or which buggy rule fires, and owns remediation: a misconception's
  firing probability drops only when a hint correctly targets it.
- `surface.py` renders the decided step into natural student language.

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
