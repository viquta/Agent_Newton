# Components, one page each

Reference notes for the parts a reader will ask about. Each says what goes in,
how it is processed and where, and what comes out.

[architecture.md](../architecture.md) is the map — how the pieces fit together
and what coordinates them. These are the detail behind it.

| page | role | model? |
|---|---|---|
| [planner.md](planner.md) | chooses the goal and the next item | optional (`LLMPlanner`) |
| [tutor.md](tutor.md) | writes the turn the learner reads — a reply to a step, and a lesson between items | yes in the demo, no in cohorts |
| [diagnostic.md](diagnostic.md) | names which misconception produced a wrong step | yes, or oracle/noised-oracle |
| [arbitration.md](arbitration.md) | decides *when* the plan may reopen | never |
| [confusion.md](confusion.md) | reads the learner's words for "I do not know what this is" | yes — the one place a model decides |
| [verifier.md](verifier.md) | decides correct / incorrect / unreadable | **never, on principle** |

## Three of the six are agents, and three are not

**Arbitration is not** — it holds no view and makes no proposal about content; it
governs how often the planner is consulted. **The verifier is not** — it lives in
`domains/`, takes `(item, response)`, and is the ground truth every other
measurement inherits. **The confusion detector is not** — it classifies one
string, the way the verifier classifies one answer, and holds no view of the
learner model.

An agent here is a *decision-making component, identified by the Protocol role it
fulfils, that acts only on what it is handed and coordinates only through shared
state.* Note what that does **not** say: nothing about using a model. Most of
these never call one.

⚠️ **The three agents are constrained by three different mechanisms**, which is
worth knowing before someone opens a file expecting a fourth:

| role | how it is constrained |
|---|---|
| Planner | the **view object** — `FullStateView` or `ItemCorrectnessView` |
| Tutor | a view *plus* `mastery` and `prior_failures` supplied by the session, because *when* they were read is the point |
| Diagnostic | **no view at all** — its parameter list, and not implementing `OracleAccess` |

## How each of them is checked

These pages describe what each component does. **What it is scored against, and
how well it does, is a separate question** — and each component is scored against
a standard it did not produce: a hand-labelled gold set, an injected label the
agent never sees, or a reference policy holding the true profile.

Every one of those is a single command. [docker.md](../docker.md) lists them —
`verifier`, `diagnostic`, `planner`, `tutor`, `lessons`, `recall`, `confusion` —
and each writes a summary under `results/` rather than only printing.

⚠️ **Arbitration is the exception, and the absence is the design.** There is no
correct decision to score it against: no oracle produces one, and inventing one
would score the policy against a restatement of itself. It is characterised by
sweeping its thresholds instead — `newton sweep arbitration` — and read through
`replans_by_trigger` rather than a total, because the triggers compete.

## Conventions in these notes

References are `file.py` :: `symbol` rather than line numbers, because line
numbers rot — every one written first time round was stale within a session.
