# Instructional rules

Reference for `core/pedagogy/`. Teaching rules stated as prose cannot be
enforced or tested; stated as predicates over the shared state, they can be.

Every rule is a function the tutor and the planner's guardrail layer must
satisfy. Each returns a `Violation` rather than merely failing, carrying a rule
name and a readable message, so a breach can be written to the audit log instead
of raising.

## Band membership

```python
may_select(concept_id, frontier) -> Violation | None
```

An item may be given only if its concept is in the current frontier. A concept
outside it is either already mastered or has an unmet prerequisite — in both
cases it is not the item to give next.

The planner's guardrail layer calls this on every proposal, so a model that
suggests an out-of-band item has that suggestion rejected rather than merely
discouraged.

## Scaffolding

```python
hint_level(mastery, unresolved_steps, band) -> HintLevel
```

`HintLevel` is ordered by how much support it carries:

| Level | Gives |
|---|---|
| `NUDGE` | Points at the region of the error without naming it |
| `TARGETED` | Names the misconception |
| `WORKED_STEP` | Shows the step |

Two inputs. The mastery estimate sets the baseline — above `theta_lower` a
nudge suffices, below half of it the step is worked. Steps on the *current*
item that did not resolve it escalate from there, bounded at `WORKED_STEP`, so
a learner who is stuck is not nudged repeatedly.

Steps, not attempts, and the two differ: a response the verifier could not read
costs no attempt — `cohort.max_steps_per_item` counts only measured ones — but
it still leaves the learner not having got there, so support escalates on it.

## Fading

```python
check_fading(band, unresolved_steps=0) -> Violation | None
```

Support is non-increasing in mastery, **all else equal**. Escalation on repeated
failure varies support at fixed mastery, which is why the qualifier is
load-bearing rather than decorative — without it, escalation would read as a
violation.

Because this is a claim about the shape of the function rather than about
particular values, it is checked across the range rather than at sample points,
at every escalation level and for several bands. A single inverted step would
escape a spot check.

The test suite also verifies the check itself fires on a deliberately inverted
implementation — a property test that cannot fail proves nothing.

## Error first

```python
check_move(move, moves_since_confirmation, misconception_confirmed) -> Violation | None
next_required_move(moves_since_confirmation, misconception_confirmed) -> TutorMove | None
```

Once a misconception is confirmed, remediation must be preceded by a reflective
prompt: the learner is asked to look at their own reasoning before being handed
the correction. Without the ordering constraint a tutor can go straight to the
answer, which is the behaviour the rule exists to prevent.

`next_required_move` lets the tutor be *driven* by the constraint rather than
checked against it afterwards. A tutor that asks what is required and does it
cannot violate the rule — there is a test asserting exactly that, since a
constraint layer that only ever rejects is harder to build against than one that
also says what to do.

The rule is inert until a misconception is confirmed: with nothing to reflect
on, plain remediation is unconstrained.

## Measuring the turns the rules produce

The predicates above choose a move and a support level. They say nothing about
the words carrying them, and a reply can satisfy every one of them while doing
the opposite of what its level means — a `NUDGE` that states the answer breaks
no predicate.

`core/evaluation/tutor.py` measures the text. Two layers, and they are not
interchangeable.

**Deterministic checks** decide what has a right answer, and gate CI:

| check | what it decides |
|---|---|
| `answer_leaked` | any fragment of the reply verifies as the answer, below `WORKED_STEP` |
| `latex_in_reply` | a backslash command, or the control character it becomes once a JSON reply is unescaped |
| `reflect_tells` | a reflective prompt that hands over the answer |
| `over_length` | more than two sentences |

`answer_leaked` asks the domain's own verifier rather than comparing strings, so
`5x^4` and `5*x**4` are the same disclosure. Fragments the question already
contains are excluded: a number the learner was given cannot be given away.

**Judged checks** decide what does not. Whether a reply keeps to what the
student's step shows, and whether the assigned levels are visible in the text,
are judgements. A second model makes them, and it is scored against
`tests/fixtures/gold/calculus_tutor_cases.yaml` first — so the report carries its
agreement with the hand labels beside its verdicts.

Report the agreement whenever the verdicts are quoted. A judge whose agreement
is unknown produces rates whose error is unknown.

```bash
uv run agent-newton evaluate tutor --domain calculus --no-think \
    --judge-model <a different model>
```

Split by bank as well as by level and move. Only `practice` turns reach a
learner — the test banks are administered without hints — so that row is the
rate a running system exposes anyone to.
