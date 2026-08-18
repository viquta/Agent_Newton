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

The frontier itself admits two relaxations, both off by default, both recorded,
and neither available to an agent:

| `zpd.compute` | relaxes | set by |
|---|---|---|
| `waived` | the prerequisite rule — a set-aside concept stops blocking its dependants | `cohort.max_visits_per_concept` |
| `reviewing` | the mastery rule — a concept the estimate has closed stays selectable | `cohort.review_on_request`, from what the learner asked for |

`reviewing` is the only one a learner can trigger, and it lifts the upper bound
alone: a request reopens a concept and never opens the material behind one whose
prerequisites are unmet. Working the concept then tests the estimate that closed
it — answered correctly the belief holds, answered wrongly it falls and the
concept re-enters unaided.

Seeding from a held-out test is capped at `theta_upper`, so one test item can
raise the estimate but not declare mastery.

## Scaffolding

```python
hint_level(mastery, unresolved_steps, band, *, policy) -> HintLevel
```

`HintLevel` is ordered by how much support it carries:

| Level | Gives |
|---|---|
| `NONE` | Nothing. The learner is asked to look again |
| `NUDGE` | Points at the region of the error without naming it |
| `TARGETED` | Names the misconception |
| `WORKED_STEP` | Shows the step |

Two inputs. The mastery estimate sets the baseline; steps on the *current* item
that did not resolve it escalate from there, up to a ceiling, so a learner who
is stuck is not nudged repeatedly.

Steps, not attempts, and the two differ: a response the verifier could not read
costs no attempt — `cohort.max_steps_per_item` counts only measured ones — but
it still leaves the learner not having got there, so support escalates on it.

`policy` selects where the baseline and the ceiling come from.
`scaffolding.policy` sets it; the default is `banded`.

| region | `banded` base | `banded_plus` base | `banded_plus` ceiling |
|---|---|---|---|
| `P >= theta_upper` | `NUDGE` | `NONE` | `NONE` |
| `theta_lower < P < theta_upper` | `NUDGE` | `NUDGE` | `TARGETED` |
| `theta_lower / 2 < P <= theta_lower` | `TARGETED` | `TARGETED` | `WORKED_STEP` |
| `P <= theta_lower / 2` | `WORKED_STEP` | `WORKED_STEP` | `WORKED_STEP` |

`banded` uses a ceiling of `WORKED_STEP` everywhere. The two policies are
identical below `theta_lower`.

`NONE` forces a reflective turn — see `move_for` — so the only move that teaches
is withheld where the estimate says teaching is not what is missing. The region
is not reachable through selection: an item may be given only from the frontier,
and the frontier stops at `theta_upper`. There is a test asserting a session
never enters it, read off the turns a session recorded rather than off the
function.

## Support at presentation

```python
support_at_presentation(mastery, band) -> Support
```

What is shown *beside* the question, before any attempt. The rule above answers
a step the learner took; this answers where the estimate puts them before they
take one.

| region | `Support` | Shows |
|---|---|---|
| `P > theta_lower` | `NONE` | The question alone |
| `theta_lower / 2 < P <= theta_lower` | `FORMULA` | The rule, stated |
| `P <= theta_lower / 2` | `FORMULA_AND_EXAMPLE` | And a solved instance |

The boundaries are `hint_level`'s, so a learner does not sit in one tier for the
question and another for the reply. There is a test asserting they coincide
across the whole range.

Acted on only when `scaffolding.offer_at_presentation` is set, and only when the
domain supplies a `ConceptResources` — an optional sixth member, like
`ItemTemplate`. A domain that supplies none shows nothing beside any question.

The material is authored per *concept*, never per item, and its worked example
carries its own numbers. `domain validate` refuses a resource whose
`example_answer` verifies — through the domain's own verifier — as the answer to
any item on that concept, in any bank, at any template draw; and refuses a
backslash command in either text field. Both checks have a test proving they
fire.

An offer is recorded through `Blackboard.record_turn` under the move `present`,
so it reaches the audit log, the transcript and the teaching record by the route
every other instructional move takes. It targets nothing: no misconception has
been observed when it is made, and `remediation_ratio` counts what a hint aimed
at.

## Fading

```python
check_fading(band, unresolved_steps=0, *, policy) -> Violation | None
check_support_fading(band) -> Violation | None
```

Support is non-increasing in mastery, **all else equal**. Checked over both
axes and both policies. Escalation on repeated
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

```python
move_for(level, moves_since_confirmation, misconception_confirmed) -> TutorMove
```

`move_for` is what both tutors call: it applies the ordering above and returns
`REFLECT` outright at `HintLevel.NONE`. One rule in one place, rather than a
copy in each tutor.

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
