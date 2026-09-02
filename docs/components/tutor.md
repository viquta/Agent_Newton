# Tutor

Writes the turn the learner reads. What it is handed, how it works, and what it produces.

Part of the component reference — [architecture.md](../architecture.md) is
the map, and [the index](README.md) lists the rest.

Writes the turn the learner reads. **Two turns, not one**: a reply to a step
that did not come out right, and — between items — a lesson on the concept
itself.

`core/agents/base.py` (the Protocol, and `Hint`) ·
`core/agents/tutor.py` (`TemplateTutor`) · `core/agents/llm.py` (`LLMTutor`) ·
`core/pedagogy/policy.py` (the rules that choose the level, the move and the
style) · `core/agents/schemas.py` (the constrained replies)

---

## Scope — and it is the largest of the three roles

⚠️ **Both columns.** These files are more comment than code by design, so a
whole-definition count measures documented surface rather than logic. Both are
given, and the gap between them is the point rather than an embarrassment.

**The tutor proper**

| | lines | non-comment |
|---|---|---|
| `agents/base.py` :: `Tutor` — a Protocol, **never executes** | 132 | 94 |
| `agents/base.py` :: `Hint` — a dataclass, built on every turn | 20 | 12 |
| `agents/tutor.py` :: `TemplateTutor` | 107 | 76 |
| `llm.py` :: `LLMTutor` | 375 | 230 |
| `schemas.py` :: `HintReply` 4, `LessonReply` 60, `ClosingReply` 25 | 89 | 76 |
| | **723** | **488** |

⚠️ **The first two rows are not the same kind of thing, and one of them runs no
code at all.** `Tutor`'s method bodies are `...`; nothing instantiates it and
nothing calls it. It shapes the code without executing — pyright checks the
implementations against it, and that is the whole of its effect at run time.
`Hint` is a `@dataclass` constructed on every turn, four times over the two
implementations (`llm.py :: LLMTutor.respond`, and three returns in
`tutor.py :: TemplateTutor.respond`).

So "`agents/base.py` never runs" — which is easy to conclude once you notice
the `...` bodies — is true of the **Protocols** in it and false of the
**dataclasses**. The file holds both: declarations that constrain without
executing, and types that are built at run time. `Diagnosis` and `Plan` sit on
the same split.

⚠️ **`agents/`, not `base.py`.** There are four files called `base.py` —
`core/agents/`, `core/recall/`, `domains/` and `llm/` — and this document
touches three of them: the Protocol and `Hint` here, `domains/base.py` for
`ConceptResource`, and `llm/base.py` for `complete()` and `ProviderError`. A
bare `base.py` does not resolve.

⚠️ `base.py` is not the only one. `policy.py` is both `core/pedagogy/` and
`core/arbitration/`; `tutor.py` is both `core/agents/` and
`core/evaluation/` — and this document uses **both** senses of that one, the
implementation in the Scope table and the evaluation in *Measured*. Enough
of the path to disambiguate, always.

**The rules it obeys**, all in `core/pedagogy/policy.py`

| | lines | non-comment |
|---|---|---|
| `pedagogy/policy.py` :: `hint_level` 78, `move_for` 30, `style_for` 36 | 144 | 129 |
| `pedagogy/policy.py` :: `check_move` 50, `next_required_move` 20, `support_at_presentation` 29 | 99 | 77 |
| | **243** | **206** |

⚠️ **These figures move when nobody changes any code.** `Tutor` was 89 lines this
morning and is 132 now, entirely from questions and answers written into its
docstring. Re-measure before quoting, and prefer the non-comment column if the
claim is about how much machinery there is.

**Written about it**

| | lines |
|---|---|
| `core/evaluation/tutor.py` | 937 |
| `tests/component/test_lessons.py` | 1542 |
| `tests/component/test_tutor_eval.py` | 742 |
| `tests/component/test_recall.py` | 575 |
| | **3796** |

⚠️ **"Smaller than it looks" is the diagnostic's headline and would be false
here.** The diagnostic proper is 206 lines; this is 723, because `LLMTutor`
carries both halves of the role and `explain` is the larger half.

The rules are listed separately on purpose, and it is the document's central
claim: **they are not the tutor.** `respond` calls `hint_level` and `move_for`
before it writes anything, and `_offer_lesson` calls `style_for` before it calls
`explain`. Moving those into the tutor would make the instructional constraints
advisory.

---

## Inputs

The Protocol has two methods.

⚠️ **Two methods of one object.** The sections below split by method, which can
read as two components; it is one `LLMTutor` (or one `TemplateTutor`) with three
methods — `respond`, `explain`, and the private `_remembered` that `respond`
uses. One constructor, one provider, one recall strategy, shared by both.

### `respond` — a reply to a step

Called on **`INCORRECT` and on `UNPARSEABLE`**, not on `INCORRECT` alone. An
unreadable step still gets a reply; it differs only in that nothing moves —
`is_evidence` is `False`, so no BKT update, no error trace entry, and
`prior_failures` stays where it was. ⚠️ This is the opposite of the diagnostic,
which is called on `INCORRECT` only, and the asymmetry is easy to carry across
wrongly.

```python
def respond(
    self, item, diagnosis, view, domain, *,
    response: str,
    mastery: float,
    prior_failures: int,
    moves_this_item: Sequence[TutorMove],
    said_this_item: Sequence[str] = (),
    explained: bool = False,
) -> Hint
```

| input | where it comes from | why it is a parameter |
|---|---|---|
| `response` | the step just graded | without it the model has only the misconception's description and **invents a step to match** |
| `mastery` | `session.py` :: `_work_item`, read **before** any of this item's answers | read from the view instead, the failure being answered has already lowered it |
| `prior_failures` | readable failures **before** this one | a count including the current step made the mastery baseline unreachable and `nudge` impossible |
| `moves_this_item` | the session's list | so error-first ordering is checkable |
| `said_this_item` | the session's list | so the tutor does not repeat itself |
| `explained` | `bool(shown)` — they were asked how they got there and answered | the error-first rule is satisfied by their own words |

⚠️ **`mastery` and `prior_failures` are supplied rather than read.** Not a loss of
autonomy — the level was never the tutor's to choose — but a question of *when*
each is read, which the tutor cannot know. Reading them for itself put the
current failure into both inputs, and **two entire human sittings came out at
`worked_step` on every turn.**

The session derives `mastery` from its **own arm's view**, so a decoupled tutor
gets the `0.0` its view would have yielded. Nothing here is a channel to state
the arm withholds.

### `explain` — a lesson, between items

```python
def explain(
    self, resource: ConceptResource, style: TeachingStyle,
    exchanges: Sequence[tuple[str, str]] = (),
    closing: bool = False,
) -> str
```

A different question from `respond`, which is why it is a different method.
Every hint comments on an attempt and presupposes the learner *has* the concept;
this presupposes they may not, and there is **no step in front of it** — it is
called between items, from `session.py` :: `_offer_lesson`.

⚠️ **A lesson is not conditional on getting the item wrong**, and it is easy to
read the two methods as the two branches of a verdict. They are not.
`_offer_lesson` is called at the *end* of the loop body, after `_work_item`
returns, whatever the verdict was:

```python
# session.py :: run
self._work_item(item, diagnoses, repetition=repetition)
self._offer_lesson(item.concept_id)
```

⚠️ Nor is it "after the attempts run out". `_work_item` returns whenever the item
*ends* — including on a correct first attempt — and `_offer_lesson` runs on the
way back round. `max_steps_per_item` bounds a failing item; it is not the
lesson's trigger.

There are **three** triggers, and `:why` is only the most visible: the error
trace (`repeated_failure`), the learner asking (`asked`), and the confusion
detector reading their prose (`confusion_in_words`). All three read the board
rather than a view, which is what keeps the arms comparable — see diagram `16`.

⚠️ **And none of them calls the tutor.** `:why` is the clearest case: the demo's
shared reader calls `Blackboard.request_lesson()`, which sets `pending_lesson`
on the board and returns. The session collects it later with
`take_lesson_request()`, inside `_offer_lesson`. Two hops, and the gap between
them is where the rules live — which is also why `:why` can be typed *before* an
answer and still be honoured after it: the request sits on the board across the
whole item.

`store.py` :: `Blackboard.board_for_requests` states the principle it is there to
enforce:

> A front end may record what a person *said* and may not decide what is done
> about it. Asking for a lesson is input; whether one is given, and how it is
> voiced, stays with the session and the rules.

The confusion detector reaches the same place by the same route —
`_note_if_confused` calls `request_lesson(..., inferred=True)`. So a human
front end and a model-backed detector are on identical footing: both write a
request, neither decides anything.

⚠️ **The detector itself is not the tutor** — it writes a request and stops, and
it is not an agent either. It has its own file: [CONFUSION.md](CONFUSION.md).

---|---|
| `resource` | the authored lesson, already checked as plain text and already checked not to answer any item on the concept at any template draw |
| `style` | from `style_for`, not from a prompt — like the level and the move |
| `exchanges` | the conversation so far, oldest first, as `(what the tutor said, what the learner said back)`. Empty on the opening turn. |
| `closing` | this is the last turn: answer what is hanging and ask nothing new |

⚠️ **`style` has two sources and the learner's wins.** `style_for(taught,
chosen=board.teaching_style)` rotates `plain → socratic → real_world` on the
count of lessons already given, *unless* a preference was stated. In the demo
`_ask_how_to_explain` runs once at the start of the sitting and calls
`board.record_teaching_style(chosen)` — the same board-mediated pattern as
`request_lesson`: the front end records what the person said, the rules decide
what follows from it.

⚠️ And what it changes is narrower than it sounds, which the demo says out loud
to the learner: *"This changes the wording, never the mathematics."* The content
is identical under all three styles — written by a person, checked against every
question in every bank. Only the voicing moves. ⚠️ It must also stay out of every
cohort, on the same footing as `Emphasis`: choosing your own account changes what
the tutor gives you.

⚠️ **`exchanges` is handed over, never fetched** — the same rule as
`said_this_item` on `respond`. An agent is *told* what was said; it is not given
a channel to another agent. The learner's words reach here through the session
and the board like everything else.

---

## How it works

**1. The rules choose, before the model is called.**

```python
# llm.py :: LLMTutor.respond
level = hint_level(mastery, prior_failures, self._band, policy=self._policy)
move  = move_for(level, moves_this_item, misconception_confirmed=diagnosis.named, ...)
```

`hint_level` under `banded_plus`:

| region | base | escalation ceiling |
|---|---|---|
| `P >= theta_upper` | `none` | `none` — reflect only |
| `theta_lower < P < theta_upper` | `nudge` | **`targeted`** — never the step |
| `theta_lower/2 < P <= theta_lower` | `targeted` | `worked_step` |
| `P <= theta_lower/2` | `worked_step` | `worked_step` |

`move_for` returns `REFLECT` outright at `none`, otherwise `REFLECT` if the
error-first rule requires it, else `REMEDIATE` when a misconception is confirmed,
else `HINT`.

**2. The level becomes an instruction**, not a suggestion (`llm.py` :: `LLMTutor.respond`). Length
budget included — it belongs to the level, because "at most two sentences"
globally made a worked step impossible and the model restated the rule in the
abstract instead.

**3. The prompt is assembled** — exercise, the student's step, the correct
answer, the misconception's description if one was named, what the learner has
said that bears on the question, and what this tutor already said on this one.

⚠️ **How that third thing is found depends on the run.** Without a recall
strategy it is `view.said_about(concept_id)` — the last two things said about
*this* concept, which is what every measured result was produced under and stays
the default. With one, the whole history is ranked against the question and the
answer together, so a learner who asked what a gradient was while working on
limits has said something that bears on the power rule.

⚠️ And it is read **inside the `FullStateView` branch**, which is the whole of
how the ablation survives recall. Reaching those words through the session or
the store would hand the decoupled arm what it is defined by lacking, and it
would look like the coupling advantage growing rather than like a leak. There is
a test giving a decoupled tutor a recall strategy that raises if called. See
diagram `03-…-v4`.

⚠️ A dead embedder costs the context and not the sitting: `_remembered` catches
`ProviderError`, falls back to `said_about`, and counts it in
`self.recall_failures`. The count exists because a run that quietly stopped
recalling and one where nothing happened to match are otherwise
indistinguishable.

**4. The model writes the words.** `TemplateTutor` instead looks the text up in
the catalogue and **ignores `response` and `said_this_item` entirely**, with a
test asserting its output cannot vary with them — the cohorts run it, so a hint
whose wording moved with the answer would make every measured number depend on
something outside the manipulation.

**5. `check_move` checks the result against the rules** and any disagreement is
**recorded as a violation, not raised**. The tutor is driven by those rules so it
should not fire; one bad turn must not abort a cohort, and must not pass
unnoticed either. ⚠️ `EXPLAIN` returns early — the error-first ordering is about
replies to a step, and a lesson is not one.

### `explain` — the lesson path

1. **`style_for` chooses the account** before the tutor is called: the learner's
   own choice if they stated one, otherwise the next in `plain → socratic →
   real_world`. A learner who did not understand the plain account is unlikely
   to be helped by the plain account again.
2. **Which instruction is used depends on the turn**, not on a global prompt:
   `_STYLE_OPENING[style]` with no exchanges, `_STYLE_REPLY` with them,
   `_STYLE_CLOSING` when `closing`.
3. **The schema depends on the turn too.** `LessonReply` refuses a turn that
   stops mid-sentence; `ClosingReply` additionally refuses one that ends on a
   question. Both go through `complete()`'s repair loop, which shows the model
   its own reply and asks again.
4. **Two guards on what comes back**, and either sends it to the authored text:
   the plain-text check (the same pattern the content is checked against, so a
   fix to one is a fix to both) and a length cap of **3× the authored account**
   it is re-voicing — a reply that long has stopped re-voicing and started
   writing, which is the thing the design exists to prevent.
5. `TemplateTutor.explain` returns `resource.lesson()` and **ignores style,
   exchanges and closing entirely** — same reasoning as `respond`, with a test
   on it. Not a degraded lesson: `PLAIN` *is* the authored text.

⚠️ **`explain` is called in a loop, not once.** `_offer_lesson` calls it for the
opening, then again for every reply, then once more for the closing turn — so a
lesson of *n* exchanges is *n + 2* calls, each one re-sent the whole conversation
so far. The loop's only exit is the learner:

```python
while bound is None or len(exchanges) < bound:
    replied = self.learner.discuss(concept_id, said)
    if not replied:
        break
```

Two ways to stop, and both are the same thing to the code. A **blank line**
returns `""`, which is falsy. `:done` is routed through the demo's shared reader
and also returns `""`. ⚠️ The redundancy is deliberate — `END_LESSON`'s own note
says enter is the idiom every optional prompt here already uses and is what most
people will do, and `:done` is for someone who would rather say so than guess:
*a stated affordance that works is worth more than a discovered one that also
works.*

⚠️ `lesson_turns` is `None` in the demo, so the `bound` is not what ends
anything. It never was: a turn needs a reply and a reply needs someone to type
one, so the conversation already stops the moment nobody answers. It was 3, then
12, and a sitting reached both while still engaged.

---

## Output

```python
Hint(text, move, level, targets)
```

⚠️ **`targets` is the only part that reaches a simulated learner.**
`session.py` :: `_work_item` calls `learner.receive_hint(hint.targets)` on `REMEDIATE` and
nothing else. The prose is ignored entirely.

Three consequences the whole study rests on:

- a **misdiagnosis** costs something — the hint is aimed at a misconception the
  learner does not hold, and does no work;
- **wording** costs nothing measurable, which is why cohorts can run
  `TemplateTutor` without the comparison depending on model quality;
- the **support level cannot move a cohort number**, which is what made the
  banded-scaffolding work provably inert.

A reflective turn targets `None` deliberately: it costs a step and teaches
nothing, which is what makes error-first a real constraint rather than a free
one.

`explain` returns **prose, not a `Hint`**, and that is load-bearing rather than
incidental. A lesson carries no `targets` and never reaches `receive_hint`:
`remediation_ratio` is the declared primary outcome and it counts what a hint
aimed at, so a target here would credit a lesson with remediation it did not do.
A lesson explains a concept; it does not correct a misconception.

⚠️ **The tutor writes nothing at all, and the summary never passes through it.**
Both methods *return* a value; `session.py` :: `_work_item` and `_offer_lesson`
are what record it, through `board.record_turn`. And the authored summary that
closes a lesson is not the tutor's output in any sense — the session calls
`resource.lesson()` **directly**, bypassing `explain` entirely:

```python
# session.py :: _offer_lesson — after the conversation, however it ended
_say(resource.lesson(), level="summary")
```

That is the authored/generated boundary in one line. Everything `explain`
produced is the model's and is re-checked against nothing; the thing the learner
is left holding came from outside the agent and was validated. ⚠️ Note the order
of the writes: the **blackboard** holds every turn, and the store's `turn` table
is the blackboard persisted — not a second destination the tutor knows about.

⚠️ **A front end sees three hooks, not one**, and the split carries the same
boundary: `lesson_offered` for the opening and every reply,
`lesson_reply_recorded` for what the learner said back, and `lesson_summary` for
the authored account. The summary has its own hook deliberately — *a front end
should be able to mark the account the learner keeps differently from the talking
that led to it.* One is theirs to take away; the other was a conversation.

⚠️ Related, and deliberately not changed: a lesson is recorded with
`INSTRUCTION_CAUSE = observation`, so it is invisible to `dose_by_concept` and
`dose_on_gap`. Lessons are counted **beside** the dose, never inside it — folding
them in would move a figure every sitting so far was read under.

On a provider failure the turn falls back to `FALLBACK_HINT` rather than ending
the session — the *targeting* survives, which is the part that affects learning.

---
