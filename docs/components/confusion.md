# Confusion detector

Reads the learner's words for "i do not know what this is". What it is handed, how it works, and what it produces.

Part of the component reference — [architecture.md](../architecture.md) is
the map, and [the index](README.md) lists the rest.

Reads one thing the learner wrote and answers one question: do they say they do
not know what the concept **is**?

`core/agents/base.py` (the Protocol) · `core/agents/llm.py`
(`LLMConfusionDetector`, `_CONFUSION_SYSTEM`) · `core/agents/tutor.py`
(`NoConfusion`) · `core/agents/schemas.py` (`ConfusionReply`)

⚠️ **Not an agent**, and the Protocol says so itself: *"it plans nothing,
teaches nothing and holds no view of the learner model — it classifies one
string, the way the verifier classifies one answer."* It sits with
`ARBITRATION.md` and `VERIFIER.md` rather than with the three agent files.

---

## Scope

The smallest component here by some distance.

| | lines | non-comment |
|---|---|---|
| `agents/base.py` :: `ConfusionDetector` — a Protocol, **never executes** | 34 | 23 |
| `llm.py` :: `LLMConfusionDetector` | 56 | 43 |
| `llm.py` :: `_CONFUSION_SYSTEM` — the instruction | 17 | 17 |
| `agents/tutor.py` :: `NoConfusion` — the null object every cohort runs | 15 | 12 |
| `schemas.py` :: `ConfusionReply` — a pydantic model, built per check | 25 | 22 |
| | **147** | **117** |

Against the diagnostic's 206 and the tutor's 723. ⚠️ `ConfusionDetector` is a
Protocol and `ConfusionReply` is not — the first never runs, the second is
constructed on every check. Four files carry the name `base.py`; this is
`core/agents/`.

---

## Inputs

```python
def confused(self, concept_id: str, text: str) -> str | None
```

Two strings. The concept being worked, and one thing the learner wrote — the
working shown after a failed step, or a reflection.

⚠️ **No view, no state, no history**, on the same footing as the diagnostic and
for the same reason: it is constrained by its parameter list. It cannot know
whether this learner has said the same thing before, and does not need to — the
*count* of such remarks lives on the board, and `_offer_lesson` reads it there.

Called from `session.py` :: `_note_if_confused`, which is reached from the
working channel and the reflection channel. It returns early when
`teaching.detect_confusion` is off, which is every cohort.

---

## How it works

1. Empty text returns `None` without a call — a blank prompt is a refusal, not
   a statement about the concept.
2. `checks += 1`, then one constrained call with `ConfusionReply` as the format.
3. `confused is False` → `None`. Otherwise the **quote** is returned.
4. `ProviderError` → `None`, logged. ⚠️ Not knowing is not the same as "no", but
   it has to become one somewhere; the fallback costs a lesson that would have
   been offered rather than giving one that should not have been.

### The instruction spends its length on the false cases

`_CONFUSION_SYSTEM` is mostly a list of things that are **not** confusion:

| the learner… | verdict |
|---|---|
| attempts the work and gets it wrong | false |
| uses a wrong method confidently | false |
| is not sure their answer is right | false |
| hedges — *"I think"*, *"not totally sure"*, *"maybe"* | false |
| says a step was hard, or that they found it confusing | false |

> *"Someone who describes a method, even a wrong one, has met the concept. Being
> unsure of an answer is not the same as not knowing what the question is about,
> and a student who says both is doing the work."*

⚠️ `ConfusionReply`'s own docstring draws the line the design rests on: it is
**not** asked whether the learner is struggling, frustrated, or needs help —
those are judgements about a person. It is asked one question about a piece of
text. *Someone who differentiates wrongly has met the concept and slipped;
someone who writes "I don't understand what a limit is" has not met it, and no
amount of correcting the slip will help them.*

---

## Output

```python
str | None
```

The words that say so, copied exactly, or nothing. It is the **third** of the
three things that can buy a lesson — see diagram `16` — alongside an explicit
`:why` and the error trace. All three reach `_offer_lesson` through
`pending_lesson` on the board.

⚠️ An inferred request **bypasses the difficulty threshold and not the account
ceiling**. A person asking again has decided they want it again; an inference
firing repeatedly would re-teach the same three accounts round and round.

---

## Seeing it

Every cohort runs `NoConfusion`, which says no to everything, so nothing here
reaches a measured number. `detect_confusion` is off by default, off in every
cohort config with a directory scan and a can-fail test, and **refused outright
without a model-backed tutor** — rather than silently detecting nothing while
reporting itself on, which is the failure shape the human-diagnostic check
already guards.
