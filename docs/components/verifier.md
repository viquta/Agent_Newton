# Verifier

Decides correct / incorrect / unreadable. What it is handed, how it works, and what it produces.

Part of the component reference — [architecture.md](../architecture.md) is
the map, and [the index](README.md) lists the rest.

Decides `CORRECT` / `INCORRECT` / `UNPARSEABLE`, symbolically, with **no model
involved**.

`domains/calculus/verifier.py` · Protocol in `domains/base.py` :: `Verifier` ·
gold set at `tests/fixtures/gold/calculus_verifier_cases.yaml`

⚠️ **Not an agent.** It lives in `domains/`, not `core/agents/`, takes no view,
and is not a participant in the blackboard. It is *ground truth* — the thing
every other measurement inherits.

---

## Scope

**In:** whether one response to one item is right, wrong, or unreadable.

**Out:** any judgement about the learner. `UNPARSEABLE` is the boundary made
explicit — it says *the verifier could not measure*, not *the learner failed*, so
it updates no estimate, enters no error trace and costs no attempt. It is counted
and audited, because a rising rate is a fact about the instrument.

Also out: **form**. The verifier compares for equivalence, so it cannot see that
an answer restates the question, and it cannot decide between two standard
readings of ambiguous notation — where those disagree it returns `UNPARSEABLE`
rather than picking one silently.

---

## Inputs

```python
def verify(self, item: Item, response: str) -> VerificationResult
```

Called by the orchestrator after **every** student step (`session.py` :: `_work_item`) —
never as a tool a model elects to invoke.

---

## How it works

**Parsing** (`parse`). sympy with implicit multiplication and `^ → **`.
The namespace is closed explicitly — no builtins, no import machinery — because a
response is untrusted input and `parse_expr` evaluates what it parses.

Three ways a response fails to parse, all `UNPARSEABLE`:

- it does not parse at all;
- it is not an `Expr` — a relation like `x > 2` parses to a Boolean, which
  subtracted from an expression downstream yields nonsense rather than a verdict;
- it contains an **unknown symbol**. Implicit multiplication will happily read
  `no idea` as `no*idea`, inventing a symbol per word, and prose must come back
  unreadable rather than wrong.

**Equivalence**, three stages, cheapest first (`_equivalent`):

1. **structural** — sympy's own equality, instant;
2. **numeric screen** — evaluate both at 12 random irrational-ish points; one
   clear disagreement *proves* inequivalence, so most wrong answers are rejected
   in microseconds without ever calling `simplify`;
3. **symbolic confirmation** — `simplify`, under a hard 2 s timeout.

On a timeout, agreement at ≥ 4 evaluable points is accepted as equivalent and
recorded in `detail`. Below that, `VerificationUnavailable` → `UNPARSEABLE`:
reporting it correct would score an answer never checked, and incorrect would
blame the learner for our own failure to measure.

**Two special cases:**

- `params["up_to_constant"]` — an antiderivative is determined only up to a
  constant, so the two sides are compared by their derivatives. Opt-in per item,
  and deliberately **not** set on the antiderivative items, where the constant is
  the whole point.
- **notation with two readings** — `a/bc` means `(a/b)*c` by formal precedence and
  `a/(bc)` in ordinary writing. See below.

---

## ⚠️ Ambiguous notation, and the three-way rule

A sitting wrote `-x/3y` for an item whose answer is `-x/(3*y)`, was told it was
wrong, wrote `-2x/6y`, and was told again. Neither reading is wrong and the
verifier was silently taking the first.

| the two readings | verdict |
|---|---|
| both correct | `CORRECT` |
| both wrong | `INCORRECT` — the notation is not the reason |
| **they disagree** | **`UNPARSEABLE`** — the verifier cannot say what was meant |

Only the third case is new, and it is reachable **only from `INCORRECT`**, so a
correct answer can never be changed by it and a genuine error is never lost.

⚠️ The message names both readings and says which is right about **neither**. This
verdict costs no attempt, so saying would be the answer handed over for free.

**Provably inert for every measured result:** no buggy rule can write an
ambiguous string — 2366 item answers and rule outputs over 64 template draws,
zero flagged, asserted as a test. And `domain validate` refuses an *item* answer
that is ambiguous, through an optional `ambiguous_notation` hook on the verifier.

---

## Output

```python
VerificationResult(verdict: Verdict, correct_answer: str, detail: str = "")
```

`is_correct`, and `is_evidence` — the property that governs whether the learner
model may update at all. `detail` carries the reason, and the demo shows it to
the learner on an unreadable answer.

---
