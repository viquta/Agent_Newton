# Diagnostic

Names which misconception produced a wrong step. What it is handed, how it works, and what it produces.

Part of the component reference — [architecture.md](../architecture.md) is
the map, and [the index](README.md) lists the rest.

Given a wrong answer, names **which** misconception produced it.

`core/agents/diagnostic.py` (oracle, noised oracle) ·
`core/agents/llm.py` (`LLMDiagnostic`, `_offered`, `_describe`) ·
`core/agents/schemas.py` (the constrained reply)

---

## Scope — smaller than it looks

The agent proper is about **205 lines across four files**:

| | lines |
|---|---|
| `agents/base.py` :: `Diagnostic` — the Protocol, one method | 4 |
| `agents/base.py` :: `Diagnosis` / `OracleAccess` | 9 / 11 |
| `agents/diagnostic.py` :: `OracleDiagnostic` | 11 |
| `agents/diagnostic.py` :: `NoisedOracleDiagnostic` | 68 |
| `llm.py` :: `LLMDiagnostic` + `_offered` + `_describe` | 58 + 19 + 4 |
| `schemas.py` :: `diagnosis_schema` | 22 |

⚠️ **`agents/`, not `diagnostic.py`.** There are two — `core/agents/` holds the
implementations, `core/evaluation/` the offline evaluation — and this document
uses both, one row apart. A bare `diagnostic.py` does not resolve. The same trap
is in `base.py` (four of them) and `policy.py` (two).

Everything else that mentions the role is **about** the agent rather than part
of it, and the distinction is worth holding on to when reading:

- `core/evaluation/diagnostic.py` — 203 lines, the offline evaluation. Almost
  exactly the size of the agent it measures.
- `experiments/run_propagation.py` — 257 lines, the A/B/C study that sweeps
  diagnostic error into outcomes.
- `session.py` — the call site and the `OracleAccess` guard. That code belongs
  to the session; the agent does not know it exists.
- `config.py`, `cli.py` — the `impl` choice, `label_space`, the
  `evaluate diagnostic` verb.
- three test files — 636 lines in `test_llm_agents.py` alone, and 257 in
  `test_no_back_channel.py`.

So roughly 205 lines of agent sit inside about 1,350 lines written about it.
The ratio is deliberate. The two model-free implementations are not
test doubles but the **experimental conditions** — `oracle` is the
perfect-diagnosis upper bound and `noised_oracle` the dose-response curve — and
the measurement harness is bigger than the thing measured because the question
being asked is *how far diagnostic error reaches system outcome*, which needs the
agent scored independently before its errors can be attributed to anything
downstream.

---

## Inputs

```python
def diagnose(self, item: Item, response: str, domain: Domain) -> Diagnosis
```

⚠️ **No `StateView`.** Of the three agent roles this is the one that receives no
view at all — it is constrained by its parameter list, not by a window onto
state. It sees the question, what the learner wrote, and the catalogue. Nothing
about mastery, nothing about history.

⚠️ **And no verdict.** It is called only after an `INCORRECT` one, and receives
no trace of it. `session.py` :: `_work_item` holds the `VerificationResult` in a
local and simply does not make the call unless the verdict is `INCORRECT`, so
*being wrong* is carried by control flow rather than by data. There is no
parameter it could arrive in.

The sharpest form of that: **the verifier and the diagnostic are handed exactly
the same two things.** `item` and `response` are the same locals, passed to one
and then to the other; nothing travels between them. One returns a verdict, the
other a label, and the diagnostic could not tell you whether the answer was
right — it is never asked, and has no way to find out. Call `diagnose` on a
correct answer and it will name a misconception perfectly happily, which is why
the guard has to live in the session and cannot live in the agent.

⚠️ **The working is not an input either.** Between verify and diagnose the
session calls `show_working(required=True)` — the learner is asked to account
for the step *before* the verdict is shown and before the diagnostic sees it.
Told they were wrong first, a person writes an account of an error they now know
about rather than the reasoning they actually used. What they write is recorded
as an `Utterance(kind="working")` and reaches the **tutor** through the view. It
never reaches here.

**`UNPARSEABLE` never arrives**, and that follows from what the verifier is. It
is a symbolic equivalence check, not a string match, so `5x^4` and `5*x**4` are
the same answer while `dy/dx = -x/y` comes back unreadable — an equals sign is a
syntax error to it. `UNPARSEABLE` means *could not measure*, which is a fact
about the verifier rather than about the learner, and there is nothing in it for
a diagnosis to explain.

The oracle variants receive one thing more, through a separate protocol:

```python
@runtime_checkable
class OracleAccess(Protocol):
    def observe_ground_truth(self, label: str | None) -> None: ...
```

```python
# session.py :: _work_item
if isinstance(self.diagnostic, OracleAccess):
    self.diagnostic.observe_ground_truth(step.fired)
```

**`LLMDiagnostic` deliberately does not implement it.** There is no method by
which the injected label could reach it, and three tests hold that line.

⚠️ In a **human sitting** that branch is present and never taken twice over:
`demo.yaml` forces `impl: llm`, and a person has no injected label anyway —
`HumanLearner.answer` returns `fired=None`. See diagram `21-human-diagnosis`.

---

## How it works


### `OracleDiagnostic`

Returns the injected label at confidence 1.0. Perfect by construction.

### `NoisedOracleDiagnostic`

Corrupts the label at a fixed rate, keyed on
`(seed, item, response, label, **occurrence**)` — a hash, not a running
generator, so a learner meeting the same situation for the nth time is
misdiagnosed the same way in **both** arms. Otherwise the arms would differ by
their noise streams as well as by their tutoring.

⚠️ The occurrence counter was missing once, and a session revisits few situations
many times — 331 diagnoses across 12 labels, one of them 169 times. A repeated
situation replayed its own verdict and a nominal rate of 0.10 realised as 0.69.
**Compare conditions on the realised accuracy each run reports, never on the
nominal rate.**

### `LLMDiagnostic`

1. `_offered(domain, item, label_space)` fixes the choices — under
   `label_space: concept`, only the misconceptions belonging to **the item's own
   concept**, plus `unknown`. (**vh_comment:** just thought of a cool idea! --> since when think = true, for the ones it does give a tag, it is (alledgedly) 100%, but it can only give that to 89% of the known math problems, the rest it gives unknown, regardless of how much it gets to think. Well, what if this threshold can actually lead to a :why with the student to check what it is that they did. Or maybe, it could be a :deeper_diagnose or something like that which the tutor or another agent can pick up on. But this can be written in the discussion as an upgrade to the system. It is already working well enough as a proof of concept)
2. `diagnosis_schema(...)` builds a JSON schema over exactly those ids, passed to (**vh_comment:** this means that I shouldn't be promoting putting other API keys like openai or anthropic, since they might use a different schema. Also, it could be that other AIs interpret the json schema differently, qwen vs gemma for example... )
   Ollama's `format`, so an invalid label is **impossible by constrained
   decoding** rather than merely discouraged.
3. The prompt carries the exercise, the correct answer, the student's step and
   the offered catalogue with descriptions.
4. A `ProviderError` is counted in `self.failures` and returns `Diagnosis(None)`.

⚠️ **What `_describe` puts in the prompt is `- id: description`, and nothing
else.** `concept_id` is a *filter* — it is what `_offered` selects on — and
`source` is provenance for a human reader. Neither reaches the model. Under
`concept` that is immaterial, since every offered entry shares the concept; under
`catalogue` it means the agent is shown sixteen labels with **no indication of
which concept any of them belongs to**, and the information that would let it
rule fifteen out on subject matter alone is present in the YAML and absent from
the prompt. Worth stating when the wide-space figure is quoted: that condition's
failure mode is partly a property of the condition, not purely of the model.

⚠️ **`_offered` is a module-level helper, not a method.** It takes `domain` and
`item` from `diagnose`'s own parameters and `label_space` from the constructor —
configuration, fixed for the run. Nothing runtime-ish arrives through it.

---

## Output

```python
Diagnosis(misconception_id: str | None, confidence: float)
```

`named` is `misconception_id is not None`. `unknown` from the model becomes
`None` — absence means nothing was inferred, which is distinct from a label
literally called "unknown".

It reaches two places:

- the **tutor**, as the thing a `REMEDIATE` turn will target;
- the **error trace**, via `record_observation(misconception_label=...)`, which
  is what `misconception_repeat` counts and what `CONSOLIDATE` ranks by.

⚠️ It is **returned to the session**, which writes it. The agent writes nothing,
like every other agent here.

⚠️ **The injected label and the inferred label meet at exactly one place**: the
accuracy measurement. Nowhere else. That is what makes diagnostic accuracy have
consequences for learning outcomes rather than being reported beside them — a
misdiagnosis produces a misaimed hint, and a misaimed hint does no work.

---
