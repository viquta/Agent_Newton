# The domain interface

The subject matter is a plug-in. `core/` is generic over five Protocols and
receives a `Domain` as a parameter; it never imports a concrete domain.

`toy_algebra` is the reference implementation — deliberately small, and used as
the fast test fixture. `calculus` is the primary domain.

Two domains ship from the start, and they verify answers by deliberately
different means: `calculus` by symbolic equivalence in sympy, `toy_algebra` by
canonicalised term comparison with no algebra system at all. A programming
domain would verify by running unit tests. Keeping the two implementations
dissimilar is what keeps the interface honest — an abstraction exercised by only
one implementation tends to leak that implementation's assumptions.

## The five members

| Member | Supplied as | Notes |
|---|---|---|
| `ConceptGraph` | `concepts.yaml` | Prerequisite DAG; also the substrate the mastery frontier is computed over |
| `MisconceptionCatalogue` | `misconceptions.yaml` | The shared label space |
| `ItemBank` | `items/*.yaml` | Practice, pre-test, post-test |
| `Verifier` | Python | Correctness, independent of any model |
| `BuggyRule` | Python | How the simulator errs — one per misconception |

Three of the five are pure content, so adding items or concepts needs no Python.

### The shared label space

The simulator's buggy rules and the diagnostic agent's classification targets
are drawn from **the same catalogue**. That identity is what allows the
diagnostic agent's output to be scored against the label the simulator actually
injected.

Every entry requires a `source` field citing the literature the error is
documented in. `YamlMisconceptionCatalogue` refuses to load an entry without
one, so the catalogue stays traceable rather than accumulating invented errors.

### Responses are strings

Student responses cross the boundary as `str`; domains parse their own notation
internally. This keeps responses directly serialisable into the audit log, which
a generic response type would not.

### `Verifier`

```python
def verify(self, item: Item, response: str) -> VerificationResult: ...
```

Called by the orchestrator after **every** student step — never as a tool an LLM
elects to invoke. Correctness labels therefore do not depend on model quality,
which is what allows a weak local model to be used for the agents without
compromising the correctness signal.

Three verdicts, and the third matters:

- `CORRECT` / `INCORRECT` — evidence; the learner model updates.
- `UNPARSEABLE` — a measurement failure, not evidence about the learner.
  `VerificationResult.is_evidence` is `False` and the learner model must not
  update.

Verifiers must terminate. A CAS-backed implementation needs an explicit timeout;
returning `UNPARSEABLE` is better than hanging a long cohort run.

### `BuggyRule`

```python
@property
def misconception_id(self) -> str: ...
def apply(self, item: Item) -> str | None: ...
```

Deterministic — the same item must always produce the same wrong response, or a
seeded run is not reproducible. Returns `None` when the item cannot elicit this
misconception.

Rules compute from `item.params`, the item's structured form, rather than
parsing `item.prompt`. Rewording a prompt then never changes learner behaviour,
and no rule is coupled to prompt wording.

## Adding a domain

1. Create `src/agent_newton/domains/<name>/` with `concepts.yaml`,
   `misconceptions.yaml` and `items/`.
2. Implement a verifier and one buggy rule per misconception.
3. Export `build() -> Domain`.
4. Register it in `domains/registry.py`. Builders are imported lazily, so a
   `toy_algebra` run never pays for importing sympy.
5. Run the validator.

## Extending an existing domain

| Adding | How |
|---|---|
| Items | Append to the YAML — pure data |
| Concepts | Edit `concepts.yaml` and add items; acyclicity is checked at load |
| Misconceptions | Catalogue entry plus one buggy rule |
| Whole topics | Extend the graph, or add a sibling domain |

> **Adding a misconception changes the diagnostic agent's label space**, so
> accuracy figures measured before the change are not comparable with figures
> measured after. Run manifests record a catalogue hash and `assert_poolable`
> refuses to aggregate runs whose hashes differ. If it raises, re-run the older
> configuration against the current content rather than bypassing the check.

## Validation

```bash
uv run agent-newton domain validate toy_algebra
```

`all` checks every registered domain. Coverage:

- **Referential integrity** — every misconception maps to a real concept, every
  item references a real concept, every probed misconception exists, every
  misconception has a registered buggy rule and a non-empty source.
- **Coverage** — every concept has practice items; every misconception is probed
  by at least one pre-test *and* one post-test item.
- **Held-out separation** — no test item repeats a practice prompt verbatim.

Two checks carry most of the weight, because they test content against the
verifier rather than against itself:

- **`answers_verify`** — every item's stated answer is judged correct.
- **`rules_produce_errors`** — every misconception an item claims to probe
  actually yields a response the verifier judges *incorrect*.

The second catches an otherwise invisible failure: a mis-written buggy transform
that produces the *correct* answer. That misconception could never be observed,
the diagnostic agent would carry a label nothing ever triggers, and measured
accuracy would be computed against a silently broken instrument.
