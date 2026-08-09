# Working in this repository

## Orientation

Read `docs/architecture.md` first. `core/orchestration/session.py` is the best
single file for understanding control flow.

If `research_private/` is present, read `research_private/HANDOVER.md` before
anything else — it carries the design rationale, the open decisions, and the
state of play. That folder is gitignored and will not exist in a fresh clone.

## Commands

```bash
uv sync --dev
uv run pytest -q
uv run pyright
uv run agent-newton domain validate all
```

Both `pytest` and `pyright` gate CI. Run them before committing — and chain off
the command itself, not off a pipe, or a failure will not stop the commit.

## Two conventions that are not obvious from the code

**Commit messages** carry `Authored and reviewed by Victor Hristov.` as its own
line, immediately above any `Co-Authored-By:` trailer. The repository accompanies
a written document, so authorship in the permanent record has to be unambiguous.

**Tracked files must not contain prose written for that document.** The
repository is published; text that also appears in the submitted document would
exist publicly before submission. `tests/test_publishable.py` enforces this on
every tracked file and runs in CI.

The rule is about *prose*, not knowledge. Technical vocabulary, algorithm
attributions beside their implementations, and citation strings in the
misconception catalogue's `source` fields are all fine — the last are required
data provenance. Motivation, argument and the vocabulary specific to the
document are not. When in doubt, describe what the code does rather than why it
was built, and put the why in `research_private/`.

## Design invariants

These are enforced by tests rather than convention. Breaking one should mean
changing the test deliberately, not working around it.

- **`core/` never imports a concrete domain.** It is generic over the five
  Protocols in `domains/base.py` and receives a `Domain` as a parameter.
- **Agents never call one another.** All coordination passes through the shared
  state, so nothing can happen that the audit log does not record.
- **`UNPARSEABLE` is not a verdict about the learner.** It means the verifier
  could not measure. It updates no estimate and enters no error trace, but it is
  counted — a rising rate means the verifier is failing, not the learner.
- **Only agents implementing `OracleAccess` receive ground truth.** A
  model-backed agent must never satisfy it.
- **Guards are themselves tested.** Each mechanical check has a test proving it
  can fail. A guard that cannot fail proves nothing.
