# Run artifacts

One directory per run: `<UTC timestamp>_<run name>_<arm>_<config hash>/`.

| File | Committed | Why |
|---|---|---|
| `manifest.json` | yes | Provenance. Small, and the only way to know what produced a number. |
| `metrics.json` | yes | Aggregated outcomes the analysis stage reads. |
| `events.jsonl` | yes | Structured event log — arbitration decisions, replanning triggers, state transitions. This is the audit log. |
| `llm_calls.jsonl` | **no** | Raw prompt/response pairs. Hundreds of MB per cohort, and regenerable from the response cache. |
| `raw/` | **no** | Per-learner transcripts. Same reasoning. |

Figures generated from these live in `results/figures/` and are committed, so
reported numbers and the artifacts they came from stay in step.

## Pooling

`agent_newton.manifest.assert_poolable` refuses to aggregate runs whose domain,
concept graph, misconception catalogue or item-bank hashes differ. If it raises,
the fix is to re-run the older arm against the current domain content — not to
bypass the check.
