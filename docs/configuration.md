# Configuration reference

Run configurations are YAML under `experiments/configs/`. A run is reproducible
from `(config file, seed, git SHA)`.

```bash
uv run agent-newton config-check experiments/configs/smoke.yaml
```

Validation happens at load, so a bad configuration fails immediately rather than
part-way through a long run.

## Top level

| Key | Default | Meaning |
|---|---|---|
| `run_name` | `unnamed` | Label used in the run directory name |
| `domain` | `toy_algebra` | Registered domain to load |
| `arm` | `coupled` | Which state view the planner receives |
| `seed` | `20260807` | Root seed for learner profiles and all sampling |

## `agents`

Each role takes an `impl` plus a provider and model. Model-free implementations
are run conditions, not test doubles.

| Role | `impl` options |
|---|---|
| `tutor` | `llm`, `template` |
| `diagnostic` | `llm`, `oracle`, `noised_oracle` |
| `planner` | `llm`, `deterministic`, `oracle` |

- `template` — hints drawn from the misconception catalogue at the level the
  scaffolding predicate requests.
- `oracle` (diagnostic) — reads the simulator's injected label directly; perfect
  classification.
- `noised_oracle` — the injected label corrupted at `noise_rate`. Setting
  `noise_rate: 0.0` is rejected, since that is just an oracle.
- `deterministic` (planner) — fixed policy over the frontier.
- `oracle` (planner) — may additionally see the simulator's true profile.

```yaml
agents:
  tutor:      { impl: llm, provider: ollama, model: gemma4:12b }
  diagnostic: { impl: llm, provider: ollama, model: gemma4:12b }
  planner:    { impl: deterministic }
```

`Config.uses_llm()` is `False` only when no role and no simulator surface uses a
model.

## `simulator`

| Key | Default | Meaning |
|---|---|---|
| `surface` | `symbolic` | `symbolic` uses the rule engine's decision verbatim; `llm` renders it as prose |
| `surface_model` | `ollama/gpt-oss:20b` | Renderer, used only when `surface: llm` |
| `misconceptions_per_learner` | `2` | Drawn per learner |
| `p_fire_range` | `[0.6, 0.9]` | Initial firing probability range |
| `remediation_factor` | `0.55` | Multiplier applied when a hint correctly targets a misconception |

### Model-lineage separation

When `surface: llm` **and** `diagnostic.impl: llm`, the two models must come
from different model families. A configuration violating this is rejected:

```
Circularity control violated: the simulator surface model (gemma4:12b) and the
diagnostic agent (gemma4:12b) are both from family 'google'.
```

The check does not apply when the simulator is symbolic (no model generates the
step) or the diagnostic is an oracle (no classification is inferred).

Families are inferred from model-name prefixes. Unrecognised names map to
`unknown:<name>` rather than a shared bucket, so two unknown models never
compare equal by accident.

## `bkt`

Bayesian Knowledge Tracing parameters.

| Key | Default |
|---|---|
| `p_init` | `0.15` |
| `p_transit` | `0.20` |
| `p_guess` | `0.20` |
| `p_slip` | `0.10` |

`p_guess + p_slip >= 1` is rejected: the model degenerates and evidence updates
run backwards, so a correct answer would *lower* the mastery estimate.

## `zpd`

| Key | Default | Meaning |
|---|---|---|
| `theta_lower` | `0.70` | A prerequisite counts as met above this |
| `theta_upper` | `0.90` | A concept counts as mastered above this |

`theta_lower >= theta_upper` is rejected — the band would be empty.

## `arbitration`

| Key | Default | Meaning |
|---|---|---|
| `theta` | `0.15` | Mastery-change threshold that triggers replanning |
| `k_repeats` | `2` | Repeats of one misconception in the window that force a replan |
| `min_items_between_replans` | `2` | Rate limit against thrashing |
| `error_trace_length` | `20` | Rolling trace length held in state |

## `cohort`

| Key | Default |
|---|---|
| `n_learners` | `3` |
| `max_items` | `20` |
| `max_steps_per_item` | `3` |

## `paths`

| Key | Default |
|---|---|
| `results_dir` | `results` |
| `cache_dir` | `.cache/llm` |

## Shipped configurations

| File | Purpose |
|---|---|
| `smoke.yaml` | Three learners on `toy_algebra`, no model anywhere. Completes in seconds. |
