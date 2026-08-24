# Running in a container

Everything here runs from a single image: the test suite, every experiment, the
component evaluations, the technical reference, and the session at the keyboard.
The only requirement on the host is Docker.

```bash
./newton help
```

The first call builds the image, which takes a few minutes. Every call after it
starts in about a second.

`make` targets mirror the verbs — `make test`, `make paired`, `make arch
DOC=pedagogy` — and delegate to the same script.

## Without the wrapper

`./newton` is a shell script wrapping `docker compose`. Compose reaches every
verb directly, which is what to use on Windows without WSL or Git Bash:

```bash
docker compose build newton
docker compose run --rm newton help
docker compose run --rm newton test
docker compose run --rm newton paired --n 40
```

Two things the wrapper does that a plain `docker compose build` does not, so do
them yourself if you skip it:

- **It passes the commit.** Without `GIT_SHA`, the image carries no commit and
  every manifest a run writes records `git_sha: null`, which breaks the chain
  from a number back to the code that produced it. Build with
  `GIT_SHA=$(git rev-parse HEAD) docker compose build newton`.
- **On Linux, it maps your user.** Without `--user "$(id -u):$(id -g)"`, files
  written into `results/` are owned by root.

**There is no `docker compose up`.** This is a command-line program, not a
service: `up` runs the default command — the help — prefixes every line with the
service name, and leaves a stopped container behind. Use `run --rm`.

For the same reason, Docker Desktop's **Containers** tab stays empty. Each verb
creates a container, runs, and deletes itself. The image is under **Images**, and
the GUI's Run button only ever runs the help. Drop the `--rm` if you want a
container to inspect there afterwards.

## The verbs

Flags typed after a verb are appended to the command it runs, so they override
the defaults: `./newton paired --n 40` runs the paired comparison at forty
learners per arm. A verb that is not listed is passed to `agent-newton`, so the
whole CLI stays reachable: `./newton evaluate verifier --domain toy_algebra`.

### Read the design

| | |
|---|---|
| `arch` | `docs/architecture.md`, rendered |
| `arch <name>` | any other file under `docs/` |
| `docs` | list them |
| `config` | print and validate every run configuration |
| `results` | the stored summaries, and the run ids they name |

### Check it

| | |
|---|---|
| `test` | `pytest -m "not slow" -q` — what CI gates on |
| `typecheck` | `pyright` |
| `check` | domain content validation, then every run configuration |
| `version` | package, Python, and the commit the image was built from |

### Experiments

None of these calls a model.

Each writes its summary to `results/reruns/<name>/`, then compares it against the
stored one and prints either `reproduces … all identical` or the values that
differ. The committed summaries are never overwritten: each names the run
directories behind it, and rewriting it to name directories that were never
committed would leave a quotable number whose origin is not inspectable. To
overwrite deliberately, say so — `./newton paired --out results/paired_calculus`.

Per-run directories still land in `results/<run id>/` on the host, as they do
outside the container.

| | |
|---|---|
| `smoke` | end-to-end on `toy_algebra`, seconds |
| `verifier` | the symbolic verifier against its gold set, under a second |
| `cohort [arm]` | one cohort — `coupled` or `decoupled` |
| `paired` | the paired comparison, dose-matched |
| `propagation` | diagnostic error propagated to outcomes, across conditions |
| `ordering` | the ordering probe |
| `coverage` | misconception coverage against the item budget |
| `power` | power analysis — minutes, not seconds |
| `calibrate` | mastery estimate against held-out performance |
| `sweep <knob>` | `arbitration`, `prerequisites`, `headroom` or `doubt` |
| `figures` | redraw from the stored summaries, into `results/figures` — a tracked location, so the PNGs it writes are new files git will offer to commit |
| `all` | every one of the above, in order |

`paired`, `ordering`, `calibrate` and the sweeps run at `--n 160 --seed
20260811`, which is what the committed summaries were produced under — except
`sweep doubt`, whose summary records seed 20260819 and which is run at that.
`NEWTON_N` and `NEWTON_SEED` change both for a quicker look, at the cost of not
reproducing those numbers.

The three model-backed evaluations are redirected the same way, and compared
against the stored directory for the model and flags they default to.

### Needs a model

| | |
|---|---|
| `demo` | work through a session yourself, with the shared state visible |
| `diagnostic` | score the diagnostic agent against the injected labels |
| `tutor` | score the tutor on the turns a learner would read |

### Inspect a session

| | |
|---|---|
| `sitting [run]` | read a stored sitting back as prose; defaults to the latest |
| `history <learner>` | what was taught per concept, across sittings — reports no store until a demo has been sat, since `results/learners.db` is not committed |
| `shell` | a shell in the container — `/app`, with the environment on `PATH` |

## Where a model comes from

The three verbs above need Ollama with `gemma4:12b`. Everything else does not,
and a container with no model reaches every experiment in this document.

**On the host, by default.** The container looks for Ollama at
`http://host.docker.internal:11434`:

```bash
ollama serve
ollama pull gemma4:12b
./newton demo
```

The container never runs the model itself — it calls a server over HTTP.
Running that server natively on the host is the fast path, since it gets the
host's GPU.

**In a container, if there is no host install:**

```bash
./newton up-models     # starts the service and pulls the model
./newton demo
./newton down          # stops it; the weights stay in a named volume
```

That server is CPU-only on macOS, where Metal is not reachable from inside a
container, so a 12b model is slow there. It is for a host with no local install;
on Linux with a GPU it can be given one.

**Somewhere else:**

```bash
OLLAMA_HOST=http://<host>:11434 ./newton demo
```

Each model-backed verb checks the server is reachable and the model is pulled
before it starts, so an unreachable one is reported immediately rather than
after the provider's timeout.

## What is mounted

| Host | Container | |
|---|---|---|
| `./results` | `/app/results` | Runs write here. Manifests, metrics and figures land on the host. |
| `./.cache` | `/app/.cache` | The response cache. An interrupted model-backed evaluation restarts without repeating the calls. |

Source is baked into the image rather than mounted, so `docker run
agent-newton:local test` works on its own. To run against the working tree
instead — no rebuild between edits — use `./newton dev <verb>`.

One consequence worth knowing: `tests/test_publishable.py` reads `git ls-files`,
and the image has no working tree, so `./newton test` skips it and reports the
skip. `./newton dev test` mounts the working tree and runs it, as does CI.

## Provenance

Every run writes a manifest carrying the git SHA. A container built from a copy
of the source has no `.git` to read, so `./newton` passes the commit in as a
build argument and the image carries it in `AGENT_NEWTON_GIT_SHA`. `./newton
version` prints what a run made from this image will record.

Rebuild after committing, so the SHA in the image is the SHA the code is at:

```bash
./newton build
```

## Build arguments

| | |
|---|---|
| `PYTHON_VERSION` | Defaults to 3.12. CI gates on 3.10; the lock file covers both. |
| `GIT_SHA`, `GIT_DIRTY` | Filled by `./newton` from the working tree. |

## Environment

| | |
|---|---|
| `OLLAMA_HOST` | Where the model server is. Defaults to the host's. |
| `NEWTON_N`, `NEWTON_SEED` | Override what the experiment verbs run at. |
| `NEWTON_MODEL` | Which model the pre-flight check looks for. |
| `NEWTON_OUT` | Where re-runs are written. Defaults to `results/reruns`. |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Passed through for a configuration naming a hosted provider. An Ollama run needs neither. |
