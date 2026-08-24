# Agent_Newton in a container.
#
# Two stages: the first resolves the locked dependency set into a virtual
# environment, the second carries that environment and the project. Build tools
# and the uv binary stay in the first stage and never reach the shipped image.
#
#   docker build -t agent-newton:local .
#   docker run --rm agent-newton:local help
#
# `./newton` wraps both of those, and supplies GIT_SHA.

ARG PYTHON_VERSION=3.12

# --- Stage 1: the environment ---------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    # pyright drives a Node runtime that nodeenv fetches on first use. Naming
    # the directory is what lets the next stage carry it across; left at its
    # default it lands under a home directory this stage does not keep.
    PYRIGHT_PYTHON_ENV_DIR=/app/nodeenv

# The Node binary nodeenv fetches links against libatomic, which the slim
# image does not carry on arm64.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only the two files that decide the dependency set, so editing source does not
# invalidate this layer. `--all-extras --dev` matches .github/workflows/ci.yml:
# the container installs what CI gates on.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --dev --no-install-project

# Fetch that runtime now. Doing it here means `newton typecheck` needs no
# network at run time.
RUN /app/.venv/bin/pyright --version

# The project itself, into the same environment. Separate from the step above so
# that editing source does not re-resolve the dependency set. `--no-editable`
# builds and installs a wheel, so the console script resolves without the source
# tree having to be importable from the working directory.
COPY README.md LICENSE ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --dev --no-editable

# --- Stage 2: the image ---------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# git: manifest.py reads the SHA from it when the working tree is mounted.
# curl: the entrypoint's reachability check before a model-backed command.
# libatomic1: pyright's Node runtime links against it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates libatomic1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Writable by any uid, so `--user $(id -u)` does not send matplotlib and
    # the response cache looking for a home directory they cannot write.
    MPLCONFIGDIR=/tmp/mpl \
    HOME=/tmp \
    # Overridden by docker-compose.yml; this is the value for a bare
    # `docker run` against an Ollama the container can reach.
    OLLAMA_HOST=http://host.docker.internal:11434 \
    PYRIGHT_PYTHON_ENV_DIR=/app/nodeenv

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/nodeenv /app/nodeenv

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE CLAUDE.md ./
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY tests/ ./tests/
COPY docs/ ./docs/
# Committed run artifacts. `run_propagation.py` reads a stored diagnostic
# summary and `agent-newton sitting` reads stored transcripts, so the image is
# incomplete without them.
COPY results/ ./results/
COPY docker/entrypoint.sh /usr/local/bin/newton
COPY docker/render_doc.py /usr/local/lib/newton/render_doc.py

# `newton shell` starts bash with this. Without it there is no PS1 at all, and
# an interactive shell looks exactly like a hung one.
RUN printf '%s\n' \
    'PS1="\[\033[36m\]newton\[\033[0m\]:\w\$ "' \
    'case $- in *i*) ;; *) return ;; esac' \
    'echo' \
    'echo "Inside the container. /app holds the project; the environment is on PATH."' \
    'echo "results/ and .cache/ are the host'"'"'s directories — anything else is discarded on exit."' \
    'echo "exit, or ctrl-D, to leave."' \
    'echo' \
    > /etc/newton.bashrc

# Recorded in every manifest a run writes. Without them the git fields come back
# null in a container built from a COPY, and a summary stops being traceable to
# a commit. `./newton` fills both from the working tree.
ARG GIT_SHA=""
ARG GIT_DIRTY=""
ENV AGENT_NEWTON_GIT_SHA=${GIT_SHA} \
    AGENT_NEWTON_GIT_DIRTY=${GIT_DIRTY}

# Anything under results/ and .cache/ is written at run time. Both are usually
# bind-mounted; these keep a bare `docker run` working too.
RUN useradd --uid 1000 --create-home --shell /bin/bash newton \
    && mkdir -p /app/.cache/llm /app/results /tmp/mpl \
    && chmod -R a+rwX /app/.cache /app/results /tmp/mpl
USER newton

ENTRYPOINT ["/usr/local/bin/newton"]
CMD ["help"]
