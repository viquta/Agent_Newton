"""Run manifests.

Every run writes one. The manifest is what makes a reported number traceable
back to the exact code, content and models that produced it — and it is what
stops the single most dangerous silent failure in this project:

Extending the misconception catalogue changes the diagnostic agent's label
space, which invalidates any previously reported diagnostic accuracy. The
catalogue hash recorded here lets analysis *refuse* to pool runs that are not
comparable, rather than averaging them into a plausible-looking wrong number.
See :func:`assert_poolable`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

from agent_newton.config import Config


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def hash_content(*parts: str | bytes) -> str:
    """Stable short hash over content parts, for catalogue/item-bank hashes."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode() if isinstance(part, str) else part)
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


class RunManifest(BaseModel):
    """Provenance for one run."""

    run_id: str
    run_name: str
    created_at: str

    git_sha: str | None = None
    #: True when the working tree had uncommitted changes. A dirty run is not
    #: reproducible from its SHA, so results carrying this flag are provisional.
    git_dirty: bool = False

    config_hash: str
    arm: str
    domain: str
    seed: int

    #: Domain content hashes. Populated once the domain is loaded.
    catalogue_hash: str | None = None
    item_bank_hash: str | None = None
    concept_graph_hash: str | None = None

    #: role -> "provider/model", or "<impl>" for model-free implementations.
    models: dict[str, str] = Field(default_factory=dict)
    uses_llm: bool = False

    python_version: str = Field(default_factory=platform.python_version)
    platform: str = Field(default_factory=platform.platform)

    @classmethod
    def create(cls, config: Config, run_id: str) -> RunManifest:
        status = _git("status", "--porcelain")
        agents = config.agents

        def role(spec, impl: str) -> str:
            return spec.label() if impl == "llm" else f"<{impl}>"

        models = {
            "tutor": role(agents.tutor, agents.tutor.impl),
            "diagnostic": role(agents.diagnostic, agents.diagnostic.impl),
            "planner": role(agents.planner, agents.planner.impl),
            "simulator_surface": (
                config.simulator.surface_model.label()
                if config.simulator.surface == "llm"
                else "<symbolic>"
            ),
        }
        return cls(
            run_id=run_id,
            run_name=config.run_name,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_sha=_git("rev-parse", "HEAD"),
            git_dirty=bool(status),
            config_hash=config.content_hash(),
            arm=config.arm,
            domain=config.domain,
            seed=config.seed,
            models=models,
            uses_llm=config.uses_llm(),
        )

    def write(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "manifest.json"
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        return path

    @classmethod
    def read(cls, run_dir: Path) -> RunManifest:
        return cls.model_validate_json((run_dir / "manifest.json").read_text())


class IncomparableRunsError(RuntimeError):
    """Raised when runs that cannot be validly pooled are about to be pooled."""


def assert_poolable(manifests: Sequence[RunManifest]) -> None:
    """Refuse to aggregate runs whose ground truth differs.

    Runs may be compared only if they share a domain and identical domain
    content. A differing catalogue hash means the diagnostic agent was
    classifying into a different label space, so accuracy figures are not on the
    same scale; a differing item-bank hash means learners saw different work.

    Called by the analysis entry points before any aggregation.
    """
    if len(manifests) < 2:
        return

    def _distinct(field: str) -> list:
        return sorted({getattr(m, field) for m in manifests}, key=str)

    for field, explanation in (
        ("domain", "runs are from different domains"),
        (
            "catalogue_hash",
            "the misconception catalogue changed, so the diagnostic agent's label "
            "space differs and accuracy figures are not on the same scale",
        ),
        ("item_bank_hash", "learners were given different items"),
        ("concept_graph_hash", "the prerequisite graph differs, so the ZPD frontier differs"),
    ):
        values = _distinct(field)
        if len(values) > 1:
            ids = ", ".join(sorted(m.run_id for m in manifests))
            raise IncomparableRunsError(
                f"Refusing to pool runs [{ids}]: {explanation} "
                f"({field} = {values}). Re-run the older arm against the current "
                f"domain content, or analyse the groups separately."
            )


def load_manifests(results_dir: Path) -> Iterable[RunManifest]:
    for path in sorted(results_dir.glob("*/manifest.json")):
        yield RunManifest.model_validate_json(path.read_text())
