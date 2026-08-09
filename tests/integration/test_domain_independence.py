"""``core/`` may depend on the domain *interface*, never on a concrete domain.

This is the invariant that makes the architecture retargetable: a new subject
area is added by writing a domain, not by editing the system. Modularity that is
not tested erodes, so it is checked mechanically rather than left as a
convention in a README.

``agent_newton.domains.base`` holds the five Protocols and the shared dataclasses.
Depending on that is the whole point. Depending on ``domains.calculus``,
``domains.registry`` or ``domains.content`` is not: the first hardcodes a
subject, and the others let ``core/`` reach concrete content on its own instead
of being handed a ``Domain``.

Dynamic lookups (``import_module("agent_newton.domains.calculus")``) are caught
too — that is the loophole that would otherwise reopen the coupling quietly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
CORE = SRC / "agent_newton" / "core"

DOMAIN_PACKAGE = "agent_newton.domains"
#: The interface module, and only it.
ALLOWED = frozenset({f"{DOMAIN_PACKAGE}.base"})


def _core_modules() -> list[Path]:
    return sorted(CORE.rglob("*.py"))


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(node: ast.ImportFrom, module_name: str) -> str | None:
    """Absolute module for an ``ImportFrom``, resolving relative levels."""
    if node.level == 0:
        return node.module
    parts = module_name.split(".")
    base = parts[: len(parts) - node.level + 1]
    prefix = ".".join(base)
    return f"{prefix}.{node.module}" if node.module else prefix


def _is_violation(target: str) -> bool:
    if target in ALLOWED:
        return False
    return target == DOMAIN_PACKAGE or target.startswith(f"{DOMAIN_PACKAGE}.")


def test_core_has_modules_to_check() -> None:
    # Guards against this suite silently passing because the glob found nothing.
    assert _core_modules(), f"no modules found under {CORE}"


@pytest.mark.parametrize("path", _core_modules(), ids=lambda p: p.stem)
def test_core_module_does_not_import_a_concrete_domain(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    module_name = _module_name(path)
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if _is_violation(a.name)]

        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve(node, module_name)
            if not resolved:
                continue
            if resolved == DOMAIN_PACKAGE:
                # `from agent_newton.domains import base` is fine; anything else
                # names a concrete module.
                offenders += [
                    f"{resolved}.{a.name}"
                    for a in node.names
                    if _is_violation(f"{resolved}.{a.name}")
                ]
            elif _is_violation(resolved):
                offenders.append(resolved)

        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _is_violation(node.value):
                offenders.append(f"{node.value!r} (dynamic)")

    assert not offenders, (
        f"{path.relative_to(SRC)} depends on a concrete domain: {offenders}. "
        f"core/ is generic over the Protocols in {DOMAIN_PACKAGE}.base — take a "
        f"Domain as a parameter rather than importing one."
    )


def test_the_check_would_catch_a_violation() -> None:
    """The guard must actually fire — a check that cannot fail proves nothing."""
    assert _is_violation("agent_newton.domains.calculus")
    assert _is_violation("agent_newton.domains.registry")
    assert _is_violation("agent_newton.domains.toy_algebra")
    assert not _is_violation("agent_newton.domains.base")
    assert not _is_violation("agent_newton.config")
