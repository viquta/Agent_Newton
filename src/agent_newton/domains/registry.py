"""Domain lookup by name.

Builders are imported lazily so a ``toy_algebra`` run — the CI path and the
threshold sweep — never pays for importing sympy.
"""

from __future__ import annotations

from importlib import import_module

from agent_newton.domains.base import Domain, DomainError

#: name -> "module:factory". Adding a domain means adding one line here and
#: implementing a verifier plus buggy rules; the three content members are YAML.
_BUILDERS: dict[str, str] = {
    "toy_algebra": "agent_newton.domains.toy_algebra:build",
    "calculus": "agent_newton.domains.calculus:build",
}


def available() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


def load_domain(name: str) -> Domain:
    try:
        target = _BUILDERS[name]
    except KeyError:
        raise DomainError(
            f"unknown domain {name!r}; available: {', '.join(available())}"
        ) from None

    module_path, _, factory_name = target.partition(":")
    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise DomainError(f"domain {name!r} failed to import: {exc}") from exc

    domain: Domain = getattr(module, factory_name)()
    if domain.name != name:
        raise DomainError(
            f"domain registered as {name!r} reports its name as {domain.name!r}; "
            f"the registry key and Domain.name must agree, or manifests will "
            f"misdescribe the run"
        )
    return domain
