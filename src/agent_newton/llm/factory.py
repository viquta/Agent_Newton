"""Building providers from config.

One place turns a :class:`~agent_newton.config.ModelSpec` into a working
provider, wrapped in the cache. Backends are imported lazily, so a run using
only Ollama never touches the hosted SDKs.
"""

from __future__ import annotations

from pathlib import Path

from agent_newton.config import Config, ModelSpec
from agent_newton.llm.base import LLMProvider, ProviderError
from agent_newton.llm.cache import CachedProvider


def build_provider(
    spec: ModelSpec, cache_dir: Path | None = None, *, cached: bool = True
) -> LLMProvider:
    """A provider for one role, cached unless asked otherwise."""
    if spec.provider == "ollama":
        from agent_newton.llm.ollama import OllamaProvider

        provider: LLMProvider = OllamaProvider(spec.model, think=spec.think)
    elif spec.provider == "anthropic":
        from agent_newton.llm.remote import AnthropicProvider

        provider = AnthropicProvider(spec.model)
    elif spec.provider == "openai":
        from agent_newton.llm.remote import OpenAIProvider

        provider = OpenAIProvider(spec.model)
    else:  # pragma: no cover — the config Literal already excludes this
        raise ProviderError(f"unknown provider {spec.provider!r}")

    if cached and cache_dir is not None:
        return CachedProvider(provider, cache_dir)
    return provider


def providers_for(config: Config) -> dict[str, LLMProvider]:
    """Build a provider per model-backed role.

    Roles running a model-free implementation get nothing: an oracle diagnostic
    has no provider to build, and constructing one would misreport the run.
    """
    built: dict[str, LLMProvider] = {}
    cache_dir = config.paths.cache_dir

    if config.agents.tutor.impl == "llm":
        built["tutor"] = build_provider(config.agents.tutor, cache_dir)
    if config.agents.diagnostic.impl == "llm":
        built["diagnostic"] = build_provider(config.agents.diagnostic, cache_dir)
    if config.agents.planner.impl == "llm":
        built["planner"] = build_provider(config.agents.planner, cache_dir)
    if config.simulator.surface == "llm":
        built["simulator_surface"] = build_provider(config.simulator.surface_model, cache_dir)

    return built
