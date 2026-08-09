"""The provider layer.

No test here contacts a real model. Backends are exercised through a fake that
implements the same protocol, so the retry, repair and caching logic is tested
without a running service.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from agent_newton.config import Config, ModelSpec
from agent_newton.llm.base import (
    Completion,
    LLMProvider,
    MalformedResponse,
    complete,
)
from agent_newton.llm.cache import CachedProvider, cache_key
from agent_newton.llm.factory import build_provider, providers_for


class Answer(BaseModel):
    label: str
    confidence: float = 0.0


class FakeProvider:
    """Replies from a script, and counts how often it was asked."""

    def __init__(self, *replies: str, label: str = "fake/model-1") -> None:
        self._replies = list(replies)
        self._label = label
        self.calls: list[str] = []

    @property
    def label(self) -> str:
        return self._label

    def generate(self, prompt: str, schema: type[BaseModel], system: str | None) -> Completion:
        self.calls.append(prompt)
        reply = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        return Completion(text=reply, model="model-1", provider="fake")


GOOD = json.dumps({"label": "chain_rule_omits_inner", "confidence": 0.8})


class TestStructuredCompletion:
    def test_returns_a_validated_model(self) -> None:
        answer = complete(FakeProvider(GOOD), "classify", Answer)
        assert answer.label == "chain_rule_omits_inner"

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(FakeProvider(GOOD), LLMProvider)

    @pytest.mark.parametrize(
        "wrapped",
        [
            f"```json\n{GOOD}\n```",
            f"Here is my answer:\n{GOOD}",
            f"```\n{GOOD}\n```",
            f"Thinking... {GOOD} ...done",
        ],
    )
    def test_extracts_json_from_surrounding_prose(self, wrapped: str) -> None:
        # A small model will not reliably return bare JSON, and a whole retry
        # for a code fence would be waste.
        assert complete(FakeProvider(wrapped), "classify", Answer).label

    def test_repairs_a_malformed_reply(self) -> None:
        provider = FakeProvider("not json at all", GOOD)
        assert complete(provider, "classify", Answer).label
        assert len(provider.calls) == 2

    def test_the_repair_shows_what_was_wrong(self) -> None:
        # Handing back the validation error makes the second attempt a
        # correction rather than another roll of the same dice.
        provider = FakeProvider("not json at all", GOOD)
        complete(provider, "classify", Answer)
        assert "could not be used" in provider.calls[1]
        assert "not json at all" in provider.calls[1]

    def test_gives_up_loudly(self) -> None:
        # Never a default: a silently-defaulted diagnosis would be recorded as a
        # real inference and corrupt the accuracy measurement.
        provider = FakeProvider("still not json")
        with pytest.raises(MalformedResponse):
            complete(provider, "classify", Answer, max_attempts=2)

    def test_respects_the_attempt_budget(self) -> None:
        provider = FakeProvider("nope")
        with pytest.raises(MalformedResponse):
            complete(provider, "classify", Answer, max_attempts=3)
        assert len(provider.calls) == 3


class TestCache:
    def test_a_repeated_call_does_not_reach_the_provider(self, tmp_path) -> None:
        inner = FakeProvider(GOOD)
        cached = CachedProvider(inner, tmp_path)
        complete(cached, "classify", Answer)
        complete(cached, "classify", Answer)
        assert len(inner.calls) == 1
        assert cached.hits == 1 and cached.misses == 1

    def test_survives_a_new_process(self, tmp_path) -> None:
        # Re-deriving a figure from stored results must not re-run a cohort.
        complete(CachedProvider(FakeProvider(GOOD), tmp_path), "classify", Answer)
        second = FakeProvider("would be wrong")
        answer = complete(CachedProvider(second, tmp_path), "classify", Answer)
        assert answer.label == "chain_rule_omits_inner"
        assert not second.calls

    @pytest.mark.parametrize(
        "changed",
        [
            {"prompt": "different"},
            {"model": "other"},
            {"provider": "openai"},
            {"system": "a system prompt"},
        ],
    )
    def test_every_component_changes_the_key(self, changed: dict) -> None:
        base = {"provider": "ollama", "model": "m", "prompt": "p", "system": None}
        assert cache_key(**base, schema=Answer) != cache_key(
            **{**base, **changed}, schema=Answer
        )

    def test_the_schema_is_part_of_the_key(self) -> None:
        # The same prompt asking for a different shape is a different call.
        class Other(BaseModel):
            reason: str

        base = {"provider": "ollama", "model": "m", "prompt": "p", "system": None}
        assert cache_key(**base, schema=Answer) != cache_key(**base, schema=Other)

    def test_reports_its_hit_rate(self, tmp_path) -> None:
        cached = CachedProvider(FakeProvider(GOOD), tmp_path)
        for _ in range(4):
            complete(cached, "classify", Answer)
        assert cached.hit_rate == pytest.approx(0.75)


class TestFactory:
    def test_builds_an_ollama_provider_without_contacting_it(self, tmp_path) -> None:
        # Construction must not connect, or every import would need a server.
        provider = build_provider(ModelSpec(provider="ollama", model="gemma4:12b"), tmp_path)
        assert provider.label == "ollama/gemma4:12b"

    def test_builds_nothing_for_model_free_roles(self) -> None:
        # An oracle diagnostic has no provider; building one would misreport
        # the run in the manifest.
        config = Config.model_validate(
            {
                "agents": {
                    "tutor": {"impl": "template"},
                    "diagnostic": {"impl": "oracle"},
                    "planner": {"impl": "deterministic"},
                },
                "simulator": {"surface": "symbolic"},
            }
        )
        assert providers_for(config) == {}

    def test_builds_only_the_model_backed_roles(self) -> None:
        config = Config.model_validate(
            {
                "agents": {
                    "tutor": {"impl": "llm", "provider": "ollama", "model": "gemma4:12b"},
                    "diagnostic": {"impl": "oracle"},
                    "planner": {"impl": "deterministic"},
                },
                "simulator": {"surface": "symbolic"},
            }
        )
        assert set(providers_for(config)) == {"tutor"}

    def test_remote_backends_are_not_imported_for_a_local_run(self) -> None:
        import subprocess
        import sys

        code = (
            "import sys;"
            "from agent_newton.llm.factory import build_provider;"
            "from agent_newton.config import ModelSpec;"
            "build_provider(ModelSpec(provider='ollama', model='m'), None, cached=False);"
            "print('anthropic' in sys.modules or 'openai' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"
