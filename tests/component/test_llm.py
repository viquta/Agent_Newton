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
    ProviderError,
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
                    "planner": {"impl": "goal_directed"},
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
                    "planner": {"impl": "goal_directed"},
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


class TestGenerationIsBounded:
    """A call must not be able to run for minutes.

    Constrained decoding bounds a reply's *shape*, not its length. A schema
    holding an array or a free string permits an arbitrarily long reply, and a
    model that begins repeating itself will decode until it hits the context
    limit. Two independent bounds close that off: the schema caps what it can,
    and the provider caps the rest.
    """

    class RecordingClient:
        """Stands in for ``ollama.Client``, capturing what it was asked."""

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.calls: list[dict] = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return {"message": {"content": GOOD}}

    def _provider(self, monkeypatch, **kwargs):
        from agent_newton.llm.ollama import OllamaProvider

        provider = OllamaProvider("gemma4:12b", **kwargs)
        client = self.RecordingClient()
        monkeypatch.setattr(provider, "_connect", lambda: client)
        return provider, client

    def test_every_call_carries_a_token_cap(self, monkeypatch) -> None:
        from agent_newton.llm.ollama import MAX_TOKENS

        provider, client = self._provider(monkeypatch)
        provider.generate("classify this", Answer, system=None)
        assert client.calls[0]["options"]["num_predict"] == MAX_TOKENS

    def test_the_cap_is_far_above_a_real_reply(self, monkeypatch) -> None:
        # Set too low it would truncate legitimate replies, and every call would
        # come back malformed. The longest reply here is a two-sentence hint.
        from agent_newton.llm.ollama import MAX_TOKENS

        assert MAX_TOKENS >= 256

    def test_the_client_is_given_a_timeout(self, monkeypatch) -> None:
        # A request that never returns would hold a cohort run open for as long
        # as the machine stayed up.
        import ollama

        from agent_newton.llm.ollama import REQUEST_TIMEOUT, OllamaProvider

        built: dict = {}

        def _client(**kwargs):
            built.update(kwargs)
            return TestGenerationIsBounded.RecordingClient(**kwargs)

        monkeypatch.setattr(ollama, "Client", _client)
        OllamaProvider("gemma4:12b").generate("classify", Answer, system=None)
        assert built["timeout"] == REQUEST_TIMEOUT

    def test_a_host_is_still_honoured(self, monkeypatch) -> None:
        import ollama

        built: dict = {}

        from agent_newton.llm.ollama import OllamaProvider

        def _client(**kwargs):
            built.update(kwargs)
            return TestGenerationIsBounded.RecordingClient(**kwargs)

        monkeypatch.setattr(ollama, "Client", _client)
        OllamaProvider("gemma4:12b", host="http://elsewhere:11434").generate(
            "classify", Answer, system=None
        )
        assert built["host"] == "http://elsewhere:11434"
        assert "timeout" in built


class TestReasoningModels:
    """A model that deliberates answers in two channels.

    ``message.thinking`` fills first and ``message.content`` only once the model
    has finished. An empty ``content`` is therefore ambiguous: it may be a
    service failure, worth retrying, or a deliberation that outlasted the token
    budget, which will fail identically every time and belongs to the repair
    loop instead.
    """

    class Client:
        def __init__(self, message: dict, done_reason: str = "stop", **kwargs) -> None:
            self._message = message
            self._done_reason = done_reason
            self.kwargs = kwargs
            self.calls: list[dict] = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return {"message": self._message, "done_reason": self._done_reason}

    def _provider(self, monkeypatch, client, **kwargs):
        from agent_newton.llm.ollama import OllamaProvider

        provider = OllamaProvider("gemma4:12b", **kwargs)
        monkeypatch.setattr(provider, "_connect", lambda: client)
        return provider

    def test_a_deliberation_without_an_answer_is_a_malformed_reply(self, monkeypatch) -> None:
        client = self.Client({"content": "", "thinking": "step 1..."}, done_reason="length")
        provider = self._provider(monkeypatch, client)
        with pytest.raises(MalformedResponse):
            provider.generate("classify", Answer, system=None)

    def test_a_genuinely_empty_reply_is_still_a_provider_error(self, monkeypatch) -> None:
        # Nothing generated at all: a service problem, and worth retrying.
        client = self.Client({"content": "", "thinking": ""})
        provider = self._provider(monkeypatch, client)
        with pytest.raises(ProviderError) as raised:
            provider.generate("classify", Answer, system=None)
        assert not isinstance(raised.value, MalformedResponse)

    def test_a_malformed_reply_is_not_retried_by_the_provider(self, monkeypatch) -> None:
        # Temperature is zero: the same question gives the same non-answer, so
        # three attempts here spend the budget before the repair loop — which
        # does change the prompt — ever runs.
        client = self.Client({"content": "", "thinking": "..."}, done_reason="length")
        provider = self._provider(monkeypatch, client)
        with pytest.raises(MalformedResponse):
            provider.generate("classify", Answer, system=None)
        assert len(client.calls) == 1

    def test_the_repair_loop_handles_it(self, monkeypatch) -> None:
        # The failure must not escape as an exception that aborts a cohort run.
        class Deliberates:
            label = "ollama/gemma4:12b"

            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt, schema, system):
                self.calls += 1
                if self.calls == 1:
                    raise MalformedResponse("spent its budget thinking")
                return Completion(text=GOOD, model="m", provider="ollama")

        provider = Deliberates()
        answer = complete(provider, "classify", Answer)
        assert answer.label == "chain_rule_omits_inner"
        assert provider.calls == 2

    def test_the_mode_is_sent_only_when_configured(self, monkeypatch) -> None:
        # A model with no reasoning mode should get the server's own default.
        client = self.Client({"content": GOOD})
        self._provider(monkeypatch, client).generate("classify", Answer, system=None)
        assert "think" not in client.calls[0]

        client = self.Client({"content": GOOD})
        self._provider(monkeypatch, client, think=False).generate("classify", Answer, None)
        assert client.calls[0]["think"] is False

    def test_the_mode_changes_the_cache_key(self, tmp_path) -> None:
        # A reply produced by a deliberating model must not be served to a run
        # configured for a direct one; they are different calls.
        from agent_newton.llm.ollama import OllamaProvider

        labels = {
            OllamaProvider("gemma4:12b", think=think).label
            for think in (None, True, False)
        }
        assert len(labels) == 3

    def test_the_spec_carries_the_mode_into_the_provider(self, tmp_path) -> None:
        from agent_newton.llm.factory import build_provider

        provider = build_provider(
            ModelSpec(provider="ollama", model="gemma4:12b", think=False), None, cached=False
        )
        assert "think=false" in provider.label


class TestTheThreeCallLimits:
    """``num_predict``, ``num_ctx`` and ``timeout`` bound different things.

    Deliberation needs tokens to spend, room to hold the prompt and the thinking
    and the answer at once, and wall-clock to finish in. Raising one alone moves
    the failure rather than removing it, so all three are configurable and all
    three are checked here.

    ``num_ctx`` is the one that had never been set. Ollama drops the oldest
    context rather than refusing a prompt that does not fit, so exceeding it
    produces a confident reply to a question the model only partly saw — with
    nothing in the reply to say which part went missing.
    """

    class RecordingClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.calls: list[dict] = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return {"message": {"content": GOOD}}

    def _provider(self, monkeypatch, **kwargs):
        from agent_newton.llm.ollama import OllamaProvider

        provider = OllamaProvider("gemma4:12b", **kwargs)
        client = self.RecordingClient()
        monkeypatch.setattr(provider, "_connect", lambda: client)
        return provider, client

    def test_the_context_window_is_left_alone_unless_asked_for(self, monkeypatch) -> None:
        # What every run before this did. Sending the server's own default back
        # to it would be harmless but would make every manifest read as though a
        # window had been chosen.
        provider, client = self._provider(monkeypatch)
        provider.generate("classify", Answer, system=None)
        assert "num_ctx" not in client.calls[0]["options"]

    def test_the_context_window_is_sent_when_configured(self, monkeypatch) -> None:
        provider, client = self._provider(monkeypatch, context_tokens=8192)
        provider.generate("classify", Answer, system=None)
        assert client.calls[0]["options"]["num_ctx"] == 8192

    def test_the_token_budget_is_configurable(self, monkeypatch) -> None:
        provider, client = self._provider(monkeypatch, max_tokens=4096)
        provider.generate("classify", Answer, system=None)
        assert client.calls[0]["options"]["num_predict"] == 4096

    def test_the_timeout_reaches_the_client(self, monkeypatch) -> None:
        import ollama

        from agent_newton.llm.ollama import OllamaProvider

        built: dict = {}

        def _client(**kwargs):
            built.update(kwargs)
            return TestTheThreeCallLimits.RecordingClient(**kwargs)

        monkeypatch.setattr(ollama, "Client", _client)
        OllamaProvider("gemma4:12b", timeout=300).generate("classify", Answer, system=None)
        assert built["timeout"] == 300

    # -- the cache key ----------------------------------------------------
    #
    # The label is what the response cache keys on, so what appears in it
    # decides which stored replies stay valid.

    def test_an_unconfigured_provider_keeps_the_label_every_past_run_recorded(
        self,
    ) -> None:
        # The guard that protects the cache. Any of the new fields leaking into
        # the label when unset would invalidate every entry from every previous
        # run at once, and nothing would fail — the runs would simply become
        # slow and start costing model calls again.
        from agent_newton.llm.ollama import OllamaProvider

        assert OllamaProvider("gemma4:12b").label == "ollama/gemma4:12b"
        assert (
            OllamaProvider("gemma4:12b", think=False).label
            == "ollama/gemma4:12b+think=false"
        )

    def test_what_the_model_saw_is_part_of_the_key(self) -> None:
        # Both change what came back, so a reply produced under one must not be
        # served under the other.
        from agent_newton.llm.ollama import OllamaProvider

        assert "max_tokens=4096" in OllamaProvider("m", max_tokens=4096).label
        assert "ctx=8192" in OllamaProvider("m", context_tokens=8192).label
        assert (
            OllamaProvider("m", max_tokens=4096).label
            != OllamaProvider("m", max_tokens=1024).label
        )

    def test_the_timeout_is_not_part_of_the_key(self) -> None:
        # How long the caller was willing to wait does not change what came
        # back. Keying on it would throw away the whole cache over a knob with
        # no effect on content.
        from agent_newton.llm.ollama import OllamaProvider

        assert OllamaProvider("m", timeout=300).label == OllamaProvider("m").label

    def test_the_key_guard_can_fail(self) -> None:
        # A guard that cannot fail proves nothing. This is the shape the check
        # above rules out: a configured limit *must* move the label.
        from agent_newton.llm.ollama import OllamaProvider

        assert OllamaProvider("m", context_tokens=8192).label != OllamaProvider("m").label

    def test_the_config_carries_all_three_to_the_provider(self) -> None:
        from agent_newton.llm.ollama import OllamaProvider

        spec = ModelSpec(
            think=True, max_tokens=4096, context_tokens=8192, timeout_seconds=300
        )
        provider = build_provider(spec, cached=False)
        assert isinstance(provider, OllamaProvider)
        assert provider.label == "ollama/gemma4:12b+think=true+max_tokens=4096+ctx=8192"
        assert provider._timeout == 300

    def test_a_spec_that_sets_nothing_still_labels_as_it_always_did(self) -> None:
        assert ModelSpec().label() == "ollama/gemma4:12b"
        assert ModelSpec(think=False).label() == "ollama/gemma4:12b (think=false)"

    def test_the_manifest_states_the_limits_when_they_are_set(self) -> None:
        # A run has to say what it was produced under, and the model name alone
        # no longer says it.
        label = ModelSpec(think=True, max_tokens=4096, context_tokens=8192).label()
        assert "think=true" in label
        assert "max_tokens=4096" in label
        assert "num_ctx=8192" in label


class TestATimeoutIsNotRetriedThreeTimes:
    """A timeout is deterministic, so asking again just waits again.

    The same reasoning already applied to a malformed reply: temperature is
    zero, so the identical question produces the identical outcome. Under the
    old predicate a timeout was an ordinary ``ProviderError`` and was retried
    three times with backoff — turning one wait into three, which is felt
    hardest exactly where the timeout is longest and where it was raised on
    purpose to let a model deliberate.
    """

    class Timeouts:
        """A client whose call runs out of time.

        The exception's name is what the provider matches on, deliberately: the
        ollama package has changed transport before, and importing its HTTP
        library to name one class would break quietly the next time it does.
        """

        class ReadTimeout(Exception):
            pass

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            raise self.ReadTimeout("timed out")

    class Refuses:
        """A client that fails for some other reason. Worth asking again."""

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            raise ConnectionError("connection reset")

    def _provider(self, monkeypatch, client):
        from agent_newton.llm.ollama import OllamaProvider

        provider = OllamaProvider("gemma4:12b")
        monkeypatch.setattr(provider, "_connect", lambda: client)
        return provider

    def test_a_timeout_is_asked_once(self, monkeypatch) -> None:
        from agent_newton.llm.base import ProviderTimeout

        client = self.Timeouts()
        provider = self._provider(monkeypatch, client)
        with pytest.raises(ProviderTimeout):
            provider.generate("classify", Answer, system=None)
        assert client.calls == 1

    def test_a_dropped_connection_is_still_retried(self, monkeypatch) -> None:
        # The other half of the guard. Without this, excluding timeouts could be
        # achieved by excluding everything, and the test above would still pass.
        client = self.Refuses()
        provider = self._provider(monkeypatch, client)
        with pytest.raises(ProviderError):
            provider.generate("classify", Answer, system=None)
        assert client.calls == 3

    def test_a_timeout_is_still_a_provider_error(self, monkeypatch) -> None:
        # Every existing handler catches `ProviderError`: the diagnostic counts
        # it as a failure to infer, the tutor falls back to a fixed hint, and
        # the demo stores the sitting rather than losing it. Narrowing the type
        # must not narrow what survives a dead backend.
        from agent_newton.llm.base import ProviderTimeout

        assert issubclass(ProviderTimeout, ProviderError)
        assert not issubclass(ProviderTimeout, MalformedResponse)

    def test_the_message_says_what_to_change(self, monkeypatch) -> None:
        from agent_newton.llm.base import ProviderTimeout

        provider = self._provider(monkeypatch, self.Timeouts())
        with pytest.raises(ProviderTimeout) as raised:
            provider.generate("classify", Answer, system=None)
        assert "timeout_seconds" in str(raised.value)


class TestAPromptApproachingTheWindowSaysSo:
    """Truncation is the silent failure this project keeps finding.

    Ollama drops the oldest context rather than refusing, so an overlong prompt
    yields a well-formed, schema-valid, confident reply to a question the model
    was shown only part of. Nothing downstream can tell. The warning is the only
    thing that can.
    """

    class RecordingClient:
        def __init__(self, **kwargs) -> None:
            self.calls: list[dict] = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return {"message": {"content": GOOD}}

    def _provider(self, monkeypatch, **kwargs):
        from agent_newton.llm.ollama import OllamaProvider

        provider = OllamaProvider("gemma4:12b", **kwargs)
        monkeypatch.setattr(provider, "_connect", lambda: self.RecordingClient())
        return provider

    def test_a_long_prompt_against_a_small_window_warns(
        self, monkeypatch, caplog
    ) -> None:
        provider = self._provider(monkeypatch, context_tokens=512)
        with caplog.at_level("WARNING"):
            provider.generate("x" * 4000, Answer, system=None)
        assert any("context" in record.message for record in caplog.records)

    def test_a_short_prompt_is_silent(self, monkeypatch, caplog) -> None:
        provider = self._provider(monkeypatch, context_tokens=8192)
        with caplog.at_level("WARNING"):
            provider.generate("classify this", Answer, system=None)
        assert not caplog.records

    def test_nothing_is_claimed_when_no_window_was_configured(
        self, monkeypatch, caplog
    ) -> None:
        # There is no number to compare against — the server applies its own,
        # and guessing at it would produce warnings that mean nothing.
        provider = self._provider(monkeypatch)
        with caplog.at_level("WARNING"):
            provider.generate("x" * 40000, Answer, system=None)
        assert not caplog.records

    def test_the_system_prompt_counts_toward_the_window(
        self, monkeypatch, caplog
    ) -> None:
        # It is sent as a message like any other, so leaving it out would
        # under-count every call the agents actually make — all of them carry
        # one.
        provider = self._provider(monkeypatch, context_tokens=512)
        with caplog.at_level("WARNING"):
            provider.generate("short", Answer, system="y" * 4000)
        assert any("context" in record.message for record in caplog.records)
