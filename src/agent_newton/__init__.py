"""Agent_Newton — a multi-agent ITS with a shared, ZPD-bearing learner-state layer.

Layout
------
``core/``     Domain-agnostic. Must never import ``domains`` — enforced by
              ``tests/integration/test_domain_independence.py``.
``domains/``  Pluggable subject matter behind five Protocols. ``toy_algebra`` is
              the reference implementation; ``calculus`` is the primary domain.
``llm/``      Provider adapters (Ollama / Anthropic / OpenAI) behind one
              Protocol, with an on-disk response cache.

The learner state is a blackboard: agents never call one another, only read and
write shared state.
"""

__version__ = "0.1.0"
