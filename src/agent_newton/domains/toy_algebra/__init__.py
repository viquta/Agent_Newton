"""toy_algebra — the reference domain.

Small on purpose. It exists to keep the domain abstraction honest (an interface
validated against one implementation is not validated) and to act as the fast
CI fixture, where the whole pipeline runs end to end in seconds without a model.

It is not a subject domain in its own right. Do not grow it.
"""

from __future__ import annotations

from pathlib import Path

from agent_newton.domains.base import Domain
from agent_newton.domains.content import (
    YamlConceptGraph,
    YamlItemBank,
    YamlMisconceptionCatalogue,
)
from agent_newton.domains.toy_algebra.rules import RULES
from agent_newton.domains.toy_algebra.verifier import NormalizingVerifier

_HERE = Path(__file__).parent


def build() -> Domain:
    return Domain(
        name="toy_algebra",
        concepts=YamlConceptGraph.from_yaml(_HERE / "concepts.yaml"),
        misconceptions=YamlMisconceptionCatalogue.from_yaml(_HERE / "misconceptions.yaml"),
        items=YamlItemBank.from_dir(_HERE / "items"),
        verifier=NormalizingVerifier(),
        buggy_rules={rule.misconception_id: rule for rule in RULES},
    )
