"""Single-variable calculus — the primary domain.

Concepts run from limits through the differentiation rules to integration by
substitution. The misconception catalogue is drawn from studies that document
these errors in students; entries whose source is not yet confirmed are marked
and reported by ``domain validate``.

Answers are checked by symbolic equivalence, so a learner may write a correct
answer in any equivalent form.
"""

from __future__ import annotations

from pathlib import Path

from agent_newton.domains.base import Domain
from agent_newton.domains.calculus.rules import RULES
from agent_newton.domains.calculus.templates import TEMPLATES
from agent_newton.domains.calculus.verifier import SymbolicVerifier
from agent_newton.domains.content import (
    YamlConceptGraph,
    YamlConceptResources,
    YamlItemBank,
    YamlMisconceptionCatalogue,
)

_HERE = Path(__file__).parent


def build() -> Domain:
    return Domain(
        name="calculus",
        concepts=YamlConceptGraph.from_yaml(_HERE / "concepts.yaml"),
        misconceptions=YamlMisconceptionCatalogue.from_yaml(_HERE / "misconceptions.yaml"),
        items=YamlItemBank.from_dir(_HERE / "items"),
        verifier=SymbolicVerifier(),
        buggy_rules={rule.misconception_id: rule for rule in RULES},
        templates={template.item_id: template for template in TEMPLATES},
        resources=YamlConceptResources.from_yaml(_HERE / "resources.yaml"),
    )
