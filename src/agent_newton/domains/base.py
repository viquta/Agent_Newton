"""The domain plug-in interface.

A domain supplies five things. Everything in ``core/`` is generic over them, and
``core/`` never imports this package's concrete domains — see
``tests/integration/test_domain_independence.py``.

===========================  ======================================================
Member                       Supplied as
===========================  ======================================================
``ConceptGraph``             ``concepts.yaml``   — prerequisite DAG
``MisconceptionCatalogue``   ``misconceptions.yaml`` — the shared label space
``ItemBank``                 ``items/*.yaml``    — practice / pre-test / post-test
``Verifier``                 Python              — model-independent correctness
``BuggyRule``                Python              — how the simulator errs
===========================  ======================================================

Three of the five are pure content, so extending a domain with new items or
concepts needs no Python at all. Only the two behavioural members are code.

Student responses cross this boundary as ``str``. Domains parse their own
notation internally. That keeps responses directly serialisable into the audit
log, which a generic response type would not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal, Protocol, Sequence, runtime_checkable

Bank = Literal["practice", "pretest", "posttest"]


class Verdict(str, Enum):
    """Outcome of checking one student response.

    ``UNPARSEABLE`` is deliberately distinct from ``INCORRECT``: a response the
    verifier cannot read is a measurement failure, not evidence about the
    learner, and BKT must not update on it.
    """

    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True, slots=True)
class Concept:
    id: str
    name: str
    prerequisites: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Misconception:
    """One entry in the shared label space.

    The simulator's buggy rules and the diagnostic agent's classification
    targets are drawn from the *same* catalogue. That is what makes diagnostic
    accuracy measurable against the injected ground truth.

    ``source`` is a literature citation and is required, so the catalogue stays
    traceable rather than accumulating invented errors.
    """

    id: str
    concept_id: str
    description: str
    source: str


@dataclass(frozen=True, slots=True)
class Item:
    """One exercise.

    ``probes`` names the misconceptions this item can reveal. It is data rather
    than something the buggy rules decide, so the validator can check both
    directions: that every probed rule actually produces a wrong answer here,
    and that every misconception is probed somewhere in the pre- and post-test.
    """

    id: str
    concept_id: str
    prompt: str
    answer: str
    bank: Bank = "practice"
    probes: tuple[str, ...] = ()
    #: Structured form of the item, for buggy rules to compute against. Rules
    #: read this rather than re-parsing ``prompt``, which would be brittle and
    #: would couple every rule to prompt wording.
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    verdict: Verdict
    correct_answer: str
    detail: str = ""

    @property
    def is_correct(self) -> bool:
        return self.verdict is Verdict.CORRECT

    @property
    def is_evidence(self) -> bool:
        """Whether this result may update the learner model.

        False for unparseable responses: they say nothing about mastery.
        """
        return self.verdict in (Verdict.CORRECT, Verdict.INCORRECT)


@runtime_checkable
class Verifier(Protocol):
    """Model-independent correctness.

    Called by the orchestrator after every student step — never as a tool an
    LLM elects to invoke. Correctness labels therefore do not depend on model
    quality, which is what allows a weak local model to serve the agents without
    compromising the correctness signal.

    Implementations must terminate. A CAS-backed verifier needs an explicit
    timeout; returning ``UNPARSEABLE`` beats hanging a cohort run overnight.
    """

    def verify(self, item: Item, response: str) -> VerificationResult: ...


@runtime_checkable
class BuggyRule(Protocol):
    """How the simulated learner errs under one misconception.

    Deterministic: the same item must always yield the same wrong response, so
    a seeded cohort is exactly reproducible.
    """

    @property
    def misconception_id(self) -> str: ...

    def apply(self, item: Item) -> str | None:
        """Return the wrong response, or None if this item cannot elicit it."""
        ...


class ConceptGraph(Protocol):
    """Prerequisite DAG. Also the substrate the ZPD frontier is computed over."""

    def concepts(self) -> Sequence[Concept]: ...
    def get(self, concept_id: str) -> Concept: ...
    def ids(self) -> Sequence[str]: ...
    def prerequisites(self, concept_id: str) -> frozenset[str]: ...
    def all_prerequisites(self, concept_id: str) -> frozenset[str]: ...
    def topological_order(self) -> Sequence[str]: ...
    def content_hash(self) -> str: ...

    def depth(self, concept_id: str) -> int:
        """Longest path from any root.

        Ranking uses this rather than position in the topological order, which
        is only a total order consistent with the graph: among concepts at the
        same depth its ordering comes from the declaration order in the YAML,
        and planning must not depend on how the content file happens to be
        written.
        """
        ...

    def goals(self) -> Sequence[str]:
        """Terminal concepts, in the order they should be worked toward.

        A goal is what makes planning directed: the concepts that matter for
        reaching it are its prerequisite closure, and everything else in the
        graph is out of scope until it is reached. Ordered, so a domain can say
        which target comes first rather than leaving it to the graph's shape.
        """
        ...


class MisconceptionCatalogue(Protocol):
    def all(self) -> Sequence[Misconception]: ...
    def get(self, misconception_id: str) -> Misconception: ...
    def ids(self) -> Sequence[str]: ...
    def for_concept(self, concept_id: str) -> Sequence[Misconception]: ...
    def content_hash(self) -> str: ...


class ItemBank(Protocol):
    def all(self) -> Sequence[Item]: ...
    def get(self, item_id: str) -> Item: ...
    def bank(self, bank: Bank) -> Sequence[Item]: ...
    def for_concept(self, concept_id: str, bank: Bank = "practice") -> Sequence[Item]: ...
    def content_hash(self) -> str: ...


@dataclass(frozen=True)
class Domain:
    """A subject area, assembled from the five members."""

    name: str
    concepts: ConceptGraph
    misconceptions: MisconceptionCatalogue
    items: ItemBank
    verifier: Verifier
    buggy_rules: dict[str, BuggyRule] = field(default_factory=dict)

    def buggy_rule(self, misconception_id: str) -> BuggyRule | None:
        return self.buggy_rules.get(misconception_id)

    def content_hashes(self) -> dict[str, str]:
        """Hashes for the run manifest.

        Recorded per run so analysis can refuse to pool results produced against
        different ground truth.
        """
        return {
            "concept_graph_hash": self.concepts.content_hash(),
            "catalogue_hash": self.misconceptions.content_hash(),
            "item_bank_hash": self.items.content_hash(),
        }


class DomainError(Exception):
    """Raised when domain content is malformed or inconsistent."""


def unknown_id_error(kind: str, bad: str, known: Iterable[str]) -> DomainError:
    options = ", ".join(sorted(known))
    return DomainError(f"unknown {kind} {bad!r}; known: {options}")
