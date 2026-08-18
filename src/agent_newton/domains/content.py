"""YAML-backed implementations of the three content members.

Every domain shares these, so adding items or concepts to any domain is a data
edit. Only :class:`~agent_newton.domains.base.Verifier` and
:class:`~agent_newton.domains.base.BuggyRule` need domain-specific Python.

Content hashes are computed over the canonical parsed content rather than the
raw file bytes, so reformatting YAML or reordering entries does not spuriously
invalidate comparability between runs — only a genuine change to what a learner
faces does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import networkx as nx
import yaml

from agent_newton.domains.base import (
    Bank,
    Concept,
    ConceptGraph,
    ConceptResource,
    DomainError,
    Item,
    Misconception,
    unknown_id_error,
)
from agent_newton.manifest import hash_content


def _load_document(path: Path) -> dict:
    if not path.exists():
        raise DomainError(f"missing content file: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _load_yaml(path: Path, key: str) -> list[dict]:
    entries = _load_document(path).get(key)
    if not entries:
        raise DomainError(f"{path} has no '{key}' entries")
    return entries


class YamlConceptGraph:
    """Prerequisite DAG loaded from ``concepts.yaml``.

    Satisfies :class:`~agent_newton.domains.base.ConceptGraph` structurally; the
    Protocols in ``base`` are for typing, not inheritance.
    """

    def __init__(self, concepts: Sequence[Concept], goals: Sequence[str] = ()) -> None:
        self._by_id: dict[str, Concept] = {}
        for concept in concepts:
            if concept.id in self._by_id:
                raise DomainError(f"duplicate concept id: {concept.id!r}")
            self._by_id[concept.id] = concept

        graph = nx.DiGraph()
        for concept in concepts:
            graph.add_node(concept.id)
        for concept in concepts:
            for prereq in concept.prerequisites:
                if prereq not in self._by_id:
                    raise unknown_id_error(
                        f"prerequisite of {concept.id!r}:", prereq, self._by_id
                    )
                graph.add_edge(prereq, concept.id)

        if not nx.is_directed_acyclic_graph(graph):
            cycle = nx.find_cycle(graph)
            raise DomainError(
                f"prerequisite graph has a cycle: {' -> '.join(u for u, _ in cycle)}. "
                f"Topological ordering and the ZPD frontier both require a DAG."
            )

        self._graph = graph
        self._order = tuple(nx.topological_sort(graph))
        self._depths: dict[str, int] | None = None

        for goal in goals:
            if goal not in self._by_id:
                raise unknown_id_error("goal concept", goal, self._by_id)
        # Absent, the sinks are the goals. A concept nothing depends on is where
        # a path through the graph ends, so this is a definition rather than a
        # stopgap — a domain that declares nothing still plans toward something.
        self._goals: tuple[str, ...] = tuple(goals) or tuple(
            cid for cid in self._order if graph.out_degree(cid) == 0
        )

    @classmethod
    def from_yaml(cls, path: Path) -> YamlConceptGraph:
        document = _load_document(path)
        entries = document.get("concepts")
        if not entries:
            raise DomainError(f"{path} has no 'concepts' entries")
        return cls(
            [
                Concept(
                    id=entry["id"],
                    name=entry.get("name", entry["id"]),
                    prerequisites=tuple(entry.get("prerequisites", ()) or ()),
                )
                for entry in entries
            ],
            goals=tuple(document.get("goals", ()) or ()),
        )

    def goals(self) -> Sequence[str]:
        return self._goals

    def concepts(self) -> Sequence[Concept]:
        return tuple(self._by_id[cid] for cid in self._order)

    def get(self, concept_id: str) -> Concept:
        try:
            return self._by_id[concept_id]
        except KeyError:
            raise unknown_id_error("concept", concept_id, self._by_id) from None

    def ids(self) -> Sequence[str]:
        return self._order

    def prerequisites(self, concept_id: str) -> frozenset[str]:
        """Direct prerequisites only."""
        return frozenset(self.get(concept_id).prerequisites)

    def all_prerequisites(self, concept_id: str) -> frozenset[str]:
        """Transitive closure — every concept that must precede this one."""
        self.get(concept_id)
        return frozenset(nx.ancestors(self._graph, concept_id))

    def topological_order(self) -> Sequence[str]:
        return self._order

    def depth(self, concept_id: str) -> int:
        """Longest path from any root. The frontier fallback orders by this."""
        self.get(concept_id)
        if self._depths is None:
            # Computed in topological order, so each node's prerequisites are
            # already resolved. Memoised because the frontier fallback calls this
            # on every concept, every replan.
            depths: dict[str, int] = {}
            for cid in self._order:
                prereqs = self._by_id[cid].prerequisites
                depths[cid] = 1 + max((depths[p] for p in prereqs), default=-1)
            self._depths = depths
        return self._depths[concept_id]

    def content_hash(self) -> str:
        # Goals are part of the hash: changing what a learner is worked toward
        # changes what the run measured, so results from before and after must
        # not be pooled.
        return hash_content(
            *(f"{c.id}|{c.name}|{','.join(sorted(c.prerequisites))}" for c in self.concepts()),
            f"goals={'>'.join(self._goals)}",
        )


class YamlMisconceptionCatalogue:
    """The shared label space, loaded from ``misconceptions.yaml``."""

    def __init__(self, misconceptions: Sequence[Misconception]) -> None:
        self._by_id: dict[str, Misconception] = {}
        for entry in misconceptions:
            if entry.id in self._by_id:
                raise DomainError(f"duplicate misconception id: {entry.id!r}")
            self._by_id[entry.id] = entry
        self._order = tuple(self._by_id)

    @classmethod
    def from_yaml(cls, path: Path) -> YamlMisconceptionCatalogue:
        entries = []
        for entry in _load_yaml(path, "misconceptions"):
            missing = {"id", "concept_id", "description", "source"} - set(entry)
            if missing:
                raise DomainError(
                    f"misconception {entry.get('id', '<unnamed>')!r} is missing "
                    f"{sorted(missing)}. 'source' is required so the catalogue stays "
                    f"traceable: every entry must cite where the error is documented."
                )
            entries.append(
                Misconception(
                    id=entry["id"],
                    concept_id=entry["concept_id"],
                    description=entry["description"],
                    source=entry["source"],
                )
            )
        return cls(entries)

    def all(self) -> Sequence[Misconception]:
        return tuple(self._by_id.values())

    def get(self, misconception_id: str) -> Misconception:
        try:
            return self._by_id[misconception_id]
        except KeyError:
            raise unknown_id_error("misconception", misconception_id, self._by_id) from None

    def ids(self) -> Sequence[str]:
        return self._order

    def for_concept(self, concept_id: str) -> Sequence[Misconception]:
        return tuple(m for m in self._by_id.values() if m.concept_id == concept_id)

    def content_hash(self) -> str:
        # Sorted, so reordering the YAML does not change the hash — but adding,
        # removing or re-describing an entry does, because that changes the
        # diagnostic agent's label space.
        return hash_content(
            *(
                f"{m.id}|{m.concept_id}|{m.description}"
                for m in sorted(self._by_id.values(), key=lambda m: m.id)
            )
        )


class YamlConceptResources:
    """What may be shown beside a question, loaded from ``resources.yaml``.

    One entry per concept at most. A concept with no entry simply has nothing
    shown beside its questions; ``domain validate`` warns rather than refusing,
    for the same reason it warns about a concept with no catalogue entry — the
    gap may be deliberate, but it must not go unnoticed.
    """

    def __init__(self, resources: Sequence[ConceptResource]) -> None:
        self._by_concept: dict[str, ConceptResource] = {}
        for entry in resources:
            if entry.concept_id in self._by_concept:
                raise DomainError(
                    f"duplicate resource for concept: {entry.concept_id!r}. One "
                    f"concept has one statement of its rule; two would mean the "
                    f"learner's support depended on which was loaded first."
                )
            self._by_concept[entry.concept_id] = entry

    @classmethod
    def from_yaml(cls, path: Path) -> YamlConceptResources:
        entries = []
        for entry in _load_yaml(path, "resources"):
            missing = {
                "concept_id",
                "formula",
                "worked_example",
                "example_answer",
            } - set(entry)
            if missing:
                raise DomainError(
                    f"resource for {entry.get('concept_id', '<unnamed>')!r} is "
                    f"missing {sorted(missing)}. 'example_answer' is required "
                    f"even though it is never shown: it is what lets the "
                    f"validator check the example does not solve an item."
                )
            entries.append(
                ConceptResource(
                    concept_id=entry["concept_id"],
                    formula=entry["formula"],
                    worked_example=entry["worked_example"],
                    example_answer=entry["example_answer"],
                    source=entry.get("source", ""),
                )
            )
        return cls(entries)

    def all(self) -> Sequence[ConceptResource]:
        return tuple(self._by_concept.values())

    def get(self, concept_id: str) -> ConceptResource:
        try:
            return self._by_concept[concept_id]
        except KeyError:
            raise unknown_id_error("resource", concept_id, self._by_concept) from None

    def for_concept(self, concept_id: str) -> ConceptResource | None:
        return self._by_concept.get(concept_id)

    def content_hash(self) -> str:
        # Sorted, so reordering the file does not change the hash — but changing
        # what a learner would be shown does.
        return hash_content(
            *(
                f"{r.concept_id}|{r.formula}|{r.worked_example}|{r.example_answer}"
                for r in sorted(self._by_concept.values(), key=lambda r: r.concept_id)
            )
        )


class YamlItemBank:
    """Practice, pre-test and post-test items loaded from ``items/*.yaml``."""

    def __init__(self, items: Sequence[Item]) -> None:
        self._by_id: dict[str, Item] = {}
        for item in items:
            if item.id in self._by_id:
                raise DomainError(f"duplicate item id: {item.id!r}")
            self._by_id[item.id] = item

    @classmethod
    def from_dir(cls, directory: Path) -> YamlItemBank:
        if not directory.is_dir():
            raise DomainError(f"missing items directory: {directory}")
        paths = sorted(directory.glob("*.yaml"))
        if not paths:
            raise DomainError(f"no item files in {directory}")
        items: list[Item] = []
        for path in paths:
            for entry in _load_yaml(path, "items"):
                items.append(
                    Item(
                        id=entry["id"],
                        concept_id=entry["concept_id"],
                        prompt=entry["prompt"],
                        answer=str(entry["answer"]),
                        bank=entry.get("bank", "practice"),
                        probes=tuple(entry.get("probes", ()) or ()),
                        params=dict(entry.get("params", {}) or {}),
                    )
                )
        return cls(items)

    def all(self) -> Sequence[Item]:
        return tuple(self._by_id.values())

    def get(self, item_id: str) -> Item:
        try:
            return self._by_id[item_id]
        except KeyError:
            raise unknown_id_error("item", item_id, self._by_id) from None

    def bank(self, bank: Bank) -> Sequence[Item]:
        return tuple(i for i in self._by_id.values() if i.bank == bank)

    def for_concept(self, concept_id: str, bank: Bank = "practice") -> Sequence[Item]:
        return tuple(
            i for i in self._by_id.values() if i.concept_id == concept_id and i.bank == bank
        )

    def probing(self, misconception_id: str, bank: Bank | None = None) -> Sequence[Item]:
        return tuple(
            i
            for i in self._by_id.values()
            if misconception_id in i.probes and (bank is None or i.bank == bank)
        )

    def content_hash(self) -> str:
        return hash_content(
            *(
                f"{i.id}|{i.concept_id}|{i.bank}|{i.prompt}|{i.answer}|{','.join(sorted(i.probes))}"
                for i in sorted(self._by_id.values(), key=lambda i: i.id)
            )
        )
