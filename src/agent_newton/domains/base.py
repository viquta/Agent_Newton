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

Two further members are optional and a domain may omit both: ``ItemTemplate``
generates numeric variants of a repeated item, and ``ConceptResources`` supplies
what may be shown beside a question. A domain offering neither behaves exactly
as domains did before either existed.

Student responses cross this boundary as ``str``. Domains parse their own
notation internally. That keeps responses directly serialisable into the audit
log, which a generic response type would not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal, Protocol, Sequence, runtime_checkable

Bank = Literal["practice", "pretest", "posttest"]

#: Text that will not survive the trip to a learner.
#:
#: A backslash command, or the control character one becomes once a JSON reply
#: is unescaped. Both are the same defect seen at two stages: a tutor wrote
#: ``\frac{f(b) - f(a)}{b - a}``, ``\f`` parsed as a form feed, and the person
#: reading it saw ``rac{f(b) - f(a)}{b - a}`` and could not tell it meant a
#: division. They said so, which is the only reason it was found.
#:
#: Defined here rather than beside either check because two things need it and
#: they sit on opposite sides of the boundary: authored content, checked by
#: ``domain validate``, and generated replies, checked by the tutor evaluation.
#: One pattern, so a fix to it is a fix to both.
PLAIN_TEXT_ONLY = re.compile(r"\\[A-Za-z]|[\x00-\x08\x0b-\x1f]")


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
class ConceptResource:
    """The material shown *with* a question, for a learner well below the band.

    Separate from :class:`Misconception`, which describes what goes wrong. This
    describes what is right, and it exists because those are answers to
    different questions: a catalogue entry is only reachable once a learner has
    already erred, and a learner far from mastery may need the rule before they
    can err informatively at all.

    Keyed on the concept rather than the item, deliberately. An item's own
    worked solution is its answer, and showing it is the failure a sitting
    described as the system doing the question for them. A concept-level example
    carries its own numbers, so it can be shown beside any item on that concept
    without disclosing one.

    ``example_answer`` is the example's own result, and it is here so the
    validator can check it: ``domain validate`` refuses a resource whose answer
    verifies as any item's on that concept, through the domain's own verifier
    rather than by comparing strings.

    ``source`` follows the catalogue's convention — say where the statement of
    the rule comes from, so content stays traceable.
    """

    concept_id: str
    #: The rule itself, stated plainly. Shown from ``theta_lower`` down.
    formula: str
    #: A solved instance, on numbers no item on this concept uses. Shown from
    #: ``theta_lower / 2`` down, together with the formula.
    worked_example: str
    #: What ``worked_example`` comes out at. Checked, never shown on its own.
    example_answer: str
    source: str = ""

    # -- the lesson -------------------------------------------------------
    #
    # Two optional fields rather than a separate protocol, because everything a
    # lesson needs around it already exists here: one entry per concept, the
    # plain-text rule, the example validated against every item and every
    # template draw, a content hash and a column in the store. A second
    # structure would have duplicated all of it to hold two strings.
    #
    # What they add is a *kind* of support the artifact had none of. ``formula``
    # and ``worked_example`` state what to do; these state what the thing is and
    # why it behaves that way. Every instructional move before this was a reply
    # to a failed step, so a learner who had never met a concept and one who
    # held a misconception about it were answered identically — and a sitting
    # recorded the cost: someone asked what sin(x) was three times, in three
    # channels, and was told about the product rule each time, because that was
    # all there was to tell them.

    #: What the concept *is*, in plain words, before any rule for using it.
    #: Empty means this concept has no lesson, which is an ordinary state:
    #: ``domain validate`` warns so nobody is surprised, and never refuses.
    what_it_means: str = ""
    #: Why it behaves the way it does. Separate from :attr:`what_it_means`
    #: because they answer different questions and a learner can need one
    #: without the other — and because a definition that runs straight into a
    #: justification reads as one long paragraph nobody finishes.
    why_it_works: str = ""

    @property
    def teaches(self) -> bool:
        """Whether this concept has a lesson, as opposed to only a rule.

        Keyed on :attr:`what_it_means` alone. A concept can be explained
        without a reason being offered for it, and some genuinely cannot be
        given one at this level; a justification with nothing to justify is the
        combination that makes no sense.
        """
        return bool(self.what_it_means.strip())

    def shown(self, with_example: bool) -> str:
        """The text a learner reads, at one of the two depths.

        Here rather than at the call sites because there are two of them — the
        session, which records what was offered, and the front end, which
        displays it — and a learner reading one thing while the audit log
        records another would be unnoticeable and would make the record worth
        nothing.

        The example is labelled, and the label is load-bearing rather than
        decorative: an unlabelled solved problem sitting directly above a
        question is an invitation to copy its answer into the box. It says the
        numbers are different because that is the one thing the reader has to
        know before reading it.
        """
        if not with_example:
            return self.formula
        return (
            f"{self.formula}\n\n"
            f"An example — not your question, and the numbers are different:\n"
            f"{self.worked_example}"
        )

    def lesson(self) -> str:
        """The text of the lesson, composed here for the same reason as above.

        The session records what was taught and the front end displays it, and
        those must not be able to disagree.

        Reuses ``worked_example`` rather than carrying one of its own. It is
        already checked not to answer any item on the concept, at any template
        draw, so a lesson built on it inherits that guarantee instead of needing
        a second one — and a learner meeting the same worked example beside the
        question and inside the lesson is being shown a consistent thing, not a
        repetitive one.
        """
        parts = [self.what_it_means.strip()]
        if self.why_it_works.strip():
            parts.append(self.why_it_works.strip())
        parts.append(self.formula.strip())
        parts.append(
            f"An example — not your question, and the numbers are different:\n"
            f"{self.worked_example}"
        )
        return "\n\n".join(part for part in parts if part)


@runtime_checkable
class ConceptResources(Protocol):
    """The resources a domain offers, if it offers any.

    Optional, like :class:`BuggyRule` and :class:`ItemTemplate`. A domain with
    no resources simply never shows anything beside a question, which is what
    every domain did before this existed — so this is an improvement to content
    rather than a requirement on it.
    """

    def all(self) -> Sequence[ConceptResource]: ...
    def get(self, concept_id: str) -> ConceptResource: ...
    def for_concept(self, concept_id: str) -> ConceptResource | None: ...
    def content_hash(self) -> str: ...


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


@runtime_checkable
class ItemTemplate(Protocol):
    """Generates numeric variants of one item, so repeated practice varies.

    A concept the learner has not yet mastered is worked until it is, and most
    concepts carry a single item — so without this the same question is asked
    verbatim until the posterior clears the band. A person told us what that is:
    memorising an answer, not learning a method.

    ``prompt``, ``answer`` and ``params`` are regenerated **together** from the
    same draw. Regenerating any one alone would desynchronise the buggy rules,
    which compute against ``params``, from the question actually asked.

    Two properties implementations must hold, both checked by the validator:

    * **Draw 0 is the item as written.** The YAML stays the readable
      definition, and anything referring to an item by id — the verifier gold
      set, a stored transcript — keeps meaning what it meant.
    * **Deterministic in the draw.** Same item, same draw, same variant. The
      draw is the repetition count, which the session already tracks, so no
      generator state is involved and a seeded cohort stays reproducible.
    """

    @property
    def item_id(self) -> str: ...

    def variant(self, base: Item, draw: int) -> Item:
        """The ``draw``-th version of ``base``. ``draw == 0`` returns it as written."""
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
    templates: dict[str, ItemTemplate] = field(default_factory=dict)
    #: What may be shown beside a question. None for a domain that offers
    #: nothing, which is every domain before this existed.
    resources: ConceptResources | None = None

    def buggy_rule(self, misconception_id: str) -> BuggyRule | None:
        return self.buggy_rules.get(misconception_id)

    def variant(self, item: Item, draw: int) -> Item:
        """The ``draw``-th version of ``item``, or ``item`` if it has no template.

        Optional by design: an item with no template is simply asked again as
        written, which is what every item did before templates existed. That
        keeps this an improvement to content rather than a requirement on it.
        """
        template = self.templates.get(item.id)
        if template is None or draw == 0:
            return item
        return template.variant(item, draw)

    def resource_for(self, concept_id: str) -> ConceptResource | None:
        """What may be shown with a question on this concept, if anything.

        None is an ordinary answer, not a failure: a domain need not offer
        resources, and a concept within one that does need not have an entry.
        ``domain validate`` warns about the second case rather than refusing it,
        because what must not happen is that nobody noticed.
        """
        if self.resources is None:
            return None
        return self.resources.for_concept(concept_id)

    def content_hashes(self) -> dict[str, str]:
        """Hashes for the run manifest.

        Recorded per run so analysis can refuse to pool results produced against
        different ground truth.
        """
        hashes = {
            "concept_graph_hash": self.concepts.content_hash(),
            "catalogue_hash": self.misconceptions.content_hash(),
            "item_bank_hash": self.items.content_hash(),
        }
        # Emitted only when there are resources, so every manifest written
        # before they existed stays byte-identical and comparable. The key is
        # here at all because of what its absence cost the templates: a template
        # change alters what a learner is asked, and nothing refuses to pool
        # across one, so provenance survives only through the run's git SHA.
        # Resources are the same kind of content and do not repeat that.
        if self.resources is not None:
            hashes["resources_hash"] = self.resources.content_hash()
        return hashes


class DomainError(Exception):
    """Raised when domain content is malformed or inconsistent."""


def unknown_id_error(kind: str, bad: str, known: Iterable[str]) -> DomainError:
    options = ", ".join(sorted(known))
    return DomainError(f"unknown {kind} {bad!r}; known: {options}")
