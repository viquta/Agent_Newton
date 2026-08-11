"""Domain content validation.

Run this after any edit to a domain's YAML or rules. It is what makes "add more
later" safe rather than a source of silent drift that invalidates results: a buggy
rule that quietly produces the *correct* answer, or a misconception with no
post-test item, would otherwise surface as an inexplicable number at write-up.

Two checks do most of the work, because they compare content against the
verifier rather than against itself:

* :data:`ANSWERS_VERIFY` — every item's stated answer is judged correct.
* :data:`RULES_PRODUCE_ERRORS` — every misconception an item claims to probe
  actually yields a response the verifier judges incorrect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_newton.domains.base import Domain, DomainError, Verdict

ANSWERS_VERIFY = "answers_verify"
RULES_PRODUCE_ERRORS = "rules_produce_errors"
UNCONFIRMED_SOURCE = "unconfirmed_source"
GOALS_ARE_REACHABLE = "goals_are_reachable"
CONCEPT_HAS_A_LABEL = "concept_has_a_label"
TEMPLATES_ARE_SOUND = "templates_are_sound"

#: Draws checked per template. A learner works a concept until it is mastered,
#: which in a full session runs to a handful of repetitions on the hardest ones;
#: this covers that with room over.
VARIANT_DRAWS = 8

#: Marker for a catalogue entry whose literature source is not yet confirmed.
#: Reported as a warning, not a failure: the entry works and the domain loads,
#: but it is not yet defensible as a *documented* error. Shipping it visibly
#: incomplete beats shipping it with an attribution nobody checked.
NEEDS_SOURCE = "NEEDS-SOURCE"


@dataclass(frozen=True, slots=True)
class Problem:
    check: str
    message: str

    def __str__(self) -> str:
        return f"[{self.check}] {self.message}"


@dataclass
class ValidationReport:
    domain: str
    problems: list[Problem] = field(default_factory=list)
    warnings: list[Problem] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the content is *consistent*. Warnings do not affect this."""
        return not self.problems

    def add(self, check: str, message: str) -> None:
        self.problems.append(Problem(check, message))

    def warn(self, check: str, message: str) -> None:
        self.warnings.append(Problem(check, message))


def validate(domain: Domain) -> ValidationReport:
    """Check a domain's content for internal consistency."""
    report = ValidationReport(domain=domain.name)
    concepts = domain.concepts
    catalogue = domain.misconceptions
    items = domain.items

    concept_ids = set(concepts.ids())
    misconception_ids = set(catalogue.ids())

    report.stats = {
        "concepts": len(concept_ids),
        "misconceptions": len(misconception_ids),
        "items": len(items.all()),
        "practice": len(items.bank("practice")),
        "pretest": len(items.bank("pretest")),
        "posttest": len(items.bank("posttest")),
    }

    # --- referential integrity ------------------------------------------------
    # The graph is already checked for acyclicity and dangling prerequisites at
    # load time; YamlConceptGraph raises rather than constructing.
    for misconception in catalogue.all():
        if misconception.concept_id not in concept_ids:
            report.add(
                "misconception_concept_exists",
                f"misconception {misconception.id!r} references unknown concept "
                f"{misconception.concept_id!r}",
            )
        if not misconception.source.strip():
            report.add(
                "misconception_has_source",
                f"misconception {misconception.id!r} has an empty source; the "
                f"catalogue must stay traceable to the literature",
            )
        elif misconception.source.strip().upper().startswith(NEEDS_SOURCE):
            report.warn(
                UNCONFIRMED_SOURCE,
                f"misconception {misconception.id!r} has no confirmed source yet: "
                f"{misconception.source.strip()}",
            )
        if domain.buggy_rule(misconception.id) is None:
            report.add(
                "misconception_has_rule",
                f"misconception {misconception.id!r} has no registered buggy rule, "
                f"so no simulated learner can ever exhibit it",
            )

    # --- diagnosable concepts -------------------------------------------------
    # A concept with practice items but no catalogue entry cannot be diagnosed
    # coherently: the label space offered to the diagnostic is the whole
    # catalogue, so every label available for a wrong answer here belongs to a
    # different concept, and the agent reaches for the nearest plausible one
    # rather than abstaining. Nothing else catches this — the referential checks
    # above only look from the misconception outwards, and a concept with no
    # entry references nothing.
    #
    # A warning rather than a problem. A concept may legitimately carry no
    # documented misconception; what must not happen is that nobody noticed.
    labelled = {misconception.concept_id for misconception in catalogue.all()}
    for concept_id in concepts.ids():
        if concept_id in labelled:
            continue
        practice = items.for_concept(concept_id, "practice")
        if practice:
            report.warn(
                CONCEPT_HAS_A_LABEL,
                f"concept {concept_id!r} has {len(practice)} practice item(s) but "
                f"no misconception in the catalogue, so a wrong answer on it can "
                f"only be labelled with another concept's misconception",
            )

    for item in items.all():
        if item.concept_id not in concept_ids:
            report.add(
                "item_concept_exists",
                f"item {item.id!r} references unknown concept {item.concept_id!r}",
            )
        for probed in item.probes:
            if probed not in misconception_ids:
                report.add(
                    "item_probes_exist",
                    f"item {item.id!r} probes unknown misconception {probed!r}",
                )

    # --- goals ----------------------------------------------------------------
    # A goal is what makes planning directed, so a bad one does not fail loudly:
    # it silently narrows or empties the set of concepts the planner will
    # consider, and the run looks like it worked.
    goals = list(concepts.goals())
    report.stats["goals"] = len(goals)

    if not goals:
        report.add(
            GOALS_ARE_REACHABLE,
            "the graph declares no goals and has no sinks, so nothing can be "
            "planned toward",
        )

    for goal in goals:
        if goal not in concept_ids:
            report.add(
                GOALS_ARE_REACHABLE,
                f"goal {goal!r} is not a concept in this graph",
            )
            continue
        # Every concept on the way to a goal must be teachable, or the planner
        # reaches a concept it cannot give work for and the goal is unreachable
        # in practice while looking reachable in the graph.
        relevant = concepts.all_prerequisites(goal) | {goal}
        for concept_id in sorted(relevant):
            if not items.for_concept(concept_id, "practice"):
                report.add(
                    GOALS_ARE_REACHABLE,
                    f"goal {goal!r} requires concept {concept_id!r}, which has no "
                    f"practice items; the goal cannot be reached",
                )

    # --- coverage -------------------------------------------------------------
    for concept_id in concept_ids:
        if not items.for_concept(concept_id, "practice"):
            report.add(
                "concept_has_practice_items",
                f"concept {concept_id!r} has no practice items, so the planner can "
                f"select it but never teach it",
            )

    for misconception_id in misconception_ids:
        for bank in ("pretest", "posttest"):
            probing = [i for i in items.bank(bank) if misconception_id in i.probes]
            if not probing:
                report.add(
                    "misconception_probed_in_tests",
                    f"misconception {misconception_id!r} is not probed by any {bank} "
                    f"item, so its remediation rate cannot be measured",
                )

    # --- held-out separation --------------------------------------------------
    practice_prompts = {i.prompt.strip() for i in items.bank("practice")}
    for bank in ("pretest", "posttest"):
        for item in items.bank(bank):
            if item.prompt.strip() in practice_prompts:
                report.add(
                    "test_items_held_out",
                    f"{bank} item {item.id!r} repeats a practice prompt verbatim; "
                    f"post-test performance would measure recall, not transfer",
                )

    # --- content checked against the verifier ---------------------------------
    for item in items.all():
        result = domain.verifier.verify(item, item.answer)
        if result.verdict is not Verdict.CORRECT:
            report.add(
                ANSWERS_VERIFY,
                f"item {item.id!r}: stated answer {item.answer!r} verifies as "
                f"{result.verdict.value}, not correct"
                + (f" ({result.detail})" if result.detail else ""),
            )

    for item in items.all():
        for probed in item.probes:
            rule = domain.buggy_rule(probed)
            if rule is None:
                continue  # already reported above
            wrong = rule.apply(item)
            if wrong is None:
                report.add(
                    RULES_PRODUCE_ERRORS,
                    f"item {item.id!r} claims to probe {probed!r}, but that rule "
                    f"cannot produce a response for it (missing params?)",
                )
                continue
            verdict = domain.verifier.verify(item, wrong).verdict
            if verdict is not Verdict.INCORRECT:
                report.add(
                    RULES_PRODUCE_ERRORS,
                    f"item {item.id!r}: buggy rule {probed!r} produced {wrong!r}, "
                    f"which verifies as {verdict.value} — a misconception that yields "
                    f"a correct answer would be invisible to the diagnostic agent",
                )

    # --- item variants --------------------------------------------------------
    # A template regenerates prompt, answer and params together. Getting one of
    # them wrong produces a question that still looks fine and an answer key that
    # no longer matches it, or a buggy rule computing against numbers from a
    # different question — a learner would be told they were wrong when they were
    # right. Every draw is therefore put through the same two checks the written
    # items get, rather than trusting the arithmetic.
    report.stats["templates"] = len(domain.templates)
    for item_id, template in sorted(domain.templates.items()):
        try:
            base = items.get(item_id)
        except DomainError:
            report.add(
                TEMPLATES_ARE_SOUND,
                f"template names item {item_id!r}, which is not in the bank",
            )
            continue

        # The template, not ``Domain.variant`` — that short-circuits draw 0 and
        # would return ``base`` whatever the template does, so checking through
        # it would be a guard that could never fail.
        if template.variant(base, 0) != base:
            report.add(
                TEMPLATES_ARE_SOUND,
                f"template for {item_id!r} does not reproduce the item as written at "
                f"draw 0; the YAML would stop being the definition and anything "
                f"referring to this item by id would change meaning",
            )

        seen = {base.prompt}
        for draw in range(1, VARIANT_DRAWS):
            variant = template.variant(base, draw)
            if variant.id != base.id or variant.concept_id != base.concept_id:
                report.add(
                    TEMPLATES_ARE_SOUND,
                    f"template for {item_id!r} changed the id or concept at draw "
                    f"{draw}; a variant is the same item asked again, and the "
                    f"session counts repetitions by id",
                )
            if variant.prompt in seen:
                report.add(
                    TEMPLATES_ARE_SOUND,
                    f"template for {item_id!r} repeats a prompt at draw {draw}, so "
                    f"the learner is asked the same question again anyway",
                )
            seen.add(variant.prompt)

            verdict = domain.verifier.verify(variant, variant.answer).verdict
            if verdict is not Verdict.CORRECT:
                report.add(
                    TEMPLATES_ARE_SOUND,
                    f"template for {item_id!r} at draw {draw}: stated answer "
                    f"{variant.answer!r} verifies as {verdict.value}",
                )
            for probed in variant.probes:
                rule = domain.buggy_rule(probed)
                if rule is None:
                    continue
                wrong = rule.apply(variant)
                if wrong is None or domain.verifier.verify(variant, wrong).verdict is not (
                    Verdict.INCORRECT
                ):
                    report.add(
                        TEMPLATES_ARE_SOUND,
                        f"template for {item_id!r} at draw {draw}: rule {probed!r} "
                        f"produced {wrong!r}, which is not an error on this variant — "
                        f"the params and the question have drifted apart",
                    )

    return report
