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

from agent_newton.domains.base import Domain, Verdict

ANSWERS_VERIFY = "answers_verify"
RULES_PRODUCE_ERRORS = "rules_produce_errors"


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
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def add(self, check: str, message: str) -> None:
        self.problems.append(Problem(check, message))


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
        if domain.buggy_rule(misconception.id) is None:
            report.add(
                "misconception_has_rule",
                f"misconception {misconception.id!r} has no registered buggy rule, "
                f"so no simulated learner can ever exhibit it",
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

    return report
