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

import re
from dataclasses import dataclass, field

from agent_newton.domains.base import (
    PLAIN_TEXT_ONLY,
    Domain,
    DomainError,
    Verdict,
)

ANSWERS_VERIFY = "answers_verify"
RULES_PRODUCE_ERRORS = "rules_produce_errors"
UNCONFIRMED_SOURCE = "unconfirmed_source"
GOALS_ARE_REACHABLE = "goals_are_reachable"
CONCEPT_HAS_A_LABEL = "concept_has_a_label"
TEMPLATES_ARE_SOUND = "templates_are_sound"
GUESSABLE_FAMILY = "guessable_family"
CONCEPT_HAS_A_RESOURCE = "concept_has_a_resource"
CONCEPT_HAS_A_LESSON = "concept_has_a_lesson"
RESOURCE_KEEPS_ITS_DISTANCE = "resource_keeps_its_distance"
RESOURCE_IS_PLAIN_TEXT = "resource_is_plain_text"
ANSWERS_ARE_UNAMBIGUOUS = "answers_are_unambiguous"

#: Draws checked per template. A learner works a concept until it is mastered,
#: which in a full session runs to a handful of repetitions on the hardest ones;
#: this covers that with room over.
VARIANT_DRAWS = 8

#: Draws checked when asking whether a resource's example solves an item.
#:
#: Far deeper than ``VARIANT_DRAWS``, and it has to be. A template family is
#: unbounded — the implicit-differentiation family generates ``-x/(k*y)`` for
#: every k — so a worked example can sit clear of the first eight draws and
#: collide at the ninth. One did, while this branch was being written, and eight
#: draws would have shipped it.
#:
#: ⚠️ **This is a bound, not a proof.** No finite depth can rule out a collision
#: against an unbounded family. It is set well above the repetitions a session
#: reaches — a stuck learner was measured at 30 items on one concept — so a
#: collision inside a real sitting is what it actually excludes.
RESOURCE_DRAWS = 64

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

        _check_guessable(report, item_id, template, base)

    _check_resources(domain, report)
    _check_unambiguous_answers(domain, report)

    return report


def _check_unambiguous_answers(domain: Domain, report: ValidationReport) -> None:
    """No stated answer may be written so that it has two readings.

    ``a/bc`` means ``(a/b)*c`` by formal precedence and ``a/(bc)`` in ordinary
    mathematical writing. A learner writing one is told so and asked to bracket
    it — the verifier cannot know which was meant. An *item* has no such excuse:
    it is the thing being compared against, and an ambiguous one would be
    compared under whichever reading the parser happened to take.

    Reaches the domain through an optional hook, so a domain whose notation has
    no such ambiguity simply declines to look.

    ⚠️ **Answers only, never prose.** The first version also read the resources'
    formula and worked-example text, and flagged a sentence containing
    ``dx = du/5.`` — the scanner is looking for expression structure and prose
    supplies it by accident, in every sentence with a slash in it. A check that
    reports its own noise gets ignored, and then it is not a check.
    """
    ambiguous = getattr(domain.verifier, "ambiguous_notation", None)
    if ambiguous is None:
        return

    def _check(what: str, where: str, text: str) -> None:
        other = ambiguous(text)
        if other is not None:
            report.add(
                ANSWERS_ARE_UNAMBIGUOUS,
                f"{what} {where!r} is written {text!r}, which also reads as "
                f"{other!r}. Bracket it: the comparison would otherwise depend "
                f"on which reading the parser takes.",
            )

    for item in domain.items.all():
        _check("the answer to", item.id, item.answer)
        template = domain.templates.get(item.id)
        if template is None:
            continue
        for draw in range(1, VARIANT_DRAWS):
            _check(f"draw {draw} of", item.id, template.variant(item, draw).answer)

    if domain.resources is not None:
        for resource in domain.resources.all():
            _check("the example answer for", resource.concept_id, resource.example_answer)


def _numbers(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", text)]


def _check_guessable(report: ValidationReport, item_id: str, template, base) -> None:  # noqa: ANN001
    """Warn when the answer follows from the prompt by a rule simpler than the concept.

    Every draw of a template is checked for being *correct*. None of them was
    checked for being **earnable**, and the difference is the whole point of a
    practice item.

    The family that prompted this generated only ``(n+1)x^n``, so the answer to
    ``3x^2`` was ``x^3 + C`` and the answer to ``4x^3`` was ``x^4 + C``. Read the
    coefficient, write it as the exponent, add C. Every draw individually
    correct, every draw answerable without knowing what an antiderivative is —
    and a learner said so at the keyboard while the mastery estimate believed
    them.

    **The signature is structural constancy.** Take the integers in the prompt
    and the integers in the answer. If every answer number sits at a fixed offset
    from a fixed prompt position, *and the same map holds across every draw*,
    then one surface rule answers the whole family. Rotating the answer's shape
    breaks the map, which is why varying only the numbers does not help.

    A **warning**, not a failure, and deliberately. Some concepts genuinely are
    one step — the map existing is evidence to look, not proof of a defect. What
    must not happen is that nobody notices.
    """
    # ⚠️ Draw 0 is excluded, and that makes the check *stronger*. Draw 0 is
    # whatever the YAML says — it is the item as written and the template must
    # reproduce it — so its numbers frequently do not fit the pattern the
    # template generates. Requiring the map to cover it would let a family that
    # is answerable by one rule at all seven generated draws escape on the
    # strength of the one draw the template did not choose.
    answers = [template.variant(base, d).answer for d in range(VARIANT_DRAWS)]

    # ⚠️ The blind spot this check had, and the strongest signal there is.
    #
    # The positional-map test below reads the *numbers* in the answer, so a
    # family whose answer contains none escapes it entirely. `ca_impl_p1` asked
    # "given x^2 + y^2 = 25, find dy/dx" and then 36, then 49 — and the answer is
    # `-x/y` every time, because the constant does not appear in the derivative.
    # That is not guessable by a rule, it is answerable from memory after one
    # exposure, and the check reported nothing.
    if len(set(answers)) == 1:
        report.warn(
            GUESSABLE_FAMILY,
            f"every draw of {item_id!r} has the same answer ({answers[0]!r}), so "
            f"after one exposure the rest are recall rather than practice. Vary "
            f"what the question is *about*, not the numbers it happens to carry",
        )
        return

    draws = []
    for draw in range(1, VARIANT_DRAWS):
        variant = template.variant(base, draw)
        draws.append((_numbers(variant.prompt), _numbers(variant.answer)))

    # Structure varies across draws: no single positional map can exist, and the
    # family is not guessable in this sense.
    if len({len(answer) for _, answer in draws}) != 1:
        return
    if not draws[0][1] or not draws[0][0]:
        return
    if len({len(prompt) for prompt, _ in draws}) != 1:
        return

    predicted = []
    for position in range(len(draws[0][1])):
        source = None
        for candidate in range(len(draws[0][0])):
            offset = draws[0][1][position] - draws[0][0][candidate]
            if all(
                answer[position] - prompt[candidate] == offset for prompt, answer in draws
            ):
                source = (candidate, offset)
                break
        if source is None:
            return
        predicted.append(source)

    shown = ", ".join(
        f"answer[{i}] = prompt[{c}]{f'{o:+d}' if o else ''}"
        for i, (c, o) in enumerate(predicted)
    )
    report.warn(
        GUESSABLE_FAMILY,
        f"every draw of {item_id!r} is answerable by one fixed rule ({shown}), so "
        f"the answer can be pattern-matched off the question without the concept. "
        f"Varying the numbers will not help — the shape is what is being matched; "
        f"rotate the shape across draws instead",
    )


def _check_resources(domain: Domain, report: ValidationReport) -> None:
    """The material shown beside a question, if the domain offers any.

    Three checks, and only the first is a warning. A domain need not offer
    resources and a concept within one need not have an entry — but a resource
    that gives away an item, or that arrives as mangled notation, is content
    that would teach the wrong thing, and content is where those are cheapest to
    catch.
    """
    resources = domain.resources
    if resources is None:
        return

    known = set(domain.concepts.ids())
    for resource in resources.all():
        if resource.concept_id not in known:
            report.add(
                CONCEPT_HAS_A_RESOURCE,
                f"resource names unknown concept {resource.concept_id!r}",
            )

    # A concept a learner can be given practice on, with nothing to show them
    # when the estimate says they are a long way below it. A warning: a concept
    # may genuinely need no statement beyond its questions. What must not happen
    # is that nobody noticed — the same reasoning as the catalogue gap above,
    # which is exactly the hole that let three concepts go undiagnosable.
    covered = {resource.concept_id for resource in resources.all()}
    for concept_id in domain.concepts.ids():
        if concept_id in covered:
            continue
        practice = domain.items.for_concept(concept_id, "practice")
        if practice:
            report.warn(
                CONCEPT_HAS_A_RESOURCE,
                f"concept {concept_id!r} has {len(practice)} practice item(s) but "
                f"no resource, so a learner well below the band is shown the "
                f"question and nothing else",
            )

    # A concept that can be practised and cannot be *explained*. Separate from
    # the warning above, and the distinction is the whole point of having added
    # lessons: a resource states the rule, which answers "what do I do here"; a
    # lesson states what the thing is and why, which answers "what is this". A
    # learner who has never met the concept needs the second, and until now the
    # system had nothing to give them — a sitting recorded someone asking what
    # sin(x) was three times and being told about the product rule each time,
    # because the product rule was all there was to tell them.
    #
    # A warning rather than a failure, like every other content gap here. Some
    # concepts may genuinely not want one. What must not happen is that nobody
    # noticed.
    for concept_id in sorted(domain.concepts.ids()):
        resource = resources.for_concept(concept_id)
        if resource is None or resource.teaches:
            continue
        if domain.items.for_concept(concept_id, "practice"):
            report.warn(
                CONCEPT_HAS_A_LESSON,
                f"concept {concept_id!r} has a rule but no lesson, so a learner "
                f"who has never met it can be told what to do and never what it "
                f"is",
            )

    for resource in resources.all():
        # The lesson fields join the sweep. They are the longest prose a learner
        # reads anywhere in the system, so they are the likeliest place for a
        # backslash to arrive — and the damage is invisible to whoever wrote it,
        # because it only appears once the text has been through JSON.
        for field_name in (
            "formula",
            "worked_example",
            "what_it_means",
            "why_it_works",
        ):
            text = getattr(resource, field_name)
            if PLAIN_TEXT_ONLY.search(text):
                report.add(
                    RESOURCE_IS_PLAIN_TEXT,
                    f"resource for {resource.concept_id!r} has a backslash "
                    f"command or a control character in {field_name!r}; a learner "
                    f"read one of these as 'rac{{f(b) - f(a)}}{{b - a}}' and could "
                    f"not tell it meant a division",
                )

        # ⚠️ The one that matters. An example whose answer *is* an item's answer
        # is the answer with extra steps, and it would be shown before the
        # learner had attempted anything — strictly worse than the worked step a
        # person already described as the system cheating for them.
        #
        # Asked of the domain's own verifier rather than by comparing strings,
        # for the reason `answer_leaked` gives: `5x^4` and `5*x**4` are the same
        # disclosure, and a string comparison sees two different texts.
        for item in _every_form_of(domain, resource.concept_id):
            if domain.verifier.verify(item, resource.example_answer).verdict is Verdict.CORRECT:
                report.add(
                    RESOURCE_KEEPS_ITS_DISTANCE,
                    f"the worked example for {resource.concept_id!r} answers "
                    f"{item.id!r}: showing it would hand over that item before "
                    f"the learner had attempted it. Change the example's numbers.",
                )
                break


def _every_form_of(domain: Domain, concept_id: str):
    """Every question a learner could be asked on this concept, banks and draws.

    Templated items are expanded, because the variant is what a learner actually
    sees on a repetition — checking the item as written would clear an example
    that solves the fourth time it is asked.
    """
    for bank in ("practice", "pretest", "posttest"):
        for item in domain.items.for_concept(concept_id, bank):
            yield item
            template = domain.templates.get(item.id)
            if template is None:
                continue
            for draw in range(1, RESOURCE_DRAWS):
                yield template.variant(item, draw)
