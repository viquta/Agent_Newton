"""Tutor quality, measured on the turns a learner would actually read.

The instructional rules are enforced as predicates over the shared state, and
those predicates are tested as the properties they claim to be. That establishes
that a turn was *assigned* a support level and a move — not that the text
carrying them does what the level says. A ``NUDGE`` that states the answer
satisfies every predicate in ``core/pedagogy`` and defeats the rule it was
issued under.

So this measures the text. Two layers, and they are not interchangeable:

**Deterministic checks** decide things that have a right answer. Whether a hint
contains the answer is settled by the domain's own verifier, so a hint saying
``5x^4`` for an answer written ``5*x**4`` is caught the same as a verbatim copy —
the equivalence problem is already solved in this repository and is not solved
again here. These gate CI.

**Judged checks** decide things that do not. Whether a hint addresses the step
the learner actually wrote, and whether a worked step really gives more away than
a nudge, are judgements. They are scored by a second model against a
hand-labelled set, and the report carries **that model's agreement with the hand
labels** beside its verdicts. A judge whose agreement is not reported is an
unmeasured instrument, and its verdicts would be quoted as though they were
measurements.

Cases come from the item bank rather than from session transcripts, for the two
reasons ``evaluation/diagnostic`` gives: a session over-samples whatever sits
near the start of the syllabus, and a component figure should not depend on how a
planner happened to route anyone.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal, Sequence

import yaml
from pydantic import BaseModel, Field

from agent_newton.config import ZPDConfig
from agent_newton.core.agents.base import Diagnosis, Tutor
from agent_newton.core.agents.llm import FALLBACK_HINT
from agent_newton.core.evaluation.diagnostic import cases as wrong_answers
from agent_newton.core.pedagogy import HintLevel, TutorMove, Violation
from agent_newton.core.state.schema import Utterance
from agent_newton.core.state.views import FullStateView
from agent_newton.core.state.zpd import Frontier
from agent_newton.domains.base import Domain, DomainError, Item, Verdict
from agent_newton.llm.base import LLMProvider, MalformedResponse, complete

ANSWER_LEAKED = "answer_leaked"
LATEX_IN_REPLY = "latex_in_reply"
REFLECT_TELLS = "reflect_tells"
OVER_LENGTH = "over_length"

#: Sentences a turn may carry, per support level.
#:
#: Per level rather than one number, because the levels are asked for different
#: things. A worked step is instructed to name the rule, substitute the
#: student's own expression and show the working a line at a time; two sentences
#: cannot hold that. This check previously enforced two everywhere and would
#: have marked a proper worked step as a violation — penalising the tutor for
#: obeying the instruction it was given.
MAX_SENTENCES: dict[str, int] = {
    HintLevel.NUDGE.label: 2,
    HintLevel.TARGETED.label: 2,
    HintLevel.WORKED_STEP.label: 6,
}


# --- cases -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TurnCase:
    """One situation to put the tutor in.

    Holds the *inputs* to a turn, never the move or the level: those are the
    rules' decisions, and naming them here would let this drift from the policy
    the session actually runs. The harness reads them back off the reply.
    """

    id: str
    item_id: str
    concept_id: str
    misconception_id: str
    wrong_answer: str
    mastery: float
    prior_failures: int
    diagnosed: bool
    moves_so_far: tuple[TutorMove, ...] = ()
    working: str = ""


def _mastery_for(level: HintLevel, band: ZPDConfig) -> float:
    """A posterior that lands on this support level with no failed attempts.

    Derived from the band rather than written down, so a threshold sweep moves
    the case set with the policy instead of leaving it measuring the old one.

    "No failed attempts" is a situation a session actually reaches: the tutor is
    given the posterior as it stood when the question was posed, and the failure
    it is responding to is not counted twice. Until that was fixed these cases
    described a regime the running system never entered.
    """
    if level is HintLevel.NUDGE:
        return min(1.0, (band.theta_lower + 1.0) / 2)
    if level is HintLevel.TARGETED:
        return (band.theta_lower / 2 + band.theta_lower) / 2
    return band.theta_lower / 4


def cases(domain: Domain, band: ZPDConfig) -> list[TurnCase]:
    """Every situation worth asking the tutor about, from the bank.

    Seven per ``(item, misconception)`` pair: a plain hint at each support
    level, the reflective prompt the error-first rule requires once a
    misconception is confirmed, and the remediation that may follow it, again at
    each level. The two moves that carry a correction are covered at every level
    because that is where giving the answer away is either permitted or not.
    """
    built: list[TurnCase] = []
    for item_id, misconception_id, wrong in wrong_answers(domain):
        item = domain.items.get(item_id)
        for level in HintLevel:
            built.append(
                TurnCase(
                    id=f"{item_id}|{misconception_id}|hint|{level.label}",
                    item_id=item_id,
                    concept_id=item.concept_id,
                    misconception_id=misconception_id,
                    wrong_answer=wrong,
                    mastery=_mastery_for(level, band),
                    prior_failures=0,
                    diagnosed=False,
                )
            )
        built.append(
            TurnCase(
                id=f"{item_id}|{misconception_id}|reflect",
                item_id=item_id,
                concept_id=item.concept_id,
                misconception_id=misconception_id,
                wrong_answer=wrong,
                mastery=_mastery_for(HintLevel.TARGETED, band),
                prior_failures=0,
                diagnosed=True,
            )
        )
        for level in HintLevel:
            built.append(
                TurnCase(
                    id=f"{item_id}|{misconception_id}|remediate|{level.label}",
                    item_id=item_id,
                    concept_id=item.concept_id,
                    misconception_id=misconception_id,
                    wrong_answer=wrong,
                    mastery=_mastery_for(level, band),
                    prior_failures=0,
                    diagnosed=True,
                    moves_so_far=(TutorMove.REFLECT,),
                )
            )
    return built


def view_for(case: TurnCase, band: ZPDConfig) -> FullStateView:
    """The state a tutor would be holding in this situation.

    A real view rather than a stub, so the tutor's own reading of it — the
    posterior it takes the support level from, the words it reads back — is
    exercised rather than bypassed.
    """
    reflections = (
        (Utterance(text=case.working, item_id=case.item_id,
                   concept_id=case.concept_id, kind="working"),)
        if case.working
        else ()
    )
    return FullStateView(
        mastery={case.concept_id: case.mastery},
        error_trace=(),
        frontier=Frontier(frozenset({case.concept_id})),
        outcomes=(),
        version=1,
        plan=None,
        reflections=reflections,
    )


# --- turns -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Turn:
    """One reply, with the rules that governed it."""

    case: TurnCase
    move: str
    level: str
    targets: str | None
    text: str
    seconds: float

    @property
    def is_fallback(self) -> bool:
        """The model produced nothing usable and the tutor said so.

        Counted apart from a bad hint, the way the diagnostic agent's failed
        calls are counted apart from wrong labels: nothing was written, so
        scoring it as badly-pitched prose would be scoring an absence.
        """
        return self.text.strip() == FALLBACK_HINT


def ask(
    domain: Domain,
    tutor: Tutor,
    band: ZPDConfig,
    situations: Sequence[TurnCase],
    on_turn: Callable[[Turn], None] | None = None,
) -> Iterator[Turn]:
    """Put the tutor in each situation, yielding turns as they complete.

    A generator so a caller can write partial results: a full pass over the bank
    is minutes of inference, and it should not be all-or-nothing.
    """
    for case in situations:
        item = domain.items.get(case.item_id)
        diagnosis = (
            Diagnosis(case.misconception_id, confidence=1.0)
            if case.diagnosed
            else Diagnosis(None)
        )
        started = time.time()
        hint = tutor.respond(
            item,
            diagnosis,
            view_for(case, band),
            domain,
            response=case.wrong_answer,
            mastery=case.mastery,
            prior_failures=case.prior_failures,
            moves_this_item=list(case.moves_so_far),
        )
        turn = Turn(
            case=case,
            move=hint.move.value,
            level=hint.level.label,
            targets=hint.targets,
            text=hint.text,
            seconds=time.time() - started,
        )
        if on_turn is not None:
            on_turn(turn)
        yield turn


# --- deterministic checks ----------------------------------------------------

#: Runs of characters that could be an expression, **including the spaces an
#: operator is usually written with**. Deliberately generous: a fragment that is
#: not an expression comes back UNPARSEABLE from the verifier and costs nothing,
#: whereas a missed fragment is a leak nobody sees.
#:
#: Splitting on whitespace instead is the obvious version and it is wrong. Most
#: answers here have a space around their operator — ``x**3 + C``,
#: ``3*x**2 - 6*x`` — so a reply stating one verbatim came apart into ``x**3``,
#: ``+`` and ``C``, none of which verifies, and the leak went unreported.
_TERM = r"[A-Za-z0-9_.^*/()]+"
_MATHS = re.compile(rf"{_TERM}(?:\s*[-+*/^]\s*{_TERM})*")

#: A backslash command, or the control character one becomes after a JSON reply
#: has been unescaped. The second form is the one a learner actually saw:
#: ``\frac`` arrives as a form feed followed by ``rac``, and the division the
#: hint was explaining disappears.
_LATEX = re.compile(r"\\[A-Za-z]|[\x00-\x08\x0b-\x1f]")

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def _candidates(text: str, item: Item) -> list[str]:
    """Fragments of ``text`` that might be the answer.

    Fragments the question already contains are dropped. A number the learner
    was given cannot be given away again, and without this every hint that
    quotes the exercise reads as a leak.
    """
    prompt = " ".join(item.prompt.split())
    found = [
        fragment.strip(".,;:()")
        for fragment in _MATHS.findall(text)
    ]
    found = [f for f in found if f]
    # Answers may name several values ("0, 2"), so adjacent fragments are also
    # offered joined — otherwise a hint listing both roots looks like two wrong
    # answers rather than one right one. Built from every fragment, including
    # ones the question contains: "Solve 3x^2 - 6x = 0" holds a 0 and the answer
    # is "0, 2", so filtering first would drop the half that makes the pair.
    joined = [f"{a}, {b}" for a, b in zip(found, found[1:])]
    return [c for c in found + joined if c not in prompt]


def leaks_answer(text: str, item: Item, domain: Domain) -> bool:
    """Whether any fragment of ``text`` is the answer.

    Decided by the domain's verifier, so equivalence rather than spelling: a
    hint writing ``5x^4`` where the bank says ``5*x**4`` has given the same
    thing away.
    """
    for candidate in _candidates(text, item):
        if domain.verifier.verify(item, candidate).verdict is Verdict.CORRECT:
            return True
    return False


def check_turn(turn: Turn, item: Item, domain: Domain) -> tuple[Violation, ...]:
    """Every deterministic rule this turn breaks.

    A fallback is checked like any other reply. It has never broken one, and
    exempting it would mean the exemption is what the check was measuring.
    """
    found: list[Violation] = []
    text = turn.text

    if _LATEX.search(text):
        found.append(
            Violation(
                LATEX_IN_REPLY,
                "the reply contains a backslash command or a control character; "
                "replies arrive as JSON and a backslash command does not survive "
                "the unescaping",
            )
        )

    # A worked step is *told* to show the working, so the answer appearing in it
    # is the instruction being followed rather than broken.
    if turn.level != HintLevel.WORKED_STEP.label and leaks_answer(text, item, domain):
        found.append(
            Violation(
                ANSWER_LEAKED,
                f"a {turn.level} reply gives the answer away; the support level "
                f"chosen for this learner does not permit it",
            )
        )

    if turn.move == TutorMove.REFLECT.value and leaks_answer(text, item, domain):
        # The error-first rule inserts a step between confirming a misconception
        # and correcting it. A reflective prompt carrying the answer has skipped
        # that step whatever else it says.
        #
        # This checks only that. An earlier version also required a question
        # mark and it was wrong on the first real reply it saw: *"Please take
        # another look at your expression and tell me which part you feel least
        # confident about"* is a reflective prompt and has no question mark in
        # it. Whether a turn genuinely invites reflection is a judgement about
        # prose, and putting a punctuation test in its place would have measured
        # punctuation while reporting pedagogy.
        found.append(
            Violation(REFLECT_TELLS, "the reflective prompt gives the answer away")
        )

    allowed = MAX_SENTENCES.get(turn.level, 2)
    if len(_SENTENCE_END.findall(text)) > allowed:
        found.append(
            Violation(
                OVER_LENGTH,
                f"a {turn.level} reply runs to more than {allowed} sentences",
            )
        )

    return tuple(found)


@dataclass(frozen=True, slots=True)
class Checked:
    turn: Turn
    violations: tuple[Violation, ...]

    @property
    def clean(self) -> bool:
        return not self.violations


@dataclass
class TutorReport:
    """What the deterministic layer found."""

    checked: list[Checked] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.checked)

    @property
    def clean(self) -> int:
        return sum(1 for c in self.checked if c.clean)

    @property
    def fallbacks(self) -> int:
        """Turns where the model produced nothing and the tutor fell back."""
        return sum(1 for c in self.checked if c.turn.is_fallback)

    @property
    def clean_rate(self) -> float:
        return self.clean / self.total if self.total else 0.0

    def by_rule(self) -> dict[str, int]:
        return dict(
            Counter(v.rule for c in self.checked for v in c.violations)
        )

    def rate_by_rule(self) -> dict[str, float]:
        return {
            rule: count / self.total
            for rule, count in self.by_rule().items()
        } if self.total else {}

    def by_level(self) -> dict[str, dict[str, float]]:
        """Clean rate per support level.

        Split because the levels are asked for different things: leaking the
        answer is permitted at ``worked_step`` and forbidden below it, so a
        single figure would average a rule with its own exemption.
        """
        rows: dict[str, dict[str, float]] = {}
        for level in sorted({c.turn.level for c in self.checked}):
            scored = [c for c in self.checked if c.turn.level == level]
            rows[level] = {
                "turns": len(scored),
                "clean": sum(1 for c in scored if c.clean),
                "clean_rate": sum(1 for c in scored if c.clean) / len(scored),
            }
        return rows

    def by_move(self) -> dict[str, dict[str, float]]:
        rows: dict[str, dict[str, float]] = {}
        for move in sorted({c.turn.move for c in self.checked}):
            scored = [c for c in self.checked if c.turn.move == move]
            rows[move] = {
                "turns": len(scored),
                "clean": sum(1 for c in scored if c.clean),
                "clean_rate": sum(1 for c in scored if c.clean) / len(scored),
            }
        return rows

    def by_bank(self, domain: Domain) -> dict[str, dict[str, float]]:
        """Clean rate split by the bank each item came from.

        The whole-bank figure is the right component measure — every declared
        pair gets its share. But the tests are administered without hints, so
        only ``practice`` turns are ever read by a learner, and that row is the
        rate a running system exposes anyone to. The diagnostic evaluation draws
        the same distinction, and for the same reason: without it a reader
        compares a component figure against a system-level one and finds a gap
        nobody can explain.
        """
        rows: dict[str, dict[str, float]] = {}
        banked = [
            (domain.items.get(c.turn.case.item_id).bank, c) for c in self.checked
        ]
        for bank in sorted({b for b, _ in banked}):
            scored = [c for b, c in banked if b == bank]
            rows[bank] = {
                "turns": len(scored),
                "clean": sum(1 for c in scored if c.clean),
                "clean_rate": sum(1 for c in scored if c.clean) / len(scored),
            }
        return rows

    def failures(self) -> list[Checked]:
        return [c for c in self.checked if not c.clean]

    @property
    def seconds(self) -> float:
        return sum(c.turn.seconds for c in self.checked)


def score(domain: Domain, turns: Iterator[Turn] | Sequence[Turn]) -> TutorReport:
    """Run every deterministic check over a stream of turns."""
    report = TutorReport()
    for turn in turns:
        item = domain.items.get(turn.case.item_id)
        report.checked.append(Checked(turn, check_turn(turn, item, domain)))
    return report


# --- the judged layer --------------------------------------------------------

_JUDGE_SYSTEM = (
    "You check whether a tutor's reply is faithful to what a student actually "
    "wrote. You are not judging whether the reply is helpful, well written or "
    "mathematically elegant — only whether every claim it makes about the "
    "student's work is supported by the work shown."
)


class Groundedness(BaseModel):
    """Whether a reply asserts more about the student than their step shows."""

    grounded: bool = Field(
        description=(
            "True when every claim the reply makes about what the student did is "
            "visible in their step or their working. False when it asserts an "
            "operation, a correct part or an error the work does not show."
        )
    )
    reason: str = Field(default="", description="One sentence.")


@dataclass(frozen=True, slots=True)
class GoldTurn:
    """A hand-labelled ``(step, reply)`` pair, for scoring the judge."""

    id: str
    item_id: str
    response: str
    hint: str
    grounded: bool
    working: str = ""
    source: str = ""
    #: Why the label is what it is, where a reader might land elsewhere. Carried
    #: rather than left as a YAML comment so a disagreement can be read against
    #: the reasoning that produced the label instead of against nothing.
    note: str = ""


def load_gold(path: str | Path, domain: Domain) -> list[GoldTurn]:
    """Read a judge gold file, checking it against the domain it labels.

    Referential integrity here rather than at scoring time, so a renamed item
    fails loudly instead of silently shrinking the set.
    """
    path = Path(path)
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}

    raw = data.get("cases")
    if not raw:
        raise DomainError(f"{path} declares no cases")

    found: list[GoldTurn] = []
    seen: set[str] = set()
    for entry in raw:
        case = GoldTurn(
            id=str(entry["id"]),
            item_id=str(entry["item_id"]),
            response=str(entry["response"]),
            hint=str(entry["hint"]),
            grounded=bool(entry["grounded"]),
            working=str(entry.get("working", "")).strip(),
            source=str(entry.get("source", "")).strip(),
            note=str(entry.get("note", "")).strip(),
        )
        if case.id in seen:
            raise DomainError(f"{path}: duplicate case id {case.id!r}")
        seen.add(case.id)
        domain.items.get(case.item_id)  # raises on an unknown id
        found.append(case)
    return found


def judge_grounded(
    provider: LLMProvider,
    item: Item,
    response: str,
    hint: str,
    working: str = "",
) -> bool | None:
    """Whether the reply keeps to what the step shows. None when unobtainable.

    None rather than a default: a call that produced nothing is not a verdict,
    and folding it into either side would move the rate it is supposed to
    measure.
    """
    shown = f"\nThe student's working: {working}" if working else ""
    # The question is phrased the same way round as the field it fills. Asked
    # the other way — "does the reply claim anything the work does not show" —
    # a model answering correctly sets `grounded` to the opposite of what it
    # means, and the calibration run reads as a judge performing below chance
    # rather than as a prompt that inverts it. That happened; the gold set is
    # what showed it, which is the argument for calibrating at all.
    prompt = (
        f"Exercise: {' '.join(item.prompt.split())}\n"
        f"Correct answer: {item.answer}\n"
        f"The student wrote: {response}{shown}\n\n"
        f"The tutor replied: {hint}\n\n"
        f"Is every claim the tutor's reply makes about the student's work "
        f"supported by the work shown above?"
    )
    try:
        reply = complete(provider, prompt, Groundedness, system=_JUDGE_SYSTEM)
    except MalformedResponse:
        return None
    return reply.grounded


@dataclass
class JudgeReport:
    """The judge's verdicts, and its agreement with the hand labels.

    Both are reported. The verdicts are the measurement of the tutor; the
    agreement is the measurement of the instrument, and quoting the first
    without the second states a rate whose error is unknown.
    """

    #: (case id, hand label, judged) over the gold set. None where the call
    #: failed and nothing was judged.
    calibration: list[tuple[str, bool, bool | None]] = field(default_factory=list)
    #: (case id, judged) over the generated turns.
    verdicts: list[tuple[str, bool | None]] = field(default_factory=list)

    @property
    def scored(self) -> list[tuple[str, bool, bool]]:
        return [(i, h, j) for i, h, j in self.calibration if j is not None]

    @property
    def agreement(self) -> float:
        scored = self.scored
        return sum(1 for _, h, j in scored if h == j) / len(scored) if scored else 0.0

    @property
    def unobtainable(self) -> int:
        return sum(1 for _, _, j in self.calibration if j is None) + sum(
            1 for _, j in self.verdicts if j is None
        )

    @property
    def grounded_rate(self) -> float:
        judged = [j for _, j in self.verdicts if j is not None]
        return sum(1 for j in judged if j) / len(judged) if judged else 0.0

    def disagreements(self) -> list[tuple[str, bool, bool]]:
        """Gold cases the judge read differently. Read these before the rate."""
        return [(i, h, j) for i, h, j in self.scored if h != j]


def calibrate(
    provider: LLMProvider, domain: Domain, gold: Sequence[GoldTurn]
) -> JudgeReport:
    """Score the judge against the hand labels, before it scores anything else."""
    report = JudgeReport()
    for case in gold:
        item = domain.items.get(case.item_id)
        report.calibration.append(
            (
                case.id,
                case.grounded,
                judge_grounded(provider, item, case.response, case.hint, case.working),
            )
        )
    return report


def judge_turns(
    provider: LLMProvider,
    domain: Domain,
    turns: Sequence[Turn],
    report: JudgeReport | None = None,
) -> JudgeReport:
    """Ask the judge about each generated turn.

    Reflective prompts are skipped: they make no claim about the step by design,
    so groundedness has nothing to bite on and including them would dilute the
    rate with cases that cannot fail.
    """
    report = report or JudgeReport()
    for turn in turns:
        if turn.move == TutorMove.REFLECT.value or turn.is_fallback:
            continue
        item = domain.items.get(turn.case.item_id)
        report.verdicts.append(
            (
                turn.case.id,
                judge_grounded(
                    provider, item, turn.case.wrong_answer, turn.text, turn.case.working
                ),
            )
        )
    return report


# --- level discriminability --------------------------------------------------

_RANKING_SYSTEM = (
    "You compare two tutor replies to the same student mistake and say which "
    "one gives more of the answer away. Judge only how much is revealed, not "
    "which reply is better."
)


class MoreRevealing(BaseModel):
    """Which of two replies gives more away."""

    choice: Literal["first", "second", "same"] = Field(
        description="The reply that reveals more of the answer, or 'same'."
    )


@dataclass
class RankingReport:
    """Whether the assigned support levels are visible in the text.

    ``check_fading`` proves ``hint_level`` is non-increasing in mastery. That is
    a property of the function. This asks whether a learner would experience the
    difference, which is the claim the function is standing in for.
    """

    #: (case id, lower level, higher level, judged correctly)
    pairs: list[tuple[str, str, str, bool | None]] = field(default_factory=list)

    @property
    def scored(self) -> list[tuple[str, str, str, bool]]:
        return [(a, b, c, d) for a, b, c, d in self.pairs if d is not None]

    @property
    def agreement(self) -> float:
        scored = self.scored
        return sum(1 for *_, ok in scored if ok) / len(scored) if scored else 0.0


def rank_levels(
    provider: LLMProvider, domain: Domain, turns: Sequence[Turn]
) -> RankingReport:
    """Compare turns on the same case that were assigned different levels.

    Presented in the order they were generated in and judged by name rather than
    by position, so the report says which level was picked rather than which
    slot.
    """
    report = RankingReport()
    order = {level.label: int(level) for level in HintLevel}
    grouped: dict[tuple[str, str, str], list[Turn]] = {}
    for turn in turns:
        if turn.is_fallback:
            continue
        key = (turn.case.item_id, turn.case.misconception_id, turn.move)
        grouped.setdefault(key, []).append(turn)

    for group in grouped.values():
        ordered = sorted(group, key=lambda t: order[t.level])
        if len(ordered) < 2:
            continue
        low, high = ordered[0], ordered[-1]
        if order[low.level] == order[high.level]:
            continue
        item = domain.items.get(low.case.item_id)
        prompt = (
            f"Exercise: {' '.join(item.prompt.split())}\n"
            f"The student wrote: {low.case.wrong_answer}\n\n"
            f"First reply: {low.text}\n"
            f"Second reply: {high.text}\n\n"
            f"Which reply gives more of the answer away?"
        )
        try:
            reply = complete(provider, prompt, MoreRevealing, system=_RANKING_SYSTEM)
            judged = reply.choice == "second"
        except MalformedResponse:
            judged = None
        report.pairs.append((low.case.id, low.level, high.level, judged))
    return report
