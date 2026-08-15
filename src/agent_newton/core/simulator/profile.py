"""Per-learner misconception profiles.

A profile is the ground truth of a simulated learner: which misconceptions they
hold, and how strongly. It is generated from a seed, so the same seed always
produces the same learner — which is what allows one learner to be run through
both architectures and compared against themselves.

**No agent may read a profile.** The tutor, diagnostic and planner see only the
interaction history through their state view. The profile is available to the
evaluation harness, which scores the diagnostic agent's inference against the
label that was actually injected.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Mapping

from agent_newton.config import SimulatorConfig
from agent_newton.domains.base import ConceptGraph, MisconceptionCatalogue


def _seed_for(seed: int, learner_id: str) -> int:
    """A stable integer seed from the run seed and the learner's identity."""
    digest = hashlib.sha256(f"{seed}:{learner_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class MisconceptionProfile:
    """What a simulated learner gets wrong, and how reliably.

    ``firing`` maps a misconception id to the probability it fires on an item
    that can elicit it. Remediation lowers those probabilities; nothing else
    changes them.
    """

    learner_id: str
    seed: int
    firing: dict[str, float]
    #: Initial probabilities, retained so remediation can be reported as a
    #: proportion of what was there to begin with.
    initial: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.initial:
            self.initial = dict(self.firing)

    def holds(self, misconception_id: str) -> bool:
        return misconception_id in self.firing

    def probability(self, misconception_id: str) -> float:
        return self.firing.get(misconception_id, 0.0)

    def remediate(self, misconception_id: str, factor: float) -> bool:
        """Weaken one misconception. Returns whether it applied.

        This is the only route by which a learner improves, and it is why
        diagnostic accuracy has consequences: a hint aimed at a misconception
        the learner does not hold changes nothing at all.
        """
        if misconception_id not in self.firing:
            return False
        self.firing[misconception_id] *= factor
        return True

    def remediation_ratio(self) -> float:
        """How far the profile has been reduced, in [0, 1].

        1.0 means every misconception has been driven to nothing; 0.0 means no
        progress. Reported per learner as a system-level outcome.
        """
        start = sum(self.initial.values())
        if start <= 0.0:
            return 1.0
        return 1.0 - (sum(self.firing.values()) / start)

    def snapshot(self) -> dict[str, float]:
        return dict(self.firing)


def solidity(
    concept_id: str,
    profile: MisconceptionProfile,
    catalogue: MisconceptionCatalogue,
    graph: ConceptGraph,
) -> float:
    """How sound this learner's foundations are under ``concept_id``, in [0, 1].

    1.0 when nothing they hold sits anywhere in the concept's prerequisite
    closure — there is nothing shaky underneath it. It falls toward 0.0 the more
    of their *unremediated* misconception mass lies below, and rises again as
    those are taught, which is what makes teaching a prerequisite first do
    something for what depends on it.

    ⚠️ **Read from the profile, never from the learner model.** The profile is
    what is true; mastery is what the system believes. A mechanism keyed on the
    belief would let the coupled arm's own estimate drive the learner's
    behaviour, and it would win the comparison by construction rather than by
    routing — the conclusion assumed rather than measured. Nothing here touches
    a posterior, and nothing may.

    Measured against the closure rather than the immediate parents: "the
    foundations" of a concept is everything it rests on, and stopping one level
    up would call a concept solid whose grandparent is not.

    Most learners hold two misconceptions out of a catalogue of fifteen, so this
    is 1.0 for most concepts and bites exactly where the learner is actually
    weak.
    """
    beneath = graph.all_prerequisites(concept_id)
    if not beneath:
        return 1.0

    held = [
        misconception_id
        for misconception_id in profile.firing
        if catalogue.get(misconception_id).concept_id in beneath
    ]
    started = sum(profile.initial.get(m, 0.0) for m in held)
    if started <= 0.0:
        # Nothing of theirs lies underneath, so there is nothing to be shaky.
        return 1.0
    remaining = sum(profile.firing.get(m, 0.0) for m in held)
    return max(0.0, 1.0 - remaining / started)


def sample_profile(
    learner_id: str,
    seed: int,
    catalogue: MisconceptionCatalogue,
    config: SimulatorConfig,
) -> MisconceptionProfile:
    """Draw a learner from the domain's catalogue.

    Sampling is over sorted ids so the draw depends only on the seed and the
    catalogue's content, never on dictionary ordering.
    """
    rng = random.Random(_seed_for(seed, learner_id))
    available = sorted(catalogue.ids())
    if not available:
        raise ValueError("cannot sample a learner from an empty misconception catalogue")

    count = min(config.misconceptions_per_learner, len(available))
    chosen = rng.sample(available, count)

    low, high = config.p_fire_range
    firing = {m: rng.uniform(low, high) for m in sorted(chosen)}
    return MisconceptionProfile(learner_id=learner_id, seed=seed, firing=firing)
