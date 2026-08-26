"""Finding what a learner said before, when it bears on what they are doing now.

The ideas note asks for this directly: *"two weeks ago student_x asked: why is
there a 'd' in dx/dy? After receiving teaching-point_z, they still seem to
misunderstand it in today's session."* Answering that needs their own words back,
and the words are already stored — ``LearnerStore.utterances`` has existed since
sessions could span sittings, and its docstring says planning is meant to read
it. Nothing reads it.

**Two implementations, built to be compared rather than argued about.**
``bonus_lesson_idea.md`` closed retrieval for *lesson content*, and that argument
holds: fifteen lessons keyed by concept id is a dict lookup, and an approximate
index would put a non-deterministic step in the instructional path for nothing.
This is the other case the same note names as the one that would earn an index —
a corpus nobody keyed, queried in the learner's own words — so it is measured
instead of assumed.

⚠️ **A corpus is passed in; nothing here opens a database.** ``core/`` does not
import ``store/`` and this does not begin. It is the same split ``build_session``
already makes for state and profile: the caller holds the store because the
caller is the thing that has one.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from agent_newton.core.state.schema import Utterance


@runtime_checkable
class Recall(Protocol):
    """Picks out the utterances that bear on what the learner is doing now."""

    @property
    def label(self) -> str:
        """What this is, for a run manifest. Two strategies are not one result."""
        ...

    def about(
        self,
        corpus: Sequence[Utterance],
        concept_id: str,
        query: str = "",
        limit: int = 3,
    ) -> Sequence[Utterance]:
        """The most relevant ``limit`` utterances, most relevant first.

        ``concept_id`` is what the learner is working on and ``query`` is what
        they just said or were just asked. An implementation may use either,
        both, or neither — which is precisely the difference being measured.

        Returning fewer than ``limit`` is an ordinary answer. A recaller that
        pads its results to fill the quota scores better on recall and worse on
        precision, which is the wrong trade for a tutor prompt: an irrelevant
        remark handed back as context is worse than silence, because the tutor
        will try to use it.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Separate so recall can be tested without one."""

    @property
    def label(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """One vector per text, in order."""
        ...
