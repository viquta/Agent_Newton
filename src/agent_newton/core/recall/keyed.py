"""Recall by the key that is already there: the concept."""

from __future__ import annotations

from typing import Sequence

from agent_newton.core.state.schema import Utterance


class KeyedRecall:
    """Everything this learner said about this concept, most recent first.

    Deterministic, free, and it needs no model, no index and no dependency. It
    is the recall the system already half has — ``FullStateView.said_about``
    does exactly this within a sitting, and ``LearnerStore.utterances`` stores it
    across them.

    ⚠️ What it cannot do is the thing the ideas note asks for. A learner who
    wrote *"wait, what is a gradient"* while working on ``limit_concept`` and
    meets gradients again under ``power_rule`` has said something that bears on
    the new question, and this returns nothing — the key does not match, so the
    remark does not exist. Whether that matters often enough to justify an index
    is what the comparison is for.
    """

    label = "keyed"

    def about(
        self,
        corpus: Sequence[Utterance],
        concept_id: str,
        query: str = "",
        limit: int = 3,
    ) -> Sequence[Utterance]:
        # `query` is deliberately unused: this strategy is defined by keying on
        # the concept and nothing else, and reading the query here would make
        # the comparison against the embedded one measure two changes at once.
        on_topic = [u for u in corpus if u.concept_id == concept_id]
        return tuple(reversed(on_topic[-limit:]))
