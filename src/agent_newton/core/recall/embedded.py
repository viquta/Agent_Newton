"""Recall by what the words mean, over a corpus small enough to search exactly."""

from __future__ import annotations

import math
from typing import Sequence

from agent_newton.core.recall.base import Embedder
from agent_newton.core.state.schema import Utterance


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Similarity of two vectors, 0.0 when either has no length.

    Written out rather than pulled in: it is four lines, and a dependency added
    for four lines is a dependency to keep up to date forever.
    """
    dot = sum(x * y for x, y in zip(a, b))
    length = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / length if length else 0.0


class EmbeddedRecall:
    """Ranks the whole corpus against the query, by meaning rather than by key.

    ⚠️ **Exact cosine over everything, and no index.** That is what makes this
    answerable rather than merely plausible, and it is the direct reply to the
    objection recorded in ``bonus_lesson_idea.md`` — that retrieval "would put a
    non-deterministic step into the instructional path of a project whose rolls
    are hashes and whose model cache is keyed by prompt".

    The non-determinism in that objection belongs to *approximate* search. A
    learner's corpus is tens to low hundreds of strings, so scanning all of it is
    affordable, and scanning all of it is deterministic: same corpus, same query,
    same embedding model, same ranking, every time. The remaining variable is the
    embedding model, and that goes in the manifest like every other model.

    ⚠️ It is not free, and the cost is the honest argument against it. Every
    utterance and every query is a model call — cached, but a call the keyed
    strategy never makes. Whether the recall it buys is worth that is the
    measurement, not a matter of taste.

    ``threshold`` drops matches below a similarity, so a corpus with nothing
    relevant in it returns nothing rather than the least irrelevant thing it
    could find. A recaller that always returns its quota scores well on recall
    and badly on precision, and for a tutor prompt that is the wrong trade: an
    unrelated remark handed back as context is worse than silence, because the
    tutor will try to use it.
    """

    def __init__(self, embedder: Embedder, threshold: float = 0.5) -> None:
        self._embedder = embedder
        self._threshold = threshold

    @property
    def label(self) -> str:
        return f"embedded/{self._embedder.label}@{self._threshold:g}"

    def about(
        self,
        corpus: Sequence[Utterance],
        concept_id: str,
        query: str = "",
        limit: int = 3,
    ) -> Sequence[Utterance]:
        # No query is no question. Falling back to the concept here would make
        # this the keyed strategy under another name at exactly the moment the
        # comparison is being taken.
        if not query.strip() or not corpus:
            return ()

        vectors = self._embedder.embed([query, *(u.text for u in corpus)])
        asked, stored = vectors[0], vectors[1:]
        scored = [
            (cosine(asked, vector), index)
            for index, vector in enumerate(stored)
            if cosine(asked, vector) >= self._threshold
        ]
        # Ties break on the later utterance, so a learner who said the same
        # thing twice gets the more recent one back. Stable either way, which is
        # what keeps the whole thing reproducible.
        scored.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
        return tuple(corpus[index] for _, index in scored[:limit])
