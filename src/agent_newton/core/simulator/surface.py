"""How a decided step is phrased.

The rule engine owns *what* the learner does. A surface renderer owns only how
it is said. Keeping the two apart is what makes the simulator's ground truth
exact while still allowing naturalistic dialogue: a model may rewrite the
phrasing, never the answer.

``SymbolicSurface`` is the identity — the engine's response passed through
unchanged. It removes models from the simulator entirely, which is what makes
runs fast and exactly reproducible.

The model-backed renderer arrives with the provider layer. Its contract is
already fixed here: it may return different words, and if it returns something
that no longer means the same thing, the run is wrong in a way no downstream
component can detect. Hence :func:`SurfaceRenderer.render` returns prose for a
step the engine has already decided, and never receives the correct answer to
compare against.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_newton.core.simulator.engine import SimulatedStep
from agent_newton.domains.base import Item


@runtime_checkable
class SurfaceRenderer(Protocol):
    """Turns a decided step into what the learner says."""

    def render(self, item: Item, step: SimulatedStep) -> str: ...


class SymbolicSurface:
    """Pass the engine's response through verbatim.

    The default. No model is invoked, so a whole cohort runs in seconds and
    reproduces exactly from its seed.
    """

    def render(self, item: Item, step: SimulatedStep) -> str:  # noqa: ARG002
        return step.response
