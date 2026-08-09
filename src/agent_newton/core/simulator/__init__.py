"""The simulated learner.

A deterministic rule engine owns behaviour; a surface renderer owns only
phrasing. See :mod:`agent_newton.core.simulator.engine`.
"""

from agent_newton.core.simulator.engine import SimulatedLearner, SimulatedStep
from agent_newton.core.simulator.profile import MisconceptionProfile, sample_profile
from agent_newton.core.simulator.surface import SurfaceRenderer, SymbolicSurface

__all__ = [
    "MisconceptionProfile",
    "SimulatedLearner",
    "SimulatedStep",
    "SurfaceRenderer",
    "SymbolicSurface",
    "sample_profile",
]
