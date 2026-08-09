"""Bayesian Knowledge Tracing (Corbett & Anderson, 1995).

Four parameters per concept: the prior that a learner already knows it, the
chance of learning it at each opportunity, and the chances of a lucky guess or a
careless slip. An observation revises the posterior; the transition then carries
it forward.

Only correct and incorrect observations update anything. A response the verifier
could not read says nothing about mastery — see :func:`observe`.
"""

from __future__ import annotations

from agent_newton.config import BKTConfig

#: Posteriors are clamped away from 0 and 1. At exactly 1.0 no later evidence
#: can ever move the estimate, so a single lucky streak would permanently freeze
#: a concept as mastered and the frontier could never reopen it.
_EPSILON = 1e-6


def posterior(prior: float, correct: bool, params: BKTConfig) -> float:
    """Revise the mastery estimate given one observation.

    Bayes on the evidence alone, before the learning transition.
    """
    if correct:
        knew = prior * (1.0 - params.p_slip)
        did_not = (1.0 - prior) * params.p_guess
    else:
        knew = prior * params.p_slip
        did_not = (1.0 - prior) * (1.0 - params.p_guess)

    total = knew + did_not
    if total <= 0.0:
        # Unreachable while guess + slip < 1, which the config validator
        # enforces. Returning the prior keeps this total rather than dividing
        # by zero if that guarantee is ever weakened.
        return prior
    return knew / total


def transition(posterior_value: float, params: BKTConfig) -> float:
    """Apply the learning transition: what is not yet known may be learned."""
    return posterior_value + (1.0 - posterior_value) * params.p_transit


def _clamp(value: float) -> float:
    return min(1.0 - _EPSILON, max(_EPSILON, value))


def observe(prior: float, correct: bool, params: BKTConfig) -> float:
    """Full update for one graded observation.

    Call this **only** for evidence. A verdict whose ``is_evidence`` is False —
    an unreadable response, or one the verifier could not decide — is a
    measurement failure, not information about the learner, and passing it here
    would move the estimate on the basis of our own inability to measure.
    """
    return _clamp(transition(posterior(prior, correct, params), params))


def initial(params: BKTConfig) -> float:
    """Mastery estimate for a concept with no observations yet."""
    return _clamp(params.p_init)
