"""Instructional rules as checkable predicates.

See :mod:`agent_newton.core.pedagogy.policy`.
"""

from agent_newton.core.pedagogy.policy import (
    BAND_MEMBERSHIP,
    ERROR_FIRST,
    HintLevel,
    TutorMove,
    Violation,
    check_fading,
    check_move,
    hint_level,
    may_select,
    next_required_move,
)

__all__ = [
    "BAND_MEMBERSHIP",
    "ERROR_FIRST",
    "HintLevel",
    "TutorMove",
    "Violation",
    "check_fading",
    "check_move",
    "hint_level",
    "may_select",
    "next_required_move",
]
