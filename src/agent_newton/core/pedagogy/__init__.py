"""Instructional rules as checkable predicates.

See :mod:`agent_newton.core.pedagogy.policy`.
"""

from agent_newton.core.pedagogy.policy import (
    BAND_MEMBERSHIP,
    ERROR_FIRST,
    HintLevel,
    Support,
    TeachingStyle,
    TutorMove,
    Violation,
    check_fading,
    check_move,
    check_support_fading,
    hint_level,
    may_select,
    move_for,
    next_required_move,
    should_explain,
    style_for,
    support_at_presentation,
)

__all__ = [
    "BAND_MEMBERSHIP",
    "ERROR_FIRST",
    "HintLevel",
    "Support",
    "TeachingStyle",
    "TutorMove",
    "Violation",
    "check_fading",
    "check_move",
    "check_support_fading",
    "hint_level",
    "may_select",
    "move_for",
    "next_required_move",
    "should_explain",
    "style_for",
    "support_at_presentation",
]
