"""The simulated learner's profile, persisted. **Ground truth.**

Kept in its own module, reached through its own class, for one reason: the
diagnostic agent exists to *infer* what this holds, so an accuracy figure
measured with it in scope would mean nothing.

In memory that separation is structural — a ``MisconceptionProfile`` is a Python
object the agents are never handed, and
``tests/integration/test_no_back_channel.py`` walks every agent's attributes to
prove it. Persistence threatens that: profiles and beliefs share one database
file, so anything holding a connection could read the answer. The split is
therefore restored in code rather than assumed:

* :class:`~agent_newton.store.learners.LearnerStore` cannot read profiles at all.
  It has no method that touches the table.
* This module is imported by the runner and the evaluation harness. A test
  asserts that nothing under ``core/agents/`` imports it, directly or otherwise.

The database is one file because the user wants one place to query. The
separation that matters is which code can reach which table, and that is what is
enforced.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent_newton.core.simulator.profile import MisconceptionProfile


class ProfileStore:
    """Reads and writes simulated learners' misconception profiles.

    Anything constructing this is claiming the right to see ground truth. That
    is true of the cohort runner, which must carry a learner's true state across
    sessions, and of the evaluation harness, which scores inference against it.
    It is true of no agent.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> ProfileStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def save(self, session_id: int, profile: MisconceptionProfile) -> None:
        """Store the profile as it stood at the end of a session.

        Both ``firing`` and ``initial`` are kept. Remediation lowers ``firing``
        and forgetting raises it, so without the starting point there is nothing
        to report remediation as a proportion *of*.
        """
        self._db.execute(
            "INSERT OR REPLACE INTO profile (session_id, firing, initial) VALUES (?, ?, ?)",
            (session_id, json.dumps(profile.firing), json.dumps(dict(profile.initial))),
        )
        self._db.commit()

    def latest(self, learner_id: str, arm: str) -> MisconceptionProfile | None:
        """The profile this learner last left behind in this arm, or None.

        Keyed on the arm for the same reason state is: the same person under two
        architectures is two histories, and a profile remediated in one arm must
        never be inherited by the other.
        """
        row = self._db.execute(
            "SELECT p.firing, p.initial, ss.learner_id FROM profile p "
            "JOIN session ss ON ss.session_id = p.session_id "
            "WHERE ss.learner_id = ? AND ss.arm = ? "
            "ORDER BY ss.seq DESC LIMIT 1",
            (learner_id, arm),
        ).fetchone()
        if row is None:
            return None
        return MisconceptionProfile(
            learner_id=learner_id,
            seed=0,
            firing=json.loads(row["firing"]),
            initial=json.loads(row["initial"]),
        )
