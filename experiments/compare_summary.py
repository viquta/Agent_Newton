"""Compare a re-run's summary against the stored one.

    uv run python experiments/compare_summary.py \
        --rerun results/reruns/paired_calculus/summary.json \
        --stored results/paired_calculus/summary.json

Reads what the experiments already wrote and computes nothing. The question it
answers is whether a re-run landed on the stored numbers, which two JSON files
side by side do not make obvious.

Provenance differs by construction — a re-run has its own run ids and its own
timestamps — so those are skipped rather than reported. Everything else is
compared: numbers within a tolerance, other values exactly.

Exit status is 0 when the summaries agree and 1 when they do not, so it can gate
a script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterator, Sequence

#: Keys naming a run, a directory or a moment. Two runs of the same experiment
#: differ in all of them and agree on every number, which is the distinction
#: this tool exists to draw.
PROVENANCE_KEYS = frozenset(
    {
        "arms",
        "config",
        "created_at",
        "generated_at",
        "out",
        "run_id",
        "written_to",
    }
)

#: A run id anywhere else — inside a list, or under a key not named above.
RUN_ID = re.compile(r"^20\d{6}T\d{6}_")

#: Relative, and generous enough for a different CPU to reach the same result by
#: a different order of floating-point operations. Deterministic reruns on one
#: machine agree exactly.
TOLERANCE = 1e-9


def leaves(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Every scalar in the structure, with the path that reaches it."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in PROVENANCE_KEYS:
                continue
            yield from leaves(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaves(value, f"{path}[{index}]")
    else:
        yield path, node


def agrees(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def compare(
    rerun: Any,
    stored: Any,
    tolerance: float = TOLERANCE,
    ignore: Sequence[str] = (),
) -> tuple[list[str], int]:
    """Differences as readable lines, and how many values were checked.

    A value that looks like a run id is skipped wherever it appears: it is the
    one thing guaranteed to differ between two runs of the same experiment.

    ``ignore`` drops whole paths by prefix, for a part of the stored summary the
    re-run deliberately did not produce — the propagation study's model-backed
    condition, when it is run without a model. Reporting those as differences
    every time would bury a difference that means something.
    """
    left = dict(leaves(rerun))
    right = dict(leaves(stored))

    differences: list[str] = []
    checked = 0

    for path in sorted(set(left) | set(right)):
        if any(path == skip or path.startswith(f"{skip}.") for skip in ignore):
            continue
        a, b = left.get(path, _MISSING), right.get(path, _MISSING)
        if isinstance(a, str) and RUN_ID.match(a):
            continue
        if isinstance(b, str) and RUN_ID.match(b):
            continue
        if a is _MISSING:
            differences.append(f"  {path}: absent from the re-run, stored {b!r}")
            continue
        if b is _MISSING:
            differences.append(f"  {path}: {a!r} in the re-run, absent from stored")
            continue
        checked += 1
        if not agrees(a, b, tolerance):
            differences.append(f"  {path}: {a!r} — stored {b!r}")

    return differences, checked


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover — appears only in a message
        return "<absent>"


_MISSING = _Missing()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--stored", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument(
        "--ignore",
        default="",
        help="Comma-separated key paths the re-run did not produce, e.g. "
        "'conditions.llm' for a propagation run made without a model.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Say nothing when the two agree. Differences are always printed.",
    )
    args = parser.parse_args()

    for path in (args.rerun, args.stored):
        if not path.exists():
            parser.error(f"no summary at {path}")

    ignore = [part.strip() for part in args.ignore.split(",") if part.strip()]
    differences, checked = compare(
        json.loads(args.rerun.read_text()),
        json.loads(args.stored.read_text()),
        args.tolerance,
        ignore,
    )

    if differences:
        print(f"\ndiffers from {args.stored} in {len(differences)} of {checked} values:")
        print("\n".join(differences))
        return 1

    if not args.quiet:
        note = f", {len(ignore)} path(s) not compared" if ignore else ""
        print(f"\nreproduces {args.stored} — {checked} values, all identical{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
