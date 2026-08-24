"""Render a file from ``docs/`` to the terminal.

    python render_doc.py                # list what is there
    python render_doc.py architecture   # render docs/architecture.md

Used by the container's ``arch`` and ``docs`` verbs. Reads only.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

DOCS = Path("docs")

#: Reading order, for the listing. Anything in ``docs/`` not named here is
#: appended after these, so a new file appears without editing this list.
ORDER = [
    "architecture",
    "domain_interface",
    "learner_state",
    "pedagogy",
    "arbitration_policy",
    "configuration",
    "docker",
]


def available() -> list[Path]:
    found = sorted(DOCS.glob("*.md"))
    ranked = {name: index for index, name in enumerate(ORDER)}
    return sorted(found, key=lambda p: (ranked.get(p.stem, len(ORDER)), p.stem))


def first_heading(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def main() -> int:
    console = Console()
    if not DOCS.is_dir():
        console.print(f"[red]no {DOCS}/ directory here[/red]")
        return 1

    names = available()
    if len(sys.argv) < 2:
        console.print("[bold]Technical reference[/bold]\n")
        for path in names:
            console.print(f"  [cyan]{path.stem:<20}[/cyan] {first_heading(path)}")
        console.print("\n[dim]newton arch <name>[/dim]")
        return 0

    wanted = sys.argv[1].removesuffix(".md")
    match = next((p for p in names if p.stem == wanted), None)
    if match is None:
        console.print(f"[red]no docs/{wanted}.md[/red] — one of:")
        console.print("  " + ", ".join(p.stem for p in names))
        return 1

    console.print(Markdown(match.read_text(encoding="utf-8")))
    console.print(f"\n[dim]{match}[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
